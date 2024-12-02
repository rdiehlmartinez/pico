"""
Pico Language Model Training Script

This script implements end-to-end training of the Pico language model with distributed
training support via Lightning Fabric. It provides a modular and configurable training
pipeline with the following key features:

Key Components:
    - Configuration Management: YAML-based configuration for all aspects of training
    - Distributed Training: Multi-GPU support via Lightning Fabric
    - Checkpointing: Regular model saving and training state recovery
    - Evaluation: Periodic model evaluation on validation datasets
    - Logging: Comprehensive metric tracking and experiment monitoring
    - Optimization: Support for gradient accumulation, clipping, and LR scheduling

Usage:
    python train.py --config_path configs/my_config.yaml

Configuration:
    The training script expects a configuration file with the following sections:
    - data: Dataset and tokenizer settings
    - model: Model architecture parameters
    - training: Training loop, optimization, and logging settings
    - evaluation: Evaluation dataset and metrics settings

There's also gradient accumulation, gradient clipping, learning rate scheduling, checkpointing,
and a few other things, but these are made to be as simple as possible.
"""

import click
import logging
import lightning as L
import torch
import torch.nn.functional as F
import os
from lightning.fabric.utilities.rank_zero import rank_zero_only

from typing import Optional, Iterator, Callable, Dict, Any
from transformers import PreTrainedTokenizer

from model import Pico

from utils import EVAL_RESULTS_DIR

from utils.initialization import (
    initialize_run_dir,
    initialize_fabric,
    initialize_configuration,
    initialize_dataset,
    initialize_tokenizer,
    initialize_dataloader,
    initialize_lr_scheduler,
    initialize_checkpointing,
    initialize_logging,
    initialize_optimizer,
)
from utils.checkpointing import (
    load_checkpoint,
    save_checkpoint,
    save_evaluation_results,
)

from utils.evaluation import run_evaluation


def training_loop(
    sub_configs: Dict[str, Any],
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_iterator: Iterator,
    train_start_step: int,
    tokenizer: PreTrainedTokenizer,
    log_fn: Optional[Callable[[str], None]] = None,
) -> int:
    """Execute the core training loop with gradient accumulation and evaluation.

    Args:
        sub_configs: Dictionary containing configuration for training, evaluation
        fabric: Lightning Fabric instance for distributed training
        model: The neural network model to train
        optimizer: The optimizer for updating model parameters
        lr_scheduler: Learning rate scheduler for dynamic learning rate adjustment
        train_iterator: Iterator over training batches
        train_start_step: Step number to resume training from
        tokenizer: Tokenizer for processing text inputs
        log_fn: Optional logging function for outputting messages

    Returns:
        int: The final training step reached (we do this to return )

    The training loop implements several key features:
    1. Gradient accumulation for effective larger batch sizes
    2. Gradient clipping to prevent exploding gradients
    3. Periodic model evaluation and checkpointing
    4. Learning rate scheduling
    5. Logging of training metrics including loss and learning rate
    6. Handling of infinite/NaN losses
    """

    training_config = sub_configs["training"]
    evaluation_config = sub_configs["evaluation"]

    # Setup training loop variables
    gradient_step = train_start_step
    interval_loss = torch.tensor(0.0, device=fabric.device)
    interval_steps = torch.tensor(0, device=fabric.device)
    interval_inf_or_nan_count = torch.tensor(0, device=fabric.device)

    # Training loop
    for batch_idx, batch in enumerate(train_iterator, start=train_start_step):
        _input_ids = torch.tensor(batch["input_ids"], device=fabric.device)
        input_ids = _input_ids[:, :-1]
        labels = _input_ids[:, 1:]

        # Forward pass
        model_output, _ = model(input_ids)
        model_output = model_output.transpose(1, 2)

        # Gradient accumulation
        should_accumulate_gradients = (
            batch_idx + 1
        ) % training_config.optimization.gradient_accumulation_steps != 0

        with fabric.no_backward_sync(model, enabled=should_accumulate_gradients):
            loss = F.cross_entropy(model_output, labels)
            fabric.backward(
                loss / training_config.optimization.gradient_accumulation_steps
            )

            if torch.isnan(loss) or torch.isinf(loss):
                interval_inf_or_nan_count += 1
            else:
                interval_loss += loss.item()
                interval_steps += 1

        if should_accumulate_gradients:
            continue

        # Logging
        if gradient_step % training_config.logging.log_every_n_steps == 0:
            _log_training_metrics(
                fabric=fabric,
                interval_loss=interval_loss,
                interval_steps=interval_steps,
                interval_inf_or_nan_count=interval_inf_or_nan_count,
                gradient_step=gradient_step,
                lr_scheduler=lr_scheduler,
                log_fn=log_fn,
            )
            interval_loss = torch.tensor(0.0, device=fabric.device)
            interval_steps = torch.tensor(0, device=fabric.device)
            interval_inf_or_nan_count = torch.tensor(0, device=fabric.device)

        # Optimization step
        fabric.clip_gradients(
            model, optimizer, max_norm=training_config.optimization.max_norm
        )
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        gradient_step += 1

        # Checkpointing and evaluation
        if gradient_step % training_config.checkpointing.save_every_n_steps == 0:
            log_fn(f"Saving checkpoint at step {gradient_step}")
            save_checkpoint(
                sub_configs,
                fabric,
                model,
                optimizer,
                lr_scheduler,
                tokenizer,
                gradient_step,
            )

            if evaluation_config:
                evaluation_results = run_evaluation(evaluation_config)
                if evaluation_results is not None:
                    _log_evaluation_results(
                        fabric, evaluation_results, gradient_step, log_fn
                    )
                    save_evaluation_results(
                        evaluation_config, fabric, evaluation_results, gradient_step
                    )

        # Break if we've reached training steps
        if gradient_step >= training_config.training_steps:
            break

    return gradient_step


def _log_training_metrics(
    fabric,
    interval_loss,
    interval_steps,
    interval_inf_or_nan_count,
    gradient_step,
    lr_scheduler,
    log_fn,
):
    """Log training metrics across all processes in distributed training.

    Args:
        fabric: Lightning Fabric instance for distributed operations
        interval_loss: Accumulated loss over the logging interval
        interval_steps: Number of steps in the current logging interval
        interval_inf_or_nan_count: Count of infinite or NaN losses in interval
        gradient_step: Current training step
        lr_scheduler: Learning rate scheduler for current learning rate
        log_fn: Function for logging messages

    The function:
    1. Aggregates metrics across all processes using all_reduce
    2. Calculates average loss for the interval
    3. Logs loss, inf/nan counts, and learning rate to tracking system
    4. Outputs human-readable progress message
    """
    gathered_interval_loss = fabric.all_reduce(interval_loss, reduce_op="sum").item()
    gathered_interval_inf_or_nan_count = fabric.all_reduce(
        interval_inf_or_nan_count, reduce_op="sum"
    ).item()
    gathered_interval_steps = fabric.all_reduce(interval_steps, reduce_op="sum").item()

    avg_loss = (
        gathered_interval_loss / gathered_interval_steps
        if gathered_interval_steps > 0
        else float("inf")
    )

    fabric.log("train/loss", avg_loss, step=gradient_step)
    fabric.log(
        "trainer/inf_or_nan_count",
        gathered_interval_inf_or_nan_count,
        step=gradient_step,
    )
    fabric.log(
        "trainer/learning_rate", lr_scheduler.get_last_lr()[0], step=gradient_step
    )

    if log_fn:
        log_fn(
            f"Step {gradient_step} Loss: {avg_loss}, Inf/NaN count: {gathered_interval_inf_or_nan_count}"
        )


def _log_evaluation_results(fabric, evaluation_results, gradient_step, log_fn):
    """Log model evaluation metrics to tracking system and console.

    Args:
        fabric: Lightning Fabric instance for logging
        evaluation_results: Dictionary of evaluation metrics and their values
        gradient_step: Current training step when evaluation was performed
        log_fn: Function for logging messages

    The function logs each evaluation metric:
    1. To the experiment tracking system (e.g., WandB) with 'eval/' prefix
    2. To console in a human-readable format
    3. Maintains consistent metric naming conventions
    """

    # pretty print out the evaluation results
    if log_fn:
        log_fn(f"Step {gradient_step} -- Evaluation Results:")
    for metric, result in evaluation_results.items():
        if log_fn:
            log_fn(f"  {metric}: {result}")
        fabric.log(f"eval/{metric}", result, step=gradient_step)


@click.command()
@click.option(
    "--config_path",
    type=str,
)
def main(config_path: str):
    """
    Execute the core training loop for the Pico language model.

    This function orchestrates the entire training process through the following stages:
    1. Initialization
        - Configuration loading and validation
        - Model, optimizer, and dataset setup
        - Logging and experiment tracking setup
        - Checkpoint management

    2. Training Loop
        - Forward pass and loss computation
        - Gradient accumulation and optimization
        - Learning rate scheduling
        - Progress logging and metrics tracking
        - Periodic model checkpointing
        - Regular model evaluation

    3. Finalization
        - Final model checkpoint
        - Final evaluation
        - Cleanup and logging completion

    Args:
        config_path (str): Path to the YAML configuration file containing any overrides.

    Example:
        $ python train.py --config_path configs/my_config.yaml
    """

    ########################################################
    #
    # Training Setup (aka. boilerplate)
    #
    ########################################################

    # Setup Config
    sub_configs = initialize_configuration(config_path)

    data_config = sub_configs["data"]
    model_config = sub_configs["model"]
    training_config = sub_configs["training"]
    evaluation_config = sub_configs["evaluation"]

    # Setup Run Directory
    initialize_run_dir(training_config, evaluation_config)

    # Setup Logger
    logger, experiment_tracker = initialize_logging(training_config)

    # NOTE: We create a logger that only logs if the rank is zero
    @rank_zero_only
    def log(msg, level=logging.INFO):
        logger.log(level, msg)

    # Setup Fabric
    fabric = initialize_fabric(training_config, experiment_tracker)

    # Setup Dataset, Tokenizer, and Dataloader
    train_dataset = initialize_dataset(data_config)
    tokenizer = initialize_tokenizer(data_config)
    train_dataloader = initialize_dataloader(data_config, train_dataset)

    # Setup Model, Optimizer, and Dataloaders
    model = Pico(model_config, fabric)
    optimizer = initialize_optimizer(model, training_config)
    lr_scheduler = initialize_lr_scheduler(optimizer, training_config)

    # Wrap with Fabric
    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    # Setup Checkpointing
    initialize_checkpointing(training_config)
    L.seed_everything(42, verbose=False)

    should_load_checkpoint = (
        training_config.checkpointing.load_checkpoint_path is not None
        or training_config.checkpointing.load_latest_checkpoint
    )
    should_start_from_scratch = not should_load_checkpoint

    if should_load_checkpoint:
        resume_checkpoint = load_checkpoint(
            training_config, fabric, model, optimizer, lr_scheduler, train_dataloader
        )

        if resume_checkpoint is None:
            if training_config.checkpointing.load_latest_checkpoint:
                log(
                    "No checkpoint found at specified path. Starting from latest checkpoint.",
                    level=logging.WARNING,
                )
                should_start_from_scratch = True
            else:
                raise ValueError("No checkpoint found at specified path. Exiting.")
        else:
            model, optimizer, lr_scheduler, train_start_step, train_iterator = (
                resume_checkpoint
            )

    # Setup Training Start Step and Iterator
    if should_start_from_scratch:
        train_start_step = 0
        train_iterator = iter(train_dataloader)

    # Save Initial Checkpoint
    save_checkpoint(
        sub_configs,
        fabric,
        model,
        optimizer,
        lr_scheduler,
        tokenizer,
        train_start_step,
        upload_logs=False,
    )

    # Save Initial Evaluation Results
    if evaluation_config:
        if train_start_step == 0:
            evaluation_results = run_evaluation(evaluation_config)
            _log_evaluation_results(fabric, evaluation_results, train_start_step, log)
            save_evaluation_results(
                evaluation_config, fabric, evaluation_results, train_start_step
            )
        else:
            # NOTE: If the run crashed while evaluating, we need to restart the evaluation
            eval_results_path = os.path.join(
                EVAL_RESULTS_DIR, f"step_{train_start_step}.json"
            )
            if not os.path.exists(eval_results_path):
                evaluation_results = run_evaluation(evaluation_config)
                _log_evaluation_results(
                    fabric, evaluation_results, train_start_step, log
                )
                save_evaluation_results(
                    evaluation_config, fabric, evaluation_results, train_start_step
                )

    ########################################################
    #
    # Training Loop
    #
    ########################################################

    if train_start_step < training_config.training_steps:
        log(f"Training from step {train_start_step}")

        final_step = training_loop(
            sub_configs,
            fabric=fabric,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_iterator=train_iterator,
            train_start_step=train_start_step,
            tokenizer=tokenizer,
            log_fn=log,
        )

    ########################################################
    #
    # Finalization
    #
    ########################################################

    # Handle checkpointing and final evaluation
    if final_step % training_config.checkpointing.save_every_n_steps != 0:
        log(f"Saving final checkpoint at step {final_step}")
        save_checkpoint(
            sub_configs, fabric, model, optimizer, lr_scheduler, tokenizer, final_step
        )

    # Final evaluation
    log("Starting Final Evaluation!")
    evaluation_results = run_evaluation(evaluation_config)
    if evaluation_results is not None:
        _log_evaluation_results(fabric, evaluation_results, final_step, log)


if __name__ == "__main__":
    main()
