"""
This is the core training loop: the glue that holds everything together.

Here we use our initialization utils to setup all the components of the training loop (the model,
the optimizer, the dataloader, etc.) and then we enter the training loop.

The training loop itself is meant to be simple and easy to understand.

Here's what we do in the training loop:

1. Compute the model's output
2. Compute the loss
3. Do backprop
4. Update the model's weights
5. Log the loss (and maybe evaluate the model)

There's also gradient accumulation, gradient clipping, learning rate scheduling, checkpointing,
and a few other things, but these are made to be as simple as possible.
"""

import click
import logging
import lightning as L
import torch
import torch.nn.functional as F
from lightning.fabric.utilities.rank_zero import rank_zero_only

from model import Pico

from utils.initialization import (
    initialize_run_dir,
    initialize_fabric,
    initialize_config,
    initialize_dataset,
    initialize_dataloader,
    initialize_lr_scheduler,
    initialize_checkpointing,
    initialize_logging,
    initialize_optimizer,
)
from utils.checkpointing import load_checkpoint, save_checkpoint, save_config


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
        $ python train.py --config_path configs/training_config.yaml
    """

    ########################################################
    #
    # Pre-Training Setup (aka. boilerplate)
    #
    ########################################################

    # ---- Setup Configs ---- #
    data_config, model_config, training_config, evaluation_config = initialize_config(
        config_path
    )

    # ---- Setup Run Directory ---- #
    initialize_run_dir(training_config)

    # ---- Setup Logger ---- #
    logger, experiment_tracker = initialize_logging(training_config)

    @rank_zero_only
    def log(msg, level=logging.INFO):
        logger.log(level, msg)

    # ---- Setup Fabric ---- #
    fabric = initialize_fabric(training_config, experiment_tracker)

    # ---- Setup Dataset, Tokenizer, and Dataloader ---- #

    train_dataset = initialize_dataset(data_config)
    train_dataloader = initialize_dataloader(data_config, train_dataset)

    # ---- Setup Model, Optimizer, and Dataloaders ---- #
    model = Pico(model_config, fabric)
    optimizer = initialize_optimizer(model, training_config)
    lr_scheduler = initialize_lr_scheduler(optimizer, training_config)

    # ---- Wrapping with Fabric ---- #
    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    # ---- Load Checkpoint (if specified) ---- #
    initialize_checkpointing(training_config)
    L.seed_everything(42, verbose=False)

    should_load_checkpoint = (
        training_config.checkpointing.load_checkpoint_path is not None
        or training_config.checkpointing.load_latest_checkpoint
    )
    should_start_from_scratch = not should_load_checkpoint

    if should_load_checkpoint:
        resume_checkpoint = load_checkpoint(
            fabric, training_config, model, optimizer, lr_scheduler, train_dataloader
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

    if should_start_from_scratch:
        train_start_step = 0
        train_iterator = iter(train_dataloader)
        save_config(fabric, training_config, model_config, evaluation_config)

    save_checkpoint(
        fabric,
        training_config,
        model,
        optimizer,
        lr_scheduler,
        train_start_step,
        upload_logs=False,
    )

    ########################################################
    #
    # Training Commences!
    #
    ########################################################

    if train_start_step < training_config.training_steps:
        log(f"Training from step {train_start_step}")

    gradient_step = train_start_step
    # Loss tracking over log_every_n_steps interval
    interval_loss = torch.tensor(0.0, device=fabric.device)
    interval_steps = torch.tensor(0, device=fabric.device)
    interval_inf_or_nan_count = torch.tensor(0, device=fabric.device)

    for batch_idx, batch in enumerate(train_iterator, start=train_start_step):
        _input_ids = batch["input_ids"]
        _input_ids = torch.tensor(_input_ids, device=fabric.device)

        # NOTE: The model is autoregressive, so the labels are just the next token in the sequence.
        # Thus the sequence length we use in the model is 1 token less than the length of the
        # sequence stored in the dataset (i.e. 2049 tokens -> 2048 tokens).
        input_ids = _input_ids[:, :-1]
        labels = _input_ids[:, 1:]

        model_output = model(input_ids).transpose(1, 2)

        # ---- Gradient Accumulation (if enabled) ---- #
        should_accumulate_gradients = (
            batch_idx + 1
        ) % training_config.optimization.gradient_accumulation_steps != 0

        with fabric.no_backward_sync(model, enabled=should_accumulate_gradients):
            # Compute loss
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

        """
        NOTE: This is important! Code after this point only runs after gradient_accumulation_steps.
        If you've set gradient_accumulation_steps to 1, then this code will run every step. 
        """

        # ---- Logging ---- #

        if gradient_step % training_config.logging.log_every_n_steps == 0:
            # NOTE: This is annoying, I'm sorry I know, but we need the all_reduce to get the
            #       average loss across all the devices (if you're using multiple GPUs).

            gathered_interval_loss = fabric.all_reduce(
                interval_loss, reduce_op="sum"
            ).item()
            gathered_interval_inf_or_nan_count = fabric.all_reduce(
                interval_inf_or_nan_count, reduce_op="sum"
            ).item()
            gathered_interval_steps = fabric.all_reduce(
                interval_steps, reduce_op="sum"
            ).item()

            if gathered_interval_steps > 0:
                avg_loss = gathered_interval_loss / gathered_interval_steps
            else:
                avg_loss = float("inf")  # or some other appropriate value

            interval_loss = torch.tensor(0.0, device=fabric.device)
            interval_steps = torch.tensor(0, device=fabric.device)
            interval_inf_or_nan_count = torch.tensor(0, device=fabric.device)

            fabric.log("loss", avg_loss, step=gradient_step)
            fabric.log(
                "inf_or_nan_count",
                gathered_interval_inf_or_nan_count,
                step=gradient_step,
            )
            fabric.log(
                "learning_rate", lr_scheduler.get_last_lr()[0], step=gradient_step
            )
            log(
                f"Step {gradient_step} Loss: {avg_loss}, Inf/NaN count: {gathered_interval_inf_or_nan_count}"
            )

        # ---- Gradient Step ---- #

        fabric.clip_gradients(
            model, optimizer, max_norm=training_config.optimization.max_norm
        )
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        gradient_step += 1

        # ---- Checkpointing ---- #
        # Maybe save checkpoint
        if gradient_step % training_config.checkpointing.save_every_n_steps == 0:
            log(f"Saving checkpoint at step {gradient_step}")
            save_checkpoint(
                fabric, training_config, model, optimizer, lr_scheduler, gradient_step
            )

        # --- Evaluation --- #
        if gradient_step % evaluation_config.eval_every_n_steps == 0:
            log("Starting Evaluation!")

        # --- Break Training Condition --- #
        if gradient_step == training_config.training_steps:
            # Save final checkpoint if we didn't save it at already
            if gradient_step % training_config.checkpointing.save_every_n_steps != 0:
                log(f"Saving final checkpoint at step {gradient_step}")
                save_checkpoint(
                    fabric,
                    training_config,
                    model,
                    optimizer,
                    lr_scheduler,
                    gradient_step,
                )
            break

    if gradient_step < training_config.training_steps:
        log(f"Training finished early at step {gradient_step}", level=logging.WARNING)
        save_checkpoint(
            fabric, training_config, model, optimizer, lr_scheduler, gradient_step
        )

    # --- Final Evaluation --- #
    log("Starting Final Evaluation!")


if __name__ == "__main__":
    main()
