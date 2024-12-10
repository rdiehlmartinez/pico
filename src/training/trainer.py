"""
Pico Language Model Trainer

This Trainer implements a minimalistic end-to-end training pipeline of the Pico language model with
distributed training support via Lightning Fabric. It provides a modular and configurable training
pipeline with the features:

    - Configuration Management: YAML-based configuration for all aspects of training
    - Distributed Training: Multi-GPU support via Lightning Fabric
    - Checkpointing: Regular model saving and training state recovery
    - Evaluation: Periodic model evaluation on validation datasets
    - Logging: Comprehensive metric tracking and experiment monitoring
    - Optimization: Support for gradient accumulation, clipping, and LR scheduling

"""

import logging
import lightning as L
import torch
import torch.nn.functional as F
import os
from lightning.fabric.utilities.rank_zero import rank_zero_only

from typing import Dict, Any

from src.model import Pico

from src.training.utils import (
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
from src.checkpointing import (
    load_checkpoint,
    save_checkpoint,
    save_evaluation_results,
    # compute_learning_dynamics_metrics,
)

from src.evaluation import run_evaluation


class Trainer:
    def __init__(self, config_path: str):
        """
        Initializes the Trainer class. This Trainer class implements a `train` method, which is the
        main entry point for training the Pico model. Before calling `train`, the Trainer class
        initializes the following:

            - Configuration loading and validation
            - Model, optimizer, and dataset setup
            - Logging and experiment tracking setup
            - Checkpoint management

        Args:
            config_path (str): Path to the YAML configuration file containing any overrides.
        """

        # Setup Config
        self.sub_configs = initialize_configuration(config_path)
        self.data_config = self.sub_configs["data"]
        self.model_config = self.sub_configs["model"]
        self.training_config = self.sub_configs["training"]
        self.evaluation_config = self.sub_configs["evaluation"]

        # Setup Run Directory
        initialize_run_dir(self.training_config, self.evaluation_config)

        # Setup Logger
        self.logger, self.experiment_tracker = initialize_logging(self.training_config)

        # Setup Fabric
        self.fabric = initialize_fabric(self.training_config, self.experiment_tracker)

        # Setup Dataset, Tokenizer, and Dataloader
        self.train_dataset = initialize_dataset(self.data_config)
        self.tokenizer = initialize_tokenizer(self.data_config)
        self.train_dataloader = initialize_dataloader(
            self.data_config, self.train_dataset
        )

        # Setup Model, Optimizer, and Dataloaders
        self.model = Pico(self.model_config, self.fabric)
        self.optimizer = initialize_optimizer(self.model, self.training_config)
        self.lr_scheduler = initialize_lr_scheduler(
            self.optimizer, self.training_config
        )

        # Wrap with Fabric
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
        self.train_dataloader = self.fabric.setup_dataloaders(self.train_dataloader)

        # Setup Checkpointing
        initialize_checkpointing(self.training_config)
        L.seed_everything(42, verbose=False)

        self.should_load_checkpoint = (
            self.training_config.checkpointing.load_checkpoint_path is not None
            or self.training_config.checkpointing.load_latest_checkpoint
        )
        self.should_start_from_scratch = not self.should_load_checkpoint

        if self.should_load_checkpoint:
            resume_checkpoint = load_checkpoint(
                self.training_config,
                self.fabric,
                self.model,
                self.optimizer,
                self.lr_scheduler,
                self.train_dataloader,
            )

            if resume_checkpoint is None:
                if self.training_config.checkpointing.load_latest_checkpoint:
                    self.log(
                        "No checkpoint found at specified path. Starting from latest checkpoint.",
                        level=logging.WARNING,
                    )
                    self.should_start_from_scratch = True
                else:
                    raise ValueError("No checkpoint found at specified path. Exiting.")
            else:
                (
                    self.model,
                    self.optimizer,
                    self.lr_scheduler,
                    self.train_start_step,
                    self.train_iterator,
                ) = resume_checkpoint

        # Setup Training Start Step and Iterator
        if self.should_start_from_scratch:
            self.train_start_step = 0
            self.train_iterator = iter(self.train_dataloader)

    def train(self) -> None:
        """
        Executes the core training loop logic. First saves out a checkpoint and evaluates the model.
        This is followed by the `_training_loop` method, which runs the main training loop.
        """

        # Save Initial Checkpoint
        save_checkpoint(
            self.sub_configs,
            self.fabric,
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.tokenizer,
            self.train_start_step,
            upload_logs=False,
        )

        # Save Initial Evaluation Results
        if self.evaluation_config:
            if self.train_start_step == 0:
                evaluation_results = run_evaluation(self.evaluation_config)
                self._log_evaluation_results(evaluation_results, self.train_start_step)
                save_evaluation_results(
                    self.evaluation_config,
                    self.fabric,
                    evaluation_results,
                    self.train_start_step,
                )
            else:
                # NOTE: If the run crashed while evaluating, we need to restart the evaluation
                eval_results_path = os.path.join(
                    self.evaluation_config.eval_results_dir,
                    f"step_{self.train_start_step}.json",
                )
                if not os.path.exists(eval_results_path):
                    evaluation_results = run_evaluation(self.evaluation_config)
                    self._log_evaluation_results(
                        evaluation_results, self.train_start_step
                    )
                    save_evaluation_results(
                        self.evaluation_config,
                        self.fabric,
                        evaluation_results,
                        self.train_start_step,
                    )

        if self.train_start_step < self.training_config.training_steps:
            self.log(f"Training from step {self.train_start_step}")
            final_step = self._training_loop()

        # Handle checkpointing and final evaluation
        if final_step % self.training_config.checkpointing.save_every_n_steps != 0:
            self.log(f"Saving final checkpoint at step {final_step}")
            save_checkpoint(
                self.sub_configs,
                self.fabric,
                self.model,
                self.optimizer,
                self.lr_scheduler,
                self.tokenizer,
                final_step,
            )

        # Final evaluation
        self.log("Starting Final Evaluation!")
        evaluation_results = run_evaluation(self.evaluation_config)
        if evaluation_results is not None:
            self._log_evaluation_results(evaluation_results, final_step)

    def _training_loop(self) -> int:
        """Runs the core training loop and orchestrates:
            - Gradient accumulation
            - Gradient clipping
            - Periodic model evaluation and checkpointing
            - Learning rate scheduling
            - Logging of training metrics including loss and learning rate
            - Handling of infinite/NaN losses

        Returns:
            int: The final training step reached;
                NOTE: this is used to determine if the training loop finished successfully.
        """
        # Setup training loop variables
        gradient_step = self.train_start_step

        # NOTE: these are used to compute the average loss over a training interval (more accurate
        # than using the loss at the end of the interval)
        interval_loss = torch.tensor(0.0, device=self.fabric.device)
        interval_steps = torch.tensor(0, device=self.fabric.device)
        interval_inf_or_nan_count = torch.tensor(0, device=self.fabric.device)

        # Training loop
        for batch_idx, batch in enumerate(
            self.train_iterator, start=self.train_start_step
        ):
            ########################################################
            #
            # Forward Pass
            #
            ########################################################

            _input_ids = torch.tensor(batch["input_ids"], device=self.fabric.device)
            input_ids = _input_ids[:, :-1]
            labels = _input_ids[:, 1:]

            # Forward pass
            model_output, _ = self.model(input_ids)
            model_output = model_output.transpose(1, 2)

            ########################################################
            #
            # Gradient accumulation
            #
            ########################################################

            should_accumulate_gradients = (
                batch_idx + 1
            ) % self.training_config.optimization.gradient_accumulation_steps != 0

            with self.fabric.no_backward_sync(
                self.model, enabled=should_accumulate_gradients
            ):
                loss = F.cross_entropy(model_output, labels)
                self.fabric.backward(
                    loss / self.training_config.optimization.gradient_accumulation_steps
                )

                if torch.isnan(loss) or torch.isinf(loss):
                    interval_inf_or_nan_count += 1
                else:
                    interval_loss += loss.item()
                    interval_steps += 1

            # NOTE: if we are not accumulating gradients, we should skip the logging and optimization steps
            if should_accumulate_gradients:
                continue

            ########################################################
            #
            # Logging
            #
            ########################################################

            if gradient_step % self.training_config.logging.log_every_n_steps == 0:
                self._log_training_metrics(
                    interval_loss=interval_loss,
                    interval_steps=interval_steps,
                    interval_inf_or_nan_count=interval_inf_or_nan_count,
                    gradient_step=gradient_step,
                )
                interval_loss = torch.tensor(0.0, device=self.fabric.device)
                interval_steps = torch.tensor(0, device=self.fabric.device)
                interval_inf_or_nan_count = torch.tensor(0, device=self.fabric.device)

            ########################################################
            #
            # Learning Dynamics Checkpointing
            #
            ########################################################

            if self.training_config.checkpointing.learning_dynamics.enabled:
                # compute_learning_dynamics_metrics(
                #     self.training_config.checkpointing.learning_dynamics,
                #     self.fabric,
                #     self.model,
                #     batch,
                # )
                pass

            ########################################################
            #
            # Optimization step
            #
            ########################################################

            self.fabric.clip_gradients(
                self.model,
                self.optimizer,
                max_norm=self.training_config.optimization.max_norm,
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step()

            gradient_step += 1

            ########################################################
            #
            # Training Checkpointing and evaluation
            #
            ########################################################

            if (
                gradient_step % self.training_config.checkpointing.save_every_n_steps
                == 0
            ):
                self.log(f"Saving checkpoint at step {gradient_step}")
                save_checkpoint(
                    self.sub_configs,
                    self.fabric,
                    self.model,
                    self.optimizer,
                    self.lr_scheduler,
                    self.tokenizer,
                    gradient_step,
                )

                if self.evaluation_config:
                    evaluation_results = run_evaluation(
                        self.evaluation_config, self.fabric
                    )
                    if evaluation_results is not None:
                        self._log_evaluation_results(evaluation_results, gradient_step)
                        save_evaluation_results(
                            self.evaluation_config,
                            self.fabric,
                            evaluation_results,
                            gradient_step,
                        )

            # Break if we've reached training steps
            if gradient_step >= self.training_config.training_steps:
                break

        return gradient_step

    def _log_training_metrics(
        self,
        interval_loss: torch.Tensor,
        interval_steps: torch.Tensor,
        interval_inf_or_nan_count: torch.Tensor,
        gradient_step: int,
    ):
        """
        Gathers together the training metrics computed accross all processes in distributed training
        (NOTE: also works for single process training), and logs them to the experiment tracking system
        and console.
        """
        gathered_interval_loss = self.fabric.all_reduce(
            interval_loss, reduce_op="sum"
        ).item()
        gathered_interval_inf_or_nan_count = self.fabric.all_reduce(
            interval_inf_or_nan_count, reduce_op="sum"
        ).item()
        gathered_interval_steps = self.fabric.all_reduce(
            interval_steps, reduce_op="sum"
        ).item()

        avg_loss = (
            gathered_interval_loss / gathered_interval_steps
            if gathered_interval_steps > 0
            else float("inf")
        )

        self.fabric.log("train/loss", avg_loss, step=gradient_step)
        self.fabric.log(
            "trainer/inf_or_nan_count",
            gathered_interval_inf_or_nan_count,
            step=gradient_step,
        )
        self.fabric.log(
            "trainer/learning_rate",
            self.lr_scheduler.get_last_lr()[0],
            step=gradient_step,
        )

        self.log(
            f"Step {gradient_step} Loss: {avg_loss}, Inf/NaN count: {gathered_interval_inf_or_nan_count}"
        )

    def _log_evaluation_results(
        self, evaluation_results: Dict[str, Any], gradient_step: int
    ):
        """Log model evaluation metrics to experiment tracking system and console."""
        self.log(f"Step {gradient_step} -- Evaluation Results:")
        for metric, result in evaluation_results.items():
            self.log(f"  {metric}: {result}")
            self.fabric.log(f"eval/{metric}", result, step=gradient_step)

    @rank_zero_only
    def log(self, msg: str, level: int = logging.INFO) -> None:
        """Log messages only from rank zero process."""
        self.logger.log(level, msg)
