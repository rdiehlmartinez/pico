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
        self.configs = initialize_configuration(config_path)

        # Setup Run Directory
        initialize_run_dir(self.configs["checkpointing"])

        # Setup Logger
        self.logger, self.experiment_tracker = initialize_logging(
            self.configs["logging"], self.configs["checkpointing"]
        )

        # Setup Fabric
        self.fabric = initialize_fabric(
            self.configs["training"], self.experiment_tracker
        )

        # Setup Dataset, Tokenizer, and Dataloader
        self.train_dataset = initialize_dataset(self.configs["data"])
        self.tokenizer = initialize_tokenizer(self.configs["data"])
        self.train_dataloader = initialize_dataloader(
            self.configs["data"], self.train_dataset
        )

        # Setup Model, Optimizer, and Dataloaders
        self.model = Pico(self.configs["model"], self.fabric)
        self.optimizer = initialize_optimizer(self.configs["training"], self.model)
        self.lr_scheduler = initialize_lr_scheduler(
            self.configs["training"], self.optimizer
        )

        # Wrap with Fabric
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
        self.train_dataloader = self.fabric.setup_dataloaders(self.train_dataloader)

        # Setup Checkpointing
        initialize_checkpointing(self.configs["checkpointing"])
        L.seed_everything(42, verbose=False)

        # Helper flag to determine if we should load a checkpoint
        self.should_load_checkpoint = (
            self.configs["checkpointing"].training.load_checkpoint_path is not None
            or self.configs["checkpointing"].training.load_latest_checkpoint
        )
        self.should_start_from_scratch = not self.should_load_checkpoint

        # Possibly load a checkpoint
        if self.should_load_checkpoint:
            resume_checkpoint = load_checkpoint(
                self.configs["checkpointing"],
                self.fabric,
                self.model,
                self.optimizer,
                self.lr_scheduler,
                self.train_dataloader,
            )

            if resume_checkpoint is None:
                if self.configs["checkpointing"].training.load_latest_checkpoint:
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

        # Helper flag to determine if we should evaluate the model
        self.should_evaluate = (
            self.configs["evaluation"].evaluation_metrics is not None
            and len(self.configs["evaluation"].evaluation_metrics) > 0
        )

    def train(self) -> None:
        """Execute the main training workflow.

        This method orchestrates the complete training process by:
        1. Creating an initial checkpoint to save the starting state and evaluate the model as a
            baseline
        2. Running the main training loop via `_training_loop`
        3. Handling final checkpointing and evaluation

        The training progress is tracked through checkpoints and evaluations
        at intervals specified in the configuration.

        Returns:
            None
        """

        # Save Initial Checkpoint; NOTE: if the checkpoint already exists, this performs a no-op
        save_checkpoint(
            self.configs,
            self.fabric,
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.tokenizer,
            self.train_start_step,
            upload_logs=False,
        )

        # Save Initial Evaluation Results
        if self.should_evaluate:
            if self.train_start_step == 0:
                evaluation_results = run_evaluation(
                    self.configs["evaluation"],
                    self.configs["checkpointing"],
                    self.fabric,
                )
                self._log_evaluation_results(evaluation_results, self.train_start_step)
                save_evaluation_results(
                    self.configs["checkpointing"],
                    self.fabric,
                    evaluation_results,
                    self.train_start_step,
                )
            else:
                # NOTE: If the run crashed while evaluating, we need to restart the evaluation
                eval_results_path = os.path.join(
                    self.configs["checkpointing"].evaluation.eval_results_dir,
                    f"step_{self.train_start_step}.json",
                )
                if not os.path.exists(eval_results_path):
                    evaluation_results = run_evaluation(
                        self.configs["evaluation"],
                        self.configs["checkpointing"],
                        self.fabric,
                    )
                    self._log_evaluation_results(
                        evaluation_results, self.train_start_step
                    )
                    save_evaluation_results(
                        self.configs["checkpointing"],
                        self.fabric,
                        evaluation_results,
                        self.train_start_step,
                    )

        if self.train_start_step < self.configs["training"].max_steps:
            self.log(f"✨ Starting training from step {self.train_start_step}")
            final_step = self._training_loop()
        else:
            final_step = self.train_start_step

        # Handle checkpointing and final evaluation
        if final_step % self.configs["checkpointing"].save_every_n_steps != 0:
            self.log(f"Step {final_step} -- 💾 Saving Final Checkpoint")
            save_checkpoint(
                self.configs,
                self.fabric,
                self.model,
                self.optimizer,
                self.lr_scheduler,
                self.tokenizer,
                final_step,
            )

            # Final evaluation
            if self.should_evaluate:
                evaluation_results = run_evaluation(
                    self.configs["evaluation"],
                    self.configs["checkpointing"],
                    self.fabric,
                )
                self._log_evaluation_results(evaluation_results, final_step)
                save_evaluation_results(
                    self.configs["checkpointing"],
                    self.fabric,
                    evaluation_results,
                    final_step,
                )

        self.log(f"🎉 Training complete! Final step: {final_step}")

        if final_step < self.configs["training"].max_steps:
            self.log(
                f"\t Note: Training stopped before max steps ({self.configs['training'].max_steps})",
                level=logging.WARNING,
            )

    def _training_loop(self) -> int:
        """Execute the main training loop.

        This method orchestrates the core training loop and includes the following features:
            - Gradient accumulation
            - Gradient clipping
            - Periodic model evaluation and checkpointing
            - Learning rate scheduling
            - Logging of training metrics including loss and learning rate
            - Handling of infinite/NaN losses

        Returns:
            int: The final step count reached during training.
                NOTE: A complete training run should match the configured max_steps.
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

            should_accumulate_gradients = (batch_idx + 1) % self.configs[
                "training"
            ].optimization.gradient_accumulation_steps != 0

            with self.fabric.no_backward_sync(
                self.model, enabled=should_accumulate_gradients
            ):
                loss = F.cross_entropy(model_output, labels)
                self.fabric.backward(
                    loss
                    / self.configs["training"].optimization.gradient_accumulation_steps
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

            if gradient_step % self.configs["logging"].log_every_n_steps == 0:
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

            # if self.configs["checkpointing"].learning_dynamics.enabled:
            #     # compute_learning_dynamics_metrics(
            #     #     self.training_config.checkpointing.learning_dynamics,
            #     #     self.fabric,
            #     #     self.model,
            #     #     batch,
            #     # )
            #     pass

            ########################################################
            #
            # Optimization step
            #
            ########################################################

            self.fabric.clip_gradients(
                self.model,
                self.optimizer,
                max_norm=self.configs["training"].optimization.max_norm,
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

            if gradient_step % self.configs["checkpointing"].save_every_n_steps == 0:
                self.log(f"Step {gradient_step} -- 💾 Saving Checkpoint")
                save_checkpoint(
                    self.configs,
                    self.fabric,
                    self.model,
                    self.optimizer,
                    self.lr_scheduler,
                    self.tokenizer,
                    gradient_step,
                )

                if self.should_evaluate:
                    evaluation_results = run_evaluation(
                        self.configs["evaluation"],
                        self.configs["checkpointing"],
                        self.fabric,
                    )
                    if evaluation_results is not None:
                        self._log_evaluation_results(evaluation_results, gradient_step)
                        save_evaluation_results(
                            self.configs["checkpointing"],
                            self.fabric,
                            evaluation_results,
                            gradient_step,
                        )

            # Break if we've reached training steps
            if gradient_step >= self.configs["training"].max_steps:
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
        Gathers together the training metrics computed across all processes in distributed training
        and logs them in a tree-style format.
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

        # Log to console in tree format
        self.log(f"Step {gradient_step} -- 🔄 Training Metrics")
        self.log(f"├── Loss: {avg_loss:.4f}")
        self.log(f"├── Learning Rate: {self.lr_scheduler.get_last_lr()[0]:.2e}")
        self.log(f"└── Inf/NaN count: {gathered_interval_inf_or_nan_count}")

    def _log_evaluation_results(
        self, evaluation_results: Dict[str, Any], gradient_step: int
    ):
        """Log model evaluation metrics to experiment tracking system and console."""
        self.log(f"Step {gradient_step} -- 📊 Evaluation Results")
        for i, (metric, result) in enumerate(evaluation_results.items()):
            prefix = "└──" if i == len(evaluation_results) - 1 else "├──"
            self.log(f"{prefix} {metric}: {result}")
            self.fabric.log(f"eval/{metric}", result, step=gradient_step)

    @rank_zero_only
    def log(self, msg: str, level: int = logging.INFO) -> None:
        """Log messages only from rank zero process."""
        self.logger.log(level, msg)
