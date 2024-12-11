"""
Training Config

Specifies the hyperparameters for the training process, i.e. the optimizer, learning rate, etc.
"""

from dataclasses import dataclass, field
from typing import Optional, List

from ._constants import RUNS_DIR, CHECKPOINT_DIR, LOG_DIR, FABRIC_CHECKPOINT_DIR


@dataclass
class _FabricConfig:
    num_nodes: int = 1
    num_devices: int = 1
    precision: str = "16-mixed"
    accelerator: str = "cuda"


@dataclass
class _OptimizationConfig:
    # Optimizer
    optimizer: str = "adamw"
    lr: float = 1e-5

    # Learning Rate Scheduler
    lr_scheduler: str = "linear_with_warmup"
    lr_warmup_steps: int = 20

    # Gradient Clipping
    max_norm: float = 1.0

    # Gradient Accumulation
    gradient_accumulation_steps: int = 32


@dataclass
class _LoggingConfig:
    experiment_tracker: Optional[str] = "wandb"
    wandb_project: Optional[str] = "pico"
    wandb_entity: Optional[str] = "pico-lm"

    log_every_n_steps: int = 10


@dataclass
class _LearningDynamicsConfig:
    activation_data: str = "pico-lm/pico-7b-activations"

    # Specify what layers to compute learning dynamics for
    activation_layers: List[str] = field(default_factory=lambda: [])


@dataclass
class _CheckpointConfig:
    save_every_n_steps: int = 20

    # Path to load a checkpoint from a local path
    load_checkpoint_path: Optional[str] = None

    # Whether to load the latest checkpoint (Takes priority over load_checkpoint_path)
    load_latest_checkpoint: bool = False

    # HuggingFace Hub Configs - set to None to not push to HuggingFace Hub
    # Should be in the format of <(username or )>/<repo_name>, e.g. pico-lm/pico-7b
    save_checkpoint_repo_id: Optional[str] = "pico-lm/demo"


@dataclass
class TrainingConfig:
    run_name: Optional[str] = None

    fabric: _FabricConfig = field(default_factory=_FabricConfig)
    optimization: _OptimizationConfig = field(default_factory=_OptimizationConfig)

    logging: _LoggingConfig = field(default_factory=_LoggingConfig)
    checkpointing: _CheckpointConfig = field(default_factory=_CheckpointConfig)

    learning_dynamics: _LearningDynamicsConfig = field(
        default_factory=_LearningDynamicsConfig
    )

    strategy: str = "deepspeed"
    training_steps: int = 100

    # Directories used to store training runs, checkpoints, logs, and evaluation results
    runs_dir: str = RUNS_DIR
    checkpoints_dir: str = CHECKPOINT_DIR
    logs_dir: str = LOG_DIR
    fabric_checkpoint_dir: str = FABRIC_CHECKPOINT_DIR
