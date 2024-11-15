"""
Welcome to the Pico Config File!

This is where you can specify the hyperparameters for the Pico model, the dataset, the training
process, evaluation yada yada.

As with anything else in Pico, this file is designed to be as flexible as possible. If you find
yourself wanting to add a new hyperparameter, go for it! If you want to use hydra for hierarchical
configs, no problem -- this is very easy to do given that the default configs are all dataclasses.

Some things to NOTE:
- All hyperparameters are initialized with default values, which can be overridden by the
  config file.
- The default vocab size is set to the size of the OLMo tokenizer.
"""

from dataclasses import dataclass, field
from typing import Optional

VOCAB_SIZE = 50304
MAX_SEQ_LEN = 2048
BATCH_SIZE = 1024
GRADIENT_ACCUMULATION_STEPS = (
    128  # NOTE: Play with this to make the batch size fit in memory.
)

# N.B. The effective batch size is BATCH_SIZE // GRADIENT_ACCUMULATION_STEPS.

########################################################
#
# Model Config
#
########################################################


@dataclass
class _RoPEConfig:
    theta: float = 10000.0


@dataclass
class _RMSNormConfig:
    eps: float = 1e-6


@dataclass
class _ActivationConfig:
    act_hidden_dim: int = 768


@dataclass
class _AttentionConfig:
    n_heads: int = 12
    n_kv_heads: Optional[int] = 4


@dataclass
class ModelConfig:
    d_model: int = 192
    n_layers: int = 12

    vocab_size: int = VOCAB_SIZE
    batch_size: int = BATCH_SIZE
    max_seq_len: int = MAX_SEQ_LEN

    attention: _AttentionConfig = field(default_factory=_AttentionConfig)
    activation: _ActivationConfig = field(default_factory=_ActivationConfig)
    norm: _RMSNormConfig = field(default_factory=_RMSNormConfig)
    position_emb: _RoPEConfig = field(default_factory=_RoPEConfig)


########################################################
#
# Data Config
#
########################################################


@dataclass
class _DatasetConfig:
    name: str = "pico-lm/pretokenized-dolma"


@dataclass
class _DataLoaderConfig:
    batch_size: int = BATCH_SIZE // GRADIENT_ACCUMULATION_STEPS
    max_seq_len: int = MAX_SEQ_LEN


class _TokenizerConfig:
    name: str = "allenai/OLMo-7B-0724-hf"
    vocab_size: int = VOCAB_SIZE


@dataclass
class DataConfig:
    dataset: _DatasetConfig = field(default_factory=_DatasetConfig)
    dataloader: _DataLoaderConfig = field(default_factory=_DataLoaderConfig)
    tokenizer: _TokenizerConfig = field(default_factory=_TokenizerConfig)


########################################################
#
# Training Configs
#
########################################################


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

    strategy: str = "deepspeed"
    training_steps: int = 100


########################################################
#
# Evaluation Configs
#
########################################################


@dataclass
class EvaluationConfig:
    eval_every_n_steps: int = 100
    eval_batch_size: int = 1024
