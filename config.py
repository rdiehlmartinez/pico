"""Defines the set of hyperparameters to be specified in the config file."""

from dataclasses import dataclass, field
from typing import Optional

VOCAB_SIZE = 32000
MAX_SEQ_LEN = 2048
BATCH_SIZE = 1024

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
    act_hidden_dim: int = 3072

@dataclass 
class _AttentionConfig:
    n_heads: int = 12
    n_kv_heads: Optional[int] = 4

@dataclass
class ModelConfig:
    d_model: int = 768
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
    name: str = "wikitext2"

@dataclass
class _DataLoaderConfig:
    batch_size: int = BATCH_SIZE
    max_seq_len: int = MAX_SEQ_LEN

@dataclass
class _TokenizerConfig:
    vocab_size: int = VOCAB_SIZE
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

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
    gradient_accumulation_steps: int = 1

@dataclass
class _LoggingConfig:
    experiment_tracker: Optional[str] = "wandb"
    wandb_project: Optional[str] = "pico"
    wandb_entity: Optional[str] = "pico-lm"

    log_every_n_steps: int = 10

@dataclass
class _CheckpointConfig:
    save_every_n_steps: int = 20

    # Path to load a checkpoint from local path or automatically load the latest locally-saved checkpoint
    # NOTE: if both are provided, local_checkpoint_path takes priority.
    load_checkpoint_path: Optional[str] = None
    load_latest_checkpoint: bool = False

    # HuggingFace Hub Configs - set to None to not push to HuggingFace Hub
    # Should be in the format of <(username or )>/<repo_name>, e.g. pico-lm/pico-7b
    hf_repo_id: Optional[str] = "pico-lm/demo"

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