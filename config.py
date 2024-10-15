"""Defines the set of hyperparameters to be specified in the config file."""

from dataclasses import dataclass
from typing import Optional
import yaml

########################################################
#
# Pico Model Config
#
########################################################

@dataclass
class RoPEConfig:
    theta: float = 10000.0

@dataclass
class RMSNormConfig:
    eps: float = 1e-6

@dataclass
class ActivationConfig:
    act_hidden_dim: int = 3072

@dataclass
class ModelConfig:
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: Optional[int] = 4
    max_seq_len: int = 2048
    max_batch_size: int = 1024

@dataclass
class TokenizerConfig:
    vocab_size: int = 32000
    max_seq_len: int = 2048
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

@dataclass
class PicoConfig:
    model: ModelConfig = ModelConfig()
    activation: ActivationConfig = ActivationConfig()
    norm: RMSNormConfig = RMSNormConfig()
    position_emb: RoPEConfig = RoPEConfig()

    tokenizer: TokenizerConfig = TokenizerConfig()


########################################################
#
# Training Configs
#
########################################################

@dataclass
class FabricConfig:
    num_nodes: int = 1
    num_devices: int = 1
    precision: str = "16-mixed"
    accelerator: str = "cuda"

@dataclass
class OptimizationConfig:
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
class LoggingConfig:
    experiment_tracker: Optional[str] = "wandb"
    wandb_project: Optional[str] = "pico"
    wandb_entity: Optional[str] = "pico-lm"

    log_every_n_steps: int = 10

@dataclass
class CheckpointConfig:
    save_every_n_steps: int = 20

    # Path to load a checkpoint from
    load_checkpoint_path: Optional[str] = None

    # HuggingFace Hub Configs - set to None to not push to HuggingFace Hub
    # Should be in the format of <(username or )>/<repo_name>
    # e.g. pico-lm/pico-7b
    # hf_repo_id: Optional[str] = None
    hf_repo_id: Optional[str] = "pico-lm/demo"


@dataclass
class TrainingConfig:
    run_name: Optional[str] = None

    fabric: FabricConfig = FabricConfig()
    optimization: OptimizationConfig = OptimizationConfig()

    logging: LoggingConfig = LoggingConfig()
    checkpointing: CheckpointConfig = CheckpointConfig()

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
    

########################################################
#
# Helper Functions
#
########################################################
 
def update_config_from_yaml(config: PicoConfig | TrainingConfig | EvaluationConfig, yaml_path: str) -> PicoConfig | TrainingConfig | EvaluationConfig:
    """
    Update the config object with the values in the given yaml file.
    NOTE: we don't check for unknown parameters.
    """

    with open(yaml_path, 'r') as f:
        updates = yaml.safe_load(f)

    unknown_params = []

    def _update_config_recursively(config, updates):
        for key, value in updates.items():
            if isinstance(value, dict):
                _update_config_recursively(getattr(config, key), value)
            elif hasattr(config, key):
                setattr(config, key, value)
            else:
                unknown_params.append(key)

    _update_config_recursively(config, updates)

    if unknown_params:
        raise ValueError(f"Overriding parameters not specified in the config: {unknown_params}")

    return config