from config import PicoConfig

import yaml
import lightning as L
import torch

from config import PicoConfig, TrainingConfig, EvaluationConfig

def initialize_fabric(config: TrainingConfig, logger: L.Logger):
    """
    Setup the lightning fabric with the given config. 
    """

    fabric = L.Fabric(
        accelerator=config.fabric.accelerator,
        precision=config.fabric.precision,
        devices=config.fabric.num_devices,
        num_nodes=config.fabric.num_nodes,
        logger=logger
    )

    fabric.launch()

    return fabric


def initialize_config(config_override, config_type):
    """
    Setup the config with the given config_override. 
    """

    if config_type == "model":
        config_cls = PicoConfig
    elif config_type == "training":
        config_cls = TrainingConfig
    elif config_type == "evaluation":
        config_cls = EvaluationConfig
    else:
        raise ValueError(f"Invalid config type: {config_type}")

    if config_override != "":
        config = config_cls(**yaml.load(config_override))
    else:
        config = config_cls()

    return config


def initialize_optimizer(model, config):
    """
    Initialize the optimizer with the given config. 
    """

    if config.optimization.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimization.lr)
    else:
        raise ValueError(f"Invalid optimizer: {config.optimization.optimizer}")

    return optimizer
 

def initialize_logger(config):
    """
    Initialize the logger with the given config. 
    """

    if config.logging.logger == "wandb":
        logger = L.WandbLogger(project=config.logging.wandb_project, entity=config.logging.wandb_entity)
    else:
        raise ValueError(f"Invalid logger: {config.logging.logger}")

    return logger