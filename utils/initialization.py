from config import PicoConfig

import yaml
import lightning as L
import torch
import os
from datetime import datetime
from wandb.integration.lightning.fabric import WandbLogger

from typing import Optional
from config import PicoConfig, TrainingConfig, EvaluationConfig
from lightning.fabric.loggers import Logger as FabricLogger

from . import ROOT_DIR


def initialize_run_dir(training_config: TrainingConfig):
    """
    Initialize the run directory with the given config. 
    """

    run_name = training_config.run_name
    if run_name is None:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        training_config.run_name = run_name

    run_dir = os.path.join(ROOT_DIR, run_name)

    os.makedirs(run_dir, exist_ok=True)


def initialize_fabric(training_config: TrainingConfig, logger: Optional[FabricLogger] = None):
    """
    Setup the lightning fabric with the given config. 
    """

    fabric = L.Fabric(
        accelerator=training_config.fabric.accelerator,
        precision=training_config.fabric.precision,
        devices=training_config.fabric.num_devices,
        num_nodes=training_config.fabric.num_nodes,
        loggers=logger
    )

    fabric.launch()

    return fabric

def initialize_logger(config):
    """
    Initialize the logger with the given config. 
    """

    if config.logging.logger is None:
        return None

    loggers = []

    if config.logging.logger == "wandb":
        assert config.logging.wandb_project is not None, \
            "Wandb project must be provided if wandb is to be used."
        assert config.logging.wandb_entity is not None, \
            "Wandb entity must be provided if wandb is to be used."
        loggers.append( 
            WandbLogger(
                project=config.logging.wandb_project,
                entity=config.logging.wandb_entity
            )
        )
    else:
        raise ValueError(f"Invalid logger: {config.logging.logger}")

    return loggers

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
 

def initialize_lr_scheduler(optimizer, training_config: TrainingConfig):
    """
    Initialize the learning rate scheduler with the given config. 
    """

    if training_config.optimization.lr_scheduler == "linear_with_warmup":

        # Credit where credit is due: 
        # https://github.com/huggingface/transformers/blob/e71a01a104dd663c730e494eb0b6467bb51df357/src/transformers/optimization.py#L102
        def _lr_lambda(curr_step, num_warmup_steps, num_training_steps):
            if curr_step < num_warmup_steps: 
                return float(curr_step) / float(max(1, num_warmup_steps))
            else: 
                return max(0.0, float(num_training_steps - curr_step) / float(max(1, num_training_steps - num_warmup_steps)))

        lr_lambda = lambda step: _lr_lambda(
            step, 
            training_config.optimization.lr_warmup_steps, 
            training_config.training_steps
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda, 
        )
    else:
        raise ValueError(f"Invalid learning rate scheduler: {training_config.optimization.lr_scheduler}")

    return lr_scheduler