"""
Logging Config

Specifies the hyperparameters for the logging process, i.e. the experiment tracker, etc.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LoggingConfig:
    experiment_tracker: Optional[str] = "wandb"
    wandb_project: Optional[str] = "pico"
    wandb_entity: Optional[str] = "pico-lm"

    log_every_n_steps: int = 100
