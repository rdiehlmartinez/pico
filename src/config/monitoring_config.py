"""
Monitoring Config

Specifies the monitoring process, e.g. how to log metrics and keep track of training progress.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoggingConfig:
    log_level: str = "INFO"
    log_every_n_steps: int = 100


@dataclass
class ExperimentTrackerConfig:
    framework: Optional[str] = "wandb"
    wandb_project: Optional[str] = "pico"
    wandb_entity: Optional[str] = "pico-lm"


@dataclass
class MonitoringConfig:
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    experiment_tracker: ExperimentTrackerConfig = field(
        default_factory=ExperimentTrackerConfig
    )
