import lightning as L
import torch
import os
import logging
import yaml
from dataclasses import fields, is_dataclass
from datetime import datetime
import wandb
from wandb.integration.lightning.fabric import WandbLogger

from typing import Optional
from config import (
    DataConfig,
    ModelConfig,
    TrainingConfig,
    EvaluationConfig,
)

from lightning.fabric.loggers import Logger as FabricLogger

from . import RUNS_DIR, CHECKPOINT_DIR

########################################################
#
# Basic Initialization
#
########################################################


def _apply_config_overrides(config, overrides: dict):
    """
    Apply the given config overrides to the config.
    """
    for field in fields(config):
        field_value = getattr(config, field.name)
        if is_dataclass(field_value):
            _apply_config_overrides(field_value, overrides.get(field.name, {}))
        else:
            if field.name in overrides:
                setattr(config, field.name, overrides[field.name])
    return config


def initialize_config(config_path: Optional[str] = None):
    """
    Setup the config with the given config_overrie.
    """
    data_config = DataConfig()
    model_config = ModelConfig()
    training_config = TrainingConfig()
    evaluation_config = EvaluationConfig()

    if config_path:
        # NOTE: Config overriding logic - this can be reimplemented by users in any number of
        # different ways (e.g., hydra config setup); here I'm trying to setup the simplest possible
        # config override system that is still flexible and extendable.
        overrides = yaml.safe_load(open(config_path, "r"))
        data_config = _apply_config_overrides(data_config, overrides.get("data", {}))
        model_config = _apply_config_overrides(model_config, overrides.get("model", {}))
        training_config = _apply_config_overrides(
            training_config, overrides.get("training", {})
        )
        evaluation_config = _apply_config_overrides(
            evaluation_config, overrides.get("evaluation", {})
        )

    return data_config, model_config, training_config, evaluation_config


def initialize_run_dir(training_config: TrainingConfig):
    """
    Initialize the run directory with the given config.
    """

    run_name = training_config.run_name
    if run_name is None:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        training_config.run_name = run_name

    run_dir = os.path.join(RUNS_DIR, run_name)

    os.makedirs(run_dir, exist_ok=True)


def initialize_fabric(
    training_config: TrainingConfig, experiment_tracker: Optional[FabricLogger] = None
):
    """
    Setup the lightning fabric with the given config.
    """

    fabric = L.Fabric(
        accelerator=training_config.fabric.accelerator,
        precision=training_config.fabric.precision,
        devices=training_config.fabric.num_devices,
        num_nodes=training_config.fabric.num_nodes,
        loggers=[experiment_tracker] if experiment_tracker is not None else None,
    )

    fabric.launch()

    return fabric


########################################################
#
# Logging
#
########################################################


def _initialize_log_file(training_config: TrainingConfig) -> str:
    """
    Create a log file in the run directory.
    """
    run_dir = os.path.join(RUNS_DIR, training_config.run_name)
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # datetime stamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"log_{timestamp}.txt"
    log_file_path = os.path.join(logs_dir, log_file_name)

    open(log_file_path, "w").close()  # Create an empty log file

    return log_file_path


def initialize_logging(training_config: TrainingConfig):
    """
    Initialize the logging functionality. A standard file and congole logger is initialized by default.

    If experiment trackers (wandb, etc.) are specified in the config, those loggers are initialized.
    """
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create file handler
    log_file_path = _initialize_log_file(training_config)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handler
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

    # Add a stream handler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # ---- Third Party Loggers ----
    # NOTE: add other experiment trackers here that you want to use

    experiment_tracker = None
    if training_config.logging.experiment_tracker == "wandb":
        assert (
            training_config.logging.wandb_project is not None
        ), "Wandb project must be provided if wandb is to be used."
        assert (
            training_config.logging.wandb_entity is not None
        ), "Wandb entity must be provided if wandb is to be used."

        _run_id = None
        if (
            training_config.checkpointing.load_checkpoint_path
            or training_config.checkpointing.load_latest_checkpoint
        ):
            # If we are loading a checkpoint, we can try to find the run id of the previous run
            previous_runs = wandb.Api().runs(
                path="pico-lm/pico", filters={"display_name": training_config.run_name}
            )
            if len(previous_runs) == 1:
                _run_id = previous_runs[0].id

        experiment_tracker = WandbLogger(
            project=training_config.logging.wandb_project,
            entity=training_config.logging.wandb_entity,
            id=_run_id,
            name=training_config.run_name,
        )
    elif (
        training_config.logging.experiment_tracker is not None
        and training_config.logging.experiment_tracker != ""
    ):
        raise ValueError(
            f"Invalid experiment tracker: {training_config.logging.experiment_tracker}"
        )

    return logger, experiment_tracker


########################################################
#
# Checkpointing
#
########################################################k


def initialize_checkpointing(training_config: TrainingConfig):
    """
    Initialize any checkpointing functionality. The main thing right now is setting up
    HuggingFace repos for storing model checkpoints, but this could be extended to other
    checkpointing backends in the future.
    """

    huggingface_repo_id = training_config.checkpointing.save_checkpoint_repo_id
    if huggingface_repo_id is None:
        return

    import time
    from huggingface_hub.hf_api import create_repo, create_branch
    from huggingface_hub.errors import HfHubHTTPError
    from huggingface_hub.repository import Repository

    run_dir = os.path.join(RUNS_DIR, training_config.run_name)
    checkpoint_dir = os.path.join(run_dir, CHECKPOINT_DIR)

    _repo_sleep_time = 1
    _repo_created = False
    while not _repo_created:
        try:
            # Make sure the repo exists.
            create_repo(
                huggingface_repo_id,
                exist_ok=True,
            )
            _repo_created = True
        except HfHubHTTPError:
            if _repo_sleep_time > 64:
                raise RuntimeError(
                    f"Could not create huggingface repo {huggingface_repo_id} after {64} seconds."
                )
            time.sleep(_repo_sleep_time)
            _repo_sleep_time *= 2

    # create branch
    create_branch(
        repo_id=huggingface_repo_id, branch=training_config.run_name, exist_ok=True
    )

    _ = Repository(
        checkpoint_dir,
        clone_from=huggingface_repo_id,
        revision=training_config.run_name,
    )


########################################################
#
# Optimizer and Scheduler
#
########################################################


def initialize_optimizer(model, training_config: TrainingConfig):
    """
    Initialize the optimizer with the given config.
    """

    if training_config.optimization.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=training_config.optimization.lr
        )
    else:
        raise ValueError(f"Invalid optimizer: {training_config.optimization.optimizer}")

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
                return max(
                    0.0,
                    float(num_training_steps - curr_step)
                    / float(max(1, num_training_steps - num_warmup_steps)),
                )

        lr_lambda = lambda step: _lr_lambda(  # noqa: E731
            step,
            training_config.optimization.lr_warmup_steps,
            training_config.training_steps,
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda,
        )
    else:
        raise ValueError(
            f"Invalid learning rate scheduler: {training_config.optimization.lr_scheduler}"
        )

    return lr_scheduler
