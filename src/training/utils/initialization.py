"""
Utilities for initializing components of the training process.

Here, we initialize all of the components that are part of the learning process. From logging,
and checkpointing to the optimizer to the dataset and the dataloader, this file contains the
logic for setting up the classes and functions that are used in the training loop.

As always, this code is meant to be basic. We hard-code the obvious defaults, and leave the
more experimental stuff to you.
"""

import lightning as L
import torch
import os
import logging
import yaml
from dataclasses import fields, is_dataclass
from datetime import datetime
import wandb
from huggingface_hub import create_repo, create_branch
from wandb.integration.lightning.fabric import WandbLogger
from datasets import load_dataset, Dataset, DownloadConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import Optional, Dict, Union
import warnings

from src.config import (
    TrainingConfig,
    DataConfig,
    ModelConfig,
    EvaluationConfig,
    MonitoringConfig,
    CheckpointingConfig,
)

from lightning.fabric.loggers import Logger as FabricLogger


warnings.filterwarnings(
    "ignore",
    message=".*This integration is tested and supported for lightning Fabric.*",
)
warnings.filterwarnings(
    "ignore",
    message=".*Please report any issues to.*",
)

########################################################
#
# Basic Initialization
#
########################################################


def _apply_config_overrides(config, overrides: dict):
    """Recursively apply configuration overrides to a dataclass config object.

    Args:
        config: Base configuration object (must be a dataclass)
        overrides: Dictionary of override values matching config structure

    Returns:
        Modified config object with overrides to the config.
    """
    for field in fields(config):
        field_value = getattr(config, field.name)
        if is_dataclass(field_value):
            _apply_config_overrides(field_value, overrides.get(field.name, {}))
        else:
            if field.name in overrides:
                setattr(config, field.name, overrides[field.name])
    return config


def initialize_configuration(
    config_path: Optional[str] = None,
) -> Dict[
    str,
    Union[
        DataConfig,
        ModelConfig,
        TrainingConfig,
        EvaluationConfig,
        MonitoringConfig,
        CheckpointingConfig,
    ],
]:
    """Initialize configuration objects with optional overrides from a YAML file.

    This function initializes all of the configuration objects, and then applies
    any overrides from the config_path file. If no config_path is provided,
    the function will use the default configuration objects.

    Args:
        config_path: Path to a YAML file containing configuration overrides.

    Returns:
        A dictionary containing the initialized configuration objects.
    """
    data_config = DataConfig()
    model_config = ModelConfig()
    training_config = TrainingConfig()
    evaluation_config = EvaluationConfig()
    monitoring_config = MonitoringConfig()
    checkpointing_config = CheckpointingConfig()

    if config_path:
        overrides = yaml.safe_load(open(config_path, "r"))
        data_config = _apply_config_overrides(data_config, overrides.get("data", {}))
        model_config = _apply_config_overrides(model_config, overrides.get("model", {}))
        training_config = _apply_config_overrides(
            training_config, overrides.get("training", {})
        )
        evaluation_config = _apply_config_overrides(
            evaluation_config, overrides.get("evaluation", {})
        )
        monitoring_config = _apply_config_overrides(
            monitoring_config, overrides.get("monitoring", {})
        )
        checkpointing_config = _apply_config_overrides(
            checkpointing_config, overrides.get("checkpointing", {})
        )

    configs = {
        "data": data_config,
        "model": model_config,
        "training": training_config,
        "evaluation": evaluation_config,
        "monitoring": monitoring_config,
        "checkpointing": checkpointing_config,
    }

    return configs


def initialize_run_dir(checkpointing_config: CheckpointingConfig) -> str:
    """Initialize a directory for the current training run.

    Creates a unique directory for storing training, evaluation, and logging artifacts.
    If no run name is specified in the config, generates a timestamp-based name.

    Args:
        checkpointing_config: Configuration object containing run settings.
            NOTE: Must have a 'run_name' attribute that can be None, in which case
            a timestamp-based name will be generated.

    Returns:
        str: The path to the run directory.
    """
    run_name = checkpointing_config.run_name
    if run_name is None:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        checkpointing_config.run_name = run_name

    run_dir = os.path.join(checkpointing_config.runs_dir, run_name)

    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def initialize_fabric(
    training_config: TrainingConfig, experiment_tracker: Optional[FabricLogger] = None
):
    """Initialize Lightning Fabric for distributed training.

    Sets up a Lightning Fabric instance with the specified configuration for
    handling distributed training, mixed precision, and logging.

    Args:
        training_config: Configuration object containing fabric settings
            (accelerator, precision, devices, etc.).
        experiment_tracker: Optional logger instance for experiment tracking
            (e.g., WandB logger).

    Returns:
        L.Fabric: Initialized Lightning Fabric instance.

    Example:
        >>> fabric = initialize_fabric(training_config, wandb_logger)
    """

    total_devices = (
        training_config.fabric.num_devices * training_config.fabric.num_nodes
    )

    if total_devices > 1:
        strategy = "deepspeed_stage_2"
    else:
        strategy = "auto"  # Sets up SingleDevice Strategy by default

    # NOTE: The strategy is set to use either DeepSpeed (Zero Stage 2) on multi-GPU,
    # or SingleDevice Strategy on single-GPU set ups. If you'd like to use a different strategy,
    # you can change the strategy flag in the fabric initialization, but be aware that this might
    # cause issues with checkpointing, evaluation, etc.

    fabric = L.Fabric(
        accelerator=training_config.fabric.accelerator,
        precision=training_config.fabric.precision,
        devices=training_config.fabric.num_devices,
        num_nodes=training_config.fabric.num_nodes,
        loggers=[experiment_tracker] if experiment_tracker is not None else None,
        strategy=strategy,
    )

    fabric.launch()

    return fabric


########################################################
#
# Dataset and Tokenization Initialization
#
########################################################


def initialize_dataset(
    data_config: DataConfig,
    fabric: L.Fabric,
    initial_batch_step: Optional[int] = 0,
    return_fast_forward_steps: bool = False,
):
    """Initialize dataset based on the given config.

    This function will return a dataset object, and optionally a fast_forward_steps value.

    The fast_forward_steps value is the number of steps that we need to fast-forward an iterator by,
    so that we can continue from a certain batch of data we would have seen had training not previously
    stopped. Depending on how the dataset is loaded, the amount of steps to fast-forward may be
    different from the initial_batch_step value.

    NOTE: This functionality is primarily useful for streaming datasets (which for large
    datasets is most of the time).

    Args:
        data_config: Configuration object containing dataset settings.
        fabric: A Lightning Fabric instance.
        initial_batch_step: The initial batch step to fast-forward to.
        return_fast_forward_steps: Whether to return the fast-forward steps value.

    Returns:
        Dataset: Initialized dataset object.
        Optional[int]: Number of steps to fast-forward the iterator by, if return_fast_forward_steps is True.
    """

    download_config = DownloadConfig(
        max_retries=10,  # default is 1 and can lead to pre-mature HTTPS errors
    )

    fast_forward_steps = 0

    if data_config.dataset.name == "pico-lm/pretokenized-dolma":
        # NOTE: We know that the dataset is sharded into 10,000 shards, so we can easily compute
        # the data file that we need to load in that contains the batch of data at
        # initial_batch_step.

        if initial_batch_step is not None:
            examples_per_shard = 20_480
            total_shards = 10_000
            batches_per_shard = examples_per_shard // data_config.dataloader.batch_size
            shard_idx = initial_batch_step // batches_per_shard

            data_files = [
                f"data/train-{str(_shard_idx).zfill(5)}-of-{total_shards}.parquet"
                for _shard_idx in range(shard_idx, total_shards)
            ]

            fast_forward_steps = initial_batch_step % batches_per_shard
        else:
            data_files = None

        base_dataset = load_dataset(
            data_config.dataset.name,
            split="train",
            streaming=True,
            data_files=data_files,
            download_config=download_config,
        )
    else:
        # NOTE: For other datasets, you might want to add some custom loading logic, especially
        # to help with loading or fast-forwarding to the correct batch.

        base_dataset = load_dataset(
            data_config.dataset.name,
            split="train",
            streaming=True,
            download_config=download_config,
        )

    if data_config.dataset.name == "pico-lm/pretokenized-dolma":
        from .data import ShardedIterableDataset

        # NOTE: We wrap the dataset in a ShardedIterableDataset, which is a custom class that
        # allows us to shard an iterable dataset across multiple processes. This is useful for
        # distributed training, where we want data-parallelism.
        dataset = ShardedIterableDataset(
            base_dataset, fabric.global_rank, fabric.world_size
        )
    else:
        dataset = base_dataset

    if return_fast_forward_steps:
        return dataset, fast_forward_steps
    else:
        return dataset


def initialize_tokenizer(data_config: DataConfig):
    """Initialize the tokenizer for text processing.

    This function can be extended to include custom tokenization logic.

    Args:
        data_config: Configuration object containing tokenizer settings.

    Returns:
        AutoTokenizer: A HuggingFace tokenizer instance.
    """

    return AutoTokenizer.from_pretrained(data_config.tokenizer.name)


def initialize_dataloader(
    data_config: DataConfig,
    training_config: TrainingConfig,
    fabric: L.Fabric,
    dataset: Dataset,
):
    """Initialize the DataLoader for efficient batch processing.

    Creates a PyTorch DataLoader that handles batching and data loading for training.
    Configured specifically for streaming tokenized text datasets.

    You might also want to extend this function to add a sampler, or some sort of custom
    collate function. For the default dataset, we don't need any of this, because the data are
    pre-shuffled, and pre-tokenized.

    Args:
        data_config: Configuration object containing dataloader settings.
        training_config: Configuration object containing training settings.
        fabric: A Lightning Fabric instance.
        dataset: A HuggingFace Dataset object containing tokenized text data.
            Expected to have 'input_ids' field in its items.

    Returns:
        DataLoader: PyTorch DataLoader instance configured for the dataset.
    """

    def _collate_fn(batch):
        return {"input_ids": [entry["input_ids"] for entry in batch]}

    sub_batch_size = data_config.dataloader.batch_size // (
        fabric.world_size * training_config.optimization.gradient_accumulation_steps
    )

    # NOTE: We use the sub-batch size for the dataloader, which is the full batch size
    # divided by the gradient accumulation steps. This ensures that the effective batch size
    # is correct.

    return DataLoader(
        dataset,
        batch_size=sub_batch_size,
        shuffle=False,  # Keep sequential for streaming datasets
        pin_memory=True,  # Speeds up transfer to GPU
        collate_fn=_collate_fn,
    )


########################################################
#
# Optimizer and Scheduler
#
########################################################


def initialize_optimizer(training_config: TrainingConfig, model: torch.nn.Module):
    """Initialize the optimizer for model training.

    Creates an optimizer instance based on the configuration settings.

    Add whatever other optimizers you want here.

    Args:
        training_config: Configuration object containing optimizer settings.
            Must have:
            - optimization.optimizer (str): Name of the optimizer ("adamw")
            - optimization.lr (float): Learning rate for the optimizer
        model: PyTorch model whose parameters will be optimized.

    Returns:
        torch.optim.Optimizer: Configured optimizer instance.

    """

    if training_config.optimization.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=training_config.optimization.lr
        )
    else:
        raise ValueError(f"Invalid optimizer: {training_config.optimization.optimizer}")

    return optimizer


def initialize_lr_scheduler(
    training_config: TrainingConfig, optimizer: torch.optim.Optimizer
):
    """Initialize a learning rate scheduler with warmup and decay.

    The default is a learning rate scheduler that implements a linear warmup followed by
    linear decay. The learning rate increases linearly from 0 to the initial lr
    during warmup, then decreases linearly to 0 during the remaining steps.

    Add other types of learning rate schedulers here.

    Args:
        training_config: Configuration object containing optimizer and scheduler settings.
        optimizer: PyTorch optimizer whose learning rate will be scheduled.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: Learning rate scheduler instance.
    """

    if training_config.optimization.lr_scheduler == "linear_with_warmup":
        # Credit where credit is due:
        # https://github.com/huggingface/transformers/blob/e71a01a104dd663c730e494eb0b6467bb51df357/src/transformers/optimization.py#L102
        def _lr_lambda(curr_step, num_warmup_steps, max_steps):
            if curr_step < num_warmup_steps:
                return float(curr_step) / float(max(1, num_warmup_steps))
            else:
                return max(
                    0.0,
                    float(max_steps - curr_step)
                    / float(max(1, max_steps - num_warmup_steps)),
                )

        lr_lambda = lambda step: _lr_lambda(  # noqa: E731
            step,
            training_config.optimization.lr_warmup_steps,
            training_config.max_steps,
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


########################################################
#
# Experiment Monitoring (Logging, Experiment Tracking, etc.)
#
########################################################


def _initialize_log_file(checkpointing_config: CheckpointingConfig) -> str:
    """Create and initialize a timestamped log file in the run's log directory.

    Sets up a log file with a unique timestamp in the run's logging directory.
    Creates the necessary directory structure if it doesn't exist.

    Directory Structure:
        {checkpointing_config.runs_dir}/
        └── {checkpointing_config.run_name}/
            └── {checkpointing_config.logs_dir}/
                └── log_YYYYMMDD_HHMMSS.txt

    Args:
        checkpointing_config: Configuration object containing checkpointing settings.

    Returns:
        str: Absolute path to the created log file.

    """

    run_dir = os.path.join(checkpointing_config.runs_dir, checkpointing_config.run_name)
    logs_dir = os.path.join(run_dir, checkpointing_config.logs_dir)
    os.makedirs(logs_dir, exist_ok=True)

    # datetime stamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"log_{timestamp}.txt"
    log_file_path = os.path.join(logs_dir, log_file_name)

    open(log_file_path, "w").close()  # Create an empty log file

    return log_file_path


def initialize_experiment_tracker(
    monitoring_config: MonitoringConfig, checkpointing_config: CheckpointingConfig
):
    """Initialize an experiment tracker.

    This function initializes an experiment tracker based on the configuration settings. Out of
    the box, Pico supports Weights and Biases.

    Args:
        monitoring_config: Configuration object containing monitoring settings.
        checkpointing_config: Configuration object containing checkpointing settings.

    Returns:
        Optional[WandbLogger]: An experiment tracker instance.
    """
    # NOTE: Add whatever other experiment trackers here that you want to use here.

    experiment_tracker = None
    if monitoring_config.experiment_tracker.framework == "wandb":
        assert (
            monitoring_config.experiment_tracker.wandb_project is not None
        ), "Wandb project must be provided if wandb is to be used."
        assert (
            monitoring_config.experiment_tracker.wandb_entity is not None
        ), "Wandb entity must be provided if wandb is to be used."

        _run_id = None
        if checkpointing_config.training.auto_resume:
            # If we are loading a checkpoint, we can try to find the run id of the previous run
            previous_runs = wandb.Api().runs(
                path="pico-lm/pico",
                filters={"display_name": checkpointing_config.run_name},
            )
            try:
                if len(previous_runs) == 1:
                    _run_id = previous_runs[0].id
            except ValueError:
                pass

        experiment_tracker = WandbLogger(
            project=monitoring_config.experiment_tracker.wandb_project,
            entity=monitoring_config.experiment_tracker.wandb_entity,
            id=_run_id,
            name=checkpointing_config.run_name,
        )
    elif (
        monitoring_config.experiment_tracker.framework is not None
        and monitoring_config.experiment_tracker.framework != ""
    ):
        raise ValueError(
            f"Invalid experiment tracker: {monitoring_config.experiment_tracker.framework}"
        )

    return experiment_tracker


def initialize_logging(
    monitoring_config: MonitoringConfig,
    checkpointing_config: CheckpointingConfig,
    fabric: L.Fabric,
):
    """Initialize logging system with default logging, to file and console.

    The default logging system uses a file handler and a stream handler.

    Args:
        monitoring_config: Configuration object containing monitoring settings.
        checkpointing_config: Configuration object containing checkpointing settings.

    Returns:
        logger: Standard Python logger configured for file and console output
    """

    # ---- Standard Local Logger ---- #
    if fabric.global_rank != 0:
        return

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create file handler
    log_file_path = _initialize_log_file(checkpointing_config)
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(monitoring_config.logging.log_level)

    # Create formatter and add it to the handler
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

    # Add a stream handler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(monitoring_config.logging.log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


########################################################
#
# HuggingFace/Remote Checkpointing
#
########################################################


def initialize_hf_checkpointing(
    checkpointing_config: CheckpointingConfig, fabric: L.Fabric
):
    """Initialize HuggingFace Checkpointing.

    Creates a HuggingFace repository if it doesn't exist, and creates a branch named after the run.

    Args:
        checkpointing_config: Configuration object containing checkpointing settings;
            must have a 'save_checkpoint_repo_id' attribute.

    Raises:
        RuntimeError: If unable to create HuggingFace repository after multiple attempts.
    """

    if fabric.global_rank != 0:
        return

    huggingface_repo_id = checkpointing_config.save_checkpoint_repo_id
    assert huggingface_repo_id is not None, "save_checkpoint_repo_id must be provided."

    create_repo(huggingface_repo_id, exist_ok=True)

    create_branch(
        repo_id=huggingface_repo_id,
        branch=checkpointing_config.run_name,
        exist_ok=True,
    )
