"""
Utilities for checkpointing training-related states (i.e. model, optimizer, lr_scheduler, etc.)

We save both a HuggingFace model and a Fabric-specific checkpoint. The HuggingFace model is
saved at the step-specific checkpoint directory, while the Fabric-specific checkpoint is saved
in a subdirectory. This is done to facilitate easier versioning of the HuggingFace model files
(which are what gets uploaded to the Hub).
"""

import os
import yaml
from huggingface_hub import upload_folder
# from lightning.fabric.utilities.seed import _collect_rng_states, _set_rng_states

# typing imports
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch import nn
from transformers import PreTrainedTokenizerBase
from lightning.fabric import Fabric
from src.config import CheckpointingConfig
from typing import Dict, Any, Union


def load_checkpoint(
    checkpointing_config: CheckpointingConfig,
    checkpoint_step: Union[str, int],
    fabric: Fabric,
    model: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
):
    """Load model checkpoint and associated states from a given step.

    Args:
        checkpointing_config: Configuration object containing checkpoint settings
        checkpoint_step: The step at which to load the checkpoint
        fabric: Lightning Fabric instance for distributed training support
        model: The model instance to load weights into
        optimizer: The optimizer instance to load states into
        lr_scheduler: The learning rate scheduler to load states into

    Returns:
        (model, optimizer, lr_scheduler, step)
        Returns None if no checkpoint is found.
    """

    if isinstance(checkpoint_step, int):
        checkpoint_step = f"step_{checkpoint_step}"

    checkpoint_path = os.path.join(
        checkpointing_config.runs_dir,
        checkpointing_config.run_name,
        checkpointing_config.checkpoints_dir,
        checkpoint_step,
    )

    if not os.path.exists(checkpoint_path):
        return None

    # Load from specified fabric checkpoint subdirectory
    fabric_checkpoint_path = os.path.join(
        checkpoint_path, checkpointing_config.fabric_checkpoint_dir
    )

    checkpoint_state = {
        "_model": model,
        "_optimizer": optimizer,
        "_lr_scheduler": lr_scheduler,
    }
    extra_state = fabric.load(fabric_checkpoint_path, state=checkpoint_state)

    # NOTE: extra_state will contain any additional states that were saved in the checkpoint
    checkpoint_step = extra_state["_checkpoint_step"]

    return model, optimizer, lr_scheduler, checkpoint_step


def save_checkpoint(
    configs: Dict[str, Any],
    checkpoint_step: int,
    fabric: Fabric,
    model: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    tokenizer: PreTrainedTokenizerBase,
    upload_logs: bool = True,
):
    """Save training checkpoint and associated states to disk and optionally to HuggingFace Hub.

    We save the following files:
    - HuggingFace model files (config.json, pytorch_model.bin)
    - Tokenizer files (vocab.json, merges.txt)
    - Fabric-specific files (config.yaml, model.pt, optimizer.pt, training.pt)

    Note that the HuggingFace model files are saved at the step-specific checkpoint directory, while the
    Fabric-specific files are saved in a subdirectory. This is done to facilitate easier
    versioning of the HuggingFace model files (which are what gets uploaded to the Hub).

    NOTE: Why do we save a HF model at all? We do this because it makes it easier to load the model
    in a separate script for evaluation and to play nicely with the HuggingFace Hub.

    Creates a versioned checkpoint directory with the following structure:

    {checkpointing_config.runs_dir}/
        └── {checkpointing_config.run_name}/
            └── {checkpointing_config.checkpoints_dir}/
                ├── step_{checkpoint_step}/
                │   ├── config.json              # HuggingFace model config
                │   ├── pytorch_model.bin        # HuggingFace model weights
                │   ├── vocab.json               # Tokenizer vocab
                │   ├── merges.txt               # Tokenizer merges
                │   └── {checkpointing_config.fabric_checkpoint_dir}/ # Fabric-specific files
                │      ├── config.yaml           # Full training config
                │      └── checkpoint/           # Distributed model checkpoint files
                └── latest -> step_{checkpoint_step}/

    Args:
        configs: A dictionary containing the initialized configuration objects.
        checkpoint_step: The current training checkpoint step (i.e. number of learning steps taken)
        fabric: Lightning Fabric instance for distributed training support
        model: The model instance to save
        optimizer: The optimizer instance to save
        lr_scheduler: The learning rate scheduler to save
        tokenizer: The tokenizer to save
        upload_logs: Whether to upload training logs to HF Hub (default: True)

    """

    checkpointing_config = configs["checkpointing"]

    # Get the directories from the training config
    runs_dir = checkpointing_config.runs_dir
    checkpoints_dir = checkpointing_config.checkpoints_dir
    fabric_checkpoint_dir = checkpointing_config.fabric_checkpoint_dir
    logs_dir = checkpointing_config.logs_dir

    run_path = os.path.join(runs_dir, checkpointing_config.run_name)
    root_checkpoint_path = os.path.join(run_path, checkpoints_dir)
    checkpoint_path = os.path.join(root_checkpoint_path, f"step_{checkpoint_step}")

    # Create directories
    os.makedirs(checkpoint_path, exist_ok=True)

    ########################################################
    #
    # Save HuggingFace files
    #
    ########################################################

    # NOTE: we convert the Pico model to a HuggingFace model before saving it. See `model.py`
    # for more details.
    if fabric.global_rank == 0:
        hf_model = model.convert_to_hf_model()
        hf_model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)

    ########################################################
    #
    # Save Fabric-specific files
    #
    ########################################################

    # Create fabric-specific subdirectory
    fabric_checkpoint_path = os.path.join(checkpoint_path, fabric_checkpoint_dir)
    os.makedirs(fabric_checkpoint_path, exist_ok=True)

    # Save model states (use underscore to avoid conflicts with third-party libraries)
    checkpoint_state = {
        "_model": model,
        "_optimizer": optimizer,
        "_lr_scheduler": lr_scheduler,
        "_checkpoint_step": checkpoint_step,
    }
    fabric.save(fabric_checkpoint_path, checkpoint_state)

    if fabric.global_rank == 0:
        # Save config in fabric directory
        config_path = os.path.join(fabric_checkpoint_path, "config.yaml")
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                yaml.dump(configs, f)

        # Update latest symlink
        latest_symlink_path = os.path.join(root_checkpoint_path, "latest")
        if os.path.lexists(latest_symlink_path):
            os.remove(latest_symlink_path)
        os.symlink(
            f"step_{checkpoint_step}", latest_symlink_path, target_is_directory=True
        )

    ########################################################
    #
    # Push to HuggingFace Hub (if configured)
    #
    ########################################################

    if fabric.global_rank == 0:
        # Push to HuggingFace Hub if configured
        if checkpointing_config.save_checkpoint_repo_id is not None:
            # Upload the HF model
            hf_model.push_to_hub(
                repo_id=checkpointing_config.save_checkpoint_repo_id,
                commit_message=f"Saving HF Model -- Step {checkpoint_step}",
                revision=checkpointing_config.run_name,
                token=os.getenv("HF_TOKEN"),
            )

            # Upload the fabric checkpoint directory
            upload_folder(
                folder_path=fabric_checkpoint_path,
                path_in_repo=fabric_checkpoint_dir,
                repo_id=checkpointing_config.save_checkpoint_repo_id,
                commit_message=f"Saving Fabric Checkpoint -- Step {checkpoint_step}",
                revision=checkpointing_config.run_name,
                token=os.getenv("HF_TOKEN"),
            )

            # Upload logs if requested
            if upload_logs:
                logs_path = os.path.join(run_path, logs_dir)
                upload_folder(
                    folder_path=logs_path,
                    path_in_repo=logs_dir,
                    repo_id=checkpointing_config.save_checkpoint_repo_id,
                    commit_message=f"Saving Logs -- Step {checkpoint_step}",
                    revision=checkpointing_config.run_name,
                    token=os.getenv("HF_TOKEN"),
                )
