"""
Utilities for checkpointing.

Good, clean checkpointing is probably one of the most important parts of training pipeline,
especially for researching learning dynamics
"""

import os
import yaml
from huggingface_hub import upload_folder
from lightning.fabric.utilities.seed import _collect_rng_states, _set_rng_states
import json
from . import RUNS_DIR, CHECKPOINT_DIR, FABRIC_CHECKPOINT_DIR, EVAL_RESULTS_DIR, LOG_DIR


def load_checkpoint(
    training_config, fabric, model, optimizer, lr_scheduler, train_dataloader=None
):
    """
    Load model checkpoint and associated states from disk or latest checkpoint.

    Args:
        training_config: Configuration object containing checkpoint settings
            - load_checkpoint_path: Optional specific checkpoint path
            - load_latest_checkpoint: Boolean to load most recent checkpoint
        fabric: Lightning Fabric instance for distributed training support
        model: The model instance to load weights into
        optimizer: The optimizer instance to load states into
        lr_scheduler: The learning rate scheduler to load states into
        train_dataloader: Optional dataloader to fast-forward to the saved step

    Returns:
        If train_dataloader is provided:
            (model, optimizer, lr_scheduler, step, train_iterator)
        Otherwise:
            (model, optimizer, lr_scheduler, step)
        Returns None if no checkpoint is found.

    Raises:
        ValueError: If no checkpoint path is specified in config
    """

    if training_config.checkpointing.load_checkpoint_path:
        checkpoint_path = training_config.checkpointing.load_checkpoint_path
    elif training_config.checkpointing.load_latest_checkpoint:
        checkpoint_path = os.path.join(
            RUNS_DIR, training_config.run_name, CHECKPOINT_DIR, "latest"
        )
    else:
        raise ValueError("No checkpoint path specified")

    if not os.path.exists(checkpoint_path):
        return None

    # Load from fabric_state subdirectory
    fabric_checkpoint_dir = os.path.join(checkpoint_path, "fabric_state")

    # Load fabric-specific states
    model_state_path = os.path.join(fabric_checkpoint_dir, "model.pt")
    optimizer_state_path = os.path.join(fabric_checkpoint_dir, "optimizer.pt")
    training_state_path = os.path.join(fabric_checkpoint_dir, "training.pt")

    # load the checkpoint
    model_state = fabric.load(model_state_path)
    optimizer_state = fabric.load(optimizer_state_path)
    training_state = fabric.load(training_state_path)

    model.load_state_dict(model_state["model"])
    optimizer.load_state_dict(optimizer_state["optimizer"])
    lr_scheduler.load_state_dict(optimizer_state["lr_scheduler"])

    step = training_state["step"]
    if train_dataloader is not None:
        train_iterator = iter(train_dataloader)
        for _ in range(step):
            next(train_iterator)

    _set_rng_states(training_state["rng_state"])

    if train_dataloader is not None:
        return model, optimizer, lr_scheduler, step, train_iterator
    else:
        return model, optimizer, lr_scheduler, step


def save_checkpoint(
    sub_configs,
    fabric,
    model,
    optimizer,
    lr_scheduler,
    tokenizer,
    step,
    upload_logs=True,
):
    """
    Save model checkpoint and associated states to disk and optionally to HuggingFace Hub.

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

    {RUNS_DIR}/
        └── {training_config.run_name}/
            └── {CHECKPOINT_DIR}/
                ├── step_{step}/
                │   ├── config.json              # HuggingFace model config
                │   ├── pytorch_model.bin        # HuggingFace model weights
                │   ├── vocab.json               # Tokenizer vocab
                │   ├── merges.txt               # Tokenizer merges
                │   └── {FABRIC_CHECKPOINT_DIR}/ # Fabric-specific files
                │      ├── config.yaml           # Full training config
                │      ├── model.pt              # Fabric model state
                │      ├── optimizer.pt          # Optimizer and LR scheduler states
                │      └── training.pt           # Training progress and RNG states
                └── latest -> step_{step}/

    Args:
        sub_configs: A dictionary containing the initialized configuration objects.
        fabric: Lightning Fabric instance for distributed training support
        model: The model instance to save
        optimizer: The optimizer instance to save
        lr_scheduler: The learning rate scheduler to save
        tokenizer: The tokenizer to save
        step: Current training step
        upload_logs: Whether to upload training logs to HF Hub (default: True)

    Notes:
        - Only rank 0 process saves checkpoints in distributed training
        - Checkpoints are saved under: {RUNS_DIR}/{training_config.run_name}/{CHECKPOINT_DIR}/step_{step}
        - Existing checkpoints are not overwritten
        - HuggingFace Hub uploads are incremental (only new files are uploaded)
    """
    if fabric.global_rank != 0:
        fabric.barrier()
        return

    training_config = sub_configs["training"]

    run_dir = os.path.join(RUNS_DIR, training_config.run_name)
    root_checkpoint_dir = os.path.join(run_dir, CHECKPOINT_DIR)
    curr_checkpoint_dir = os.path.join(root_checkpoint_dir, f"step_{step}")

    # Create directories
    os.makedirs(root_checkpoint_dir, exist_ok=True)
    os.makedirs(curr_checkpoint_dir, exist_ok=True)

    ########################################################
    #
    # Save HuggingFace files
    #
    ########################################################

    # NOTE: we convert the Pico model to a HuggingFace model before saving it. See `model.py`
    # for more details.
    hf_model = model.convert_to_hf_model()
    hf_model.save_pretrained(curr_checkpoint_dir)
    tokenizer.save_pretrained(curr_checkpoint_dir)

    ########################################################
    #
    # Save Fabric-specific files
    #
    ########################################################

    # Create fabric-specific subdirectory
    fabric_checkpoint_dir = os.path.join(curr_checkpoint_dir, FABRIC_CHECKPOINT_DIR)
    os.makedirs(fabric_checkpoint_dir, exist_ok=True)

    # Save model state
    model_state_path = os.path.join(fabric_checkpoint_dir, "model.pt")
    if not os.path.exists(model_state_path):
        model_state = {"model": model.state_dict()}
        fabric.save(model_state_path, model_state)

    # Save optimizer state
    optimizer_state_path = os.path.join(fabric_checkpoint_dir, "optimizer.pt")
    if not os.path.exists(optimizer_state_path):
        optimizer_state = {
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        }
        fabric.save(optimizer_state_path, optimizer_state)

    # Save training state
    training_state_path = os.path.join(fabric_checkpoint_dir, "training.pt")
    if not os.path.exists(training_state_path):
        training_state = {
            "step": step,
            "rng_state": _collect_rng_states(),
        }
        fabric.save(training_state_path, training_state)

    # Save config in fabric directory
    config_path = os.path.join(fabric_checkpoint_dir, "config.yaml")
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            yaml.dump(sub_configs, f)

    # Update latest symlink
    latest_symlink_path = os.path.join(root_checkpoint_dir, "latest")
    if os.path.lexists(latest_symlink_path):
        os.remove(latest_symlink_path)
    os.symlink(f"step_{step}", latest_symlink_path, target_is_directory=True)

    ########################################################
    #
    # Push to HuggingFace Hub (if configured)
    #
    ########################################################

    # Push to HuggingFace Hub if configured
    if training_config.checkpointing.save_checkpoint_repo_id is not None:
        # Upload the HF model
        hf_model.push_to_hub(
            repo_id=training_config.checkpointing.save_checkpoint_repo_id,
            commit_message=f"Saving HF Model -- Step {step}",
            revision=training_config.run_name,
            token=os.getenv("HF_TOKEN"),
        )

        # Upload the fabric checkpoint directory
        upload_folder(
            folder_path=fabric_checkpoint_dir,
            path_in_repo=FABRIC_CHECKPOINT_DIR,
            repo_id=training_config.checkpointing.save_checkpoint_repo_id,
            commit_message=f"Saving Fabric Checkpoint -- Step {step}",
            revision=training_config.run_name,
            token=os.getenv("HF_TOKEN"),
        )

        # Upload logs if requested
        if upload_logs:
            upload_folder(
                folder_path=os.path.join(run_dir, LOG_DIR),
                path_in_repo=LOG_DIR,
                repo_id=training_config.checkpointing.save_checkpoint_repo_id,
                commit_message=f"Saving Logs -- Step {step}",
                revision=training_config.run_name,
                token=os.getenv("HF_TOKEN"),
            )

    fabric.barrier()


def save_evaluation_results(evaluation_config, fabric, evaluation_results, step):
    """Save evaluation results to disk and optionally to HuggingFace Hub.

    The evaluation results are saved in the following directory structure:
    {RUNS_DIR}/
        └── {evaluation_config.run_name}/
            └── {EVAL_RESULTS_DIR}/
                └── step_{step}.json

    Args:
        evaluation_config: Configuration object containing evaluation settings
        fabric: Lightning Fabric instance
        evaluation_results: Dictionary containing evaluation metrics
        step: Current training step
    """

    # Only save on rank 0 to avoid conflicts

    run_dir = os.path.join(RUNS_DIR, evaluation_config.run_name)
    eval_results_dir = os.path.join(run_dir, EVAL_RESULTS_DIR)

    if fabric.global_rank == 0:
        os.makedirs(eval_results_dir, exist_ok=True)

        curr_eval_results_path = os.path.join(eval_results_dir, f"step_{step}.json")

        # save out as json
        with open(curr_eval_results_path, "w") as f:
            json.dump(evaluation_results, f)

        if evaluation_config.save_checkpoint_repo_id is not None:
            upload_folder(
                folder_path=eval_results_dir,
                path_in_repo=EVAL_RESULTS_DIR,
                repo_id=evaluation_config.save_checkpoint_repo_id,
                commit_message=f"Saving Evaluation Results -- Step {step}",
                revision=evaluation_config.run_name,
                token=os.getenv("HF_TOKEN"),
            )

    fabric.barrier()
