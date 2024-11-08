import os
import yaml
from huggingface_hub import upload_folder, upload_file
from lightning.fabric.utilities.seed import _collect_rng_states, _set_rng_states

from . import RUNS_DIR, CHECKPOINT_DIR


def load_checkpoint(
    fabric, training_config, model, optimizer, lr_scheduler, train_dataloader=None
):
    """
    Load a checkpoint from the specified path.
    Optionally, fast forward the dataloader to the start step.


    Returns None if no checkpoint is found. The user should check for this condition and handle it
    accordingly.
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

    # load the checkpoint
    model_state_path = os.path.join(checkpoint_path, "model.pt")
    optimizer_state_path = os.path.join(checkpoint_path, "optimizer.pt")
    training_state_path = os.path.join(checkpoint_path, "training.pt")

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


def save_checkpoint(fabric, training_config, model, optimizer, lr_scheduler, step):
    """
    Save a checkpoint to the specified path.
    """

    if fabric.global_rank != 0:
        fabric.barrier()
        return

    run_dir = os.path.join(RUNS_DIR, training_config.run_name)

    root_checkpoint_dir = os.path.join(run_dir, CHECKPOINT_DIR)
    os.makedirs(root_checkpoint_dir, exist_ok=True)

    curr_checkpoint_dir = os.path.join(root_checkpoint_dir, f"step_{step}")
    os.makedirs(curr_checkpoint_dir, exist_ok=True)

    model_state_path = os.path.join(curr_checkpoint_dir, "model.pt")
    if not os.path.exists(model_state_path):
        model_state = {
            "model": model,
        }
        fabric.save(model_state_path, model_state)

    optimizer_state_path = os.path.join(curr_checkpoint_dir, "optimizer.pt")
    if not os.path.exists(optimizer_state_path):
        optimizer_state = {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }
        fabric.save(optimizer_state_path, optimizer_state)

    training_state_path = os.path.join(curr_checkpoint_dir, "training.pt")
    if not os.path.exists(training_state_path):
        training_state = {
            "step": step,
            "rng_state": _collect_rng_states(),
        }
        fabric.save(training_state_path, training_state)

    # create symlink to latest checkpoint directory
    latest_symlink_path = os.path.join(run_dir, CHECKPOINT_DIR, "latest")
    if os.path.lexists(latest_symlink_path):
        os.remove(latest_symlink_path)

    os.symlink(f"step_{step}", latest_symlink_path, target_is_directory=True)

    # Pushing to HuggingFace Hub
    # NOTE: if the file already exists, HF will not upload it again (by default)
    if training_config.checkpointing.hf_repo_id is not None:
        if step == 0:
            # upload the config to the HuggingFace Hub
            upload_file(
                path_or_fileobj=os.path.join(run_dir, "config.yaml"),
                path_in_repo="config.yaml",
                repo_id=training_config.checkpointing.hf_repo_id,
                commit_message="Saving run config",
                revision=training_config.run_name,
                token=os.getenv("HF_TOKEN"),
            )

        # uploading models and optimizer to HuggingFace Hub
        upload_folder(
            folder_path=curr_checkpoint_dir,
            repo_id=training_config.checkpointing.hf_repo_id,
            commit_message=f"Saving Model -- Step {step}",
            revision=training_config.run_name,
            token=os.getenv("HF_TOKEN"),
        )

        # uploading logs to HuggingFace Hub
        upload_folder(
            folder_path=os.path.join(run_dir, "logs"),
            path_in_repo="logs",
            repo_id=training_config.checkpointing.hf_repo_id,
            commit_message=f"Saving Logs -- Step {step}",
            revision=training_config.run_name,
            token=os.getenv("HF_TOKEN"),
        )

    fabric.barrier()


def save_config(fabric, training_config, model_config, evaluation_config):
    """
    Save the config to a file.
    """
    if fabric.global_rank != 0:
        fabric.barrier()
        return

    config_path = os.path.join(RUNS_DIR, training_config.run_name, "config.yaml")
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            yaml.dump(training_config, f)
            yaml.dump(model_config, f)
            yaml.dump(evaluation_config, f)

    fabric.barrier()
