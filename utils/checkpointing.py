
import os 
import yaml
from huggingface_hub import upload_folder
from lightning.fabric.utilities.seed import _collect_rng_states, _set_rng_states

from . import ROOT_DIR, CHECKPOINT_DIR

def load_checkpoint(fabric, training_config, model, optimizer, lr_scheduler):
    """
    Load a checkpoint from the specified path.
    """

    assert os.path.exists(training_config.checkpointing.load_path),\
        f"Checkpoint path {training_config.checkpointing.load_path} does not exist"

    # load the checkpoint
    model_state_path = os.path.join(training_config.checkpointing.load_path, "model.pt")
    optimizer_state_path = os.path.join(training_config.checkpointing.load_path, "optimizer.pt")
    training_state_path = os.path.join(training_config.checkpointing.load_path, "training.pt")

    # load the checkpoint
    model_state = fabric.load(model_state_path)
    optimizer_state = fabric.load(optimizer_state_path)
    training_state = fabric.load(training_state_path)

    model.load_state_dict(model_state["model"])
    optimizer.load_state_dict(optimizer_state["optimizer"])
    lr_scheduler.load_state_dict(optimizer_state["lr_scheduler"])
    _set_rng_states(training_state["rng_state"])
    step = training_state["step"]

    return model, optimizer, lr_scheduler, step

def save_checkpoint(fabric, training_config, model, optimizer, lr_scheduler, step):
    """
    Save a checkpoint to the specified path.
    """

    run_dir = os.path.join(ROOT_DIR, training_config.run_name)
    root_checkpoint_dir = os.path.join(run_dir, CHECKPOINT_DIR)
    os.makedirs(root_checkpoint_dir, exist_ok=True)

    curr_checkpoint_dir = os.path.join(root_checkpoint_dir, f"step_{step}")

    os.makedirs(curr_checkpoint_dir, exist_ok=True)

    model_state_path = os.path.join(curr_checkpoint_dir, "model.pt")
    optimizer_state_path = os.path.join(curr_checkpoint_dir, "optimizer.pt")
    training_state_path = os.path.join(curr_checkpoint_dir, "training.pt")

    model_state = {
        "model": model,
   }
    optimizer_state = {
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
    }
    training_state = {
        "step": step,
        "rng_state": _collect_rng_states(),
    }

    fabric.save(model_state_path, model_state)
    fabric.save(optimizer_state_path, optimizer_state)
    fabric.save(training_state_path, training_state)

    # create symlink to latest checkpoint directory
    latest_symlink_path = os.path.join(run_dir, CHECKPOINT_DIR, "latest")
    if os.path.lexists(latest_symlink_path):
        os.remove(latest_symlink_path)

    os.symlink(curr_checkpoint_dir, latest_symlink_path)
    
    # Pushing to HuggingFace Hub
    if training_config.checkpointing.hf_repo_id is not None:
        commit_message = f"Model Save -- Step {step}"
        print(f'folder_path: {curr_checkpoint_dir}')
        print(f'repo_id: {training_config.checkpointing.hf_repo_id}')
        print(f'commit_message: {commit_message}')
        print(f'revision: {training_config.run_name}')
        upload_folder(
            folder_path=curr_checkpoint_dir,
            repo_id=training_config.checkpointing.hf_repo_id,
            commit_message=commit_message,
            revision=training_config.run_name,
        )


def save_config(training_config, model_config, evaluation_config):
    """
    Save the config to a file.
    """

    config_path = os.path.join(ROOT_DIR, training_config.run_name, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(training_config, f)
        yaml.dump(model_config, f)
        yaml.dump(evaluation_config, f)