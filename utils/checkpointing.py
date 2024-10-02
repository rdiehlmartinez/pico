
import os 
import yaml
from lightning.fabric.utilities.seed import _collect_rng_states, _set_rng_states

from . import ROOT_DIR, CHECKPOINT_DIR


def load_checkpoint(fabric, training_config, model, optimizer, lr_scheduler):

    assert os.path.exists(training_config.checkpointing.load_path),\
        f"Checkpoint path {training_config.checkpointing.load_path} does not exist"

    checkpoint = fabric.load(training_config.checkpointing.load_path)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    _set_rng_states(checkpoint["rng_state"])
    step = checkpoint["step"]

    return model, optimizer, lr_scheduler, step

def save_checkpoint(fabric, training_config, model, optimizer, lr_scheduler, step):

    run_dir = os.path.join(ROOT_DIR, training_config.run_name)
    checkpoint_path = os.path.join(run_dir, CHECKPOINT_DIR, f"step_{step}.ckpt")

    state = {
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "step": step,
        "rng_state": _collect_rng_states(),
    }

    fabric.save(checkpoint_path, state)

    # create symlink to latest checkpoint
    # if latest.ckpt exists, remove it 
    latest_symlink_path = os.path.join(run_dir, CHECKPOINT_DIR, "latest.ckpt")
    if os.path.lexists(latest_symlink_path):
        print(f"Removing existing latest.ckpt at {latest_symlink_path}")
        os.remove(latest_symlink_path)
    else:
        print(f"Creating symlink to {checkpoint_path} at {latest_symlink_path}")
        print("symbolic link does not exist")

    os.symlink(checkpoint_path, latest_symlink_path)
    
    # push to HuggingFace Hub
    if training_config.checkpointing.hf_repo_id is not None:
        commit_message = f"Model Save -- Step {step}"
        fabric.print(f"Saving checkpoint to HuggingFace Hub at {training_config.checkpointing.hf_repo_id}")
        fabric.print(f"Commit Message: {commit_message}")
        # TODO: Push to HuggingFace Hub

def save_config(training_config, model_config, evaluation_config):
    """
    Save the config to a file.
    """

    config_path = os.path.join(ROOT_DIR, training_config.run_name, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(training_config, f)
        yaml.dump(model_config, f)
        yaml.dump(evaluation_config, f)
