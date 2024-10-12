ROOT_DIR = "runs"
CHECKPOINT_DIR = "checkpoints"

import os
from config import TrainingConfig


def is_logged_into_huggingface(training_config: TrainingConfig) -> bool:
    """Checks if the user is logged into HuggingFace."""
    # check that the user has write access to https://huggingface.co/pico-lm
    from huggingface_hub import HfApi

    # check that env HF_TOKEN is set
    if os.getenv("HF_TOKEN") is None:
        print("No HuggingFace token found. You might need to run `source setup.sh` or set the HF_TOKEN environment variable.")
        return False

    target_org = training_config.checkpointing.hf_repo_id.split("/")[0]
        
    api = HfApi(token=os.getenv("HF_TOKEN"))
    who_ami = api.whoami()

    # check if who_ami has access to org pico-lm
    orgs = who_ami.get('orgs', []) + [who_ami['name']]

    for org in orgs:
        if org['name'] == target_org:
            if org['roleInOrg'] not in ['admin', 'write']:
                raise ValueError(f"You don't have write access to the {target_org} organization on HuggingFace.")
            else:
                return True
    
    # the target_org might be the user's username
    if who_ami['name'] == target_org:
        return True

    raise ValueError(
        f"""Something went wrong with your HuggingFace login. 
        Either: 
        1. You don't have access to the {target_org} organization on HuggingFace or 
        2. The organization does not exist.
        3. You forgot to set the HF_TOKEN environment variable.
        Please check your HuggingFace credentials and try again.
        """
    )

def is_logged_into_wandb(training_config: TrainingConfig) -> bool:
    """Checks if the user is logged into Weights & Biases."""
    import wandb


    # check that env wandb_api_key is set
    if os.getenv("WANDB_API_KEY") is None:
        print("No Weights & Biases API key found. You might need to run `source setup.sh` or set the WANDB_API_KEY environment variable.")
        return False

    api = wandb.Api(api_key=os.getenv("WANDB_API_KEY"))
    avail_entities = api.viewer.teams 

    if training_config.logging.wandb_entity not in avail_entities:
        raise ValueError(f"You don't have access to the {training_config.logging.wandb_entity} entity on Weights & Biases.")

    return True


def login_checks(training_config: TrainingConfig) -> None:
    """
    Checks that the user is logged into HuggingFace and Weights & Biases and has access to 
    the specified organization or entity. 
    """
    # Weights & Biases 
    if training_config.logging.experiment_tracker == "wandb":
        assert is_logged_into_wandb(training_config), "Please login to Weights & Biases to continue!"

    # HuggingFace 
    if training_config.checkpointing.hf_repo_id is not None:
        assert is_logged_into_huggingface(training_config), "Please login to HuggingFace to continue!"
