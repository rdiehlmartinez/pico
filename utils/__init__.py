ROOT_DIR = "runs"
CHECKPOINT_DIR = "checkpoints"

from config import TrainingConfig

def is_logged_into_huggingface(training_config: TrainingConfig) -> bool:
    """Checks if the user is logged into HuggingFace."""
    # check that the user has write access to https://huggingface.co/pico-lm
    from huggingface_hub import HfApi

    # Get the token from the environment or the cache
    token = HfApi().get_token()
    if not token:
        raise ValueError("No HuggingFace token found. You might need to run ./setup.sh or set the HF_TOKEN environment variable.")

    target_org = training_config.checkpointing.hf_repo_id.split("/")[0]
        
    api = HfApi(token=token)
    who_ami = api.whoami()

    # check if who_ami has access to org pico-lm
    orgs = who_ami.get('orgs', []) + [who_ami['name']]

    for org in orgs:
        if org['id'] == target_org:
            if org['role'] not in ['admin', 'write']:
                raise ValueError(f"You don't have write access to the {target_org} organization on HuggingFace.")
            else:
                return True
    
    # the target_org might be the user's username
    if who_ami['name'] == target_org:
        return True

    raise ValueError(
        f"You don't have access to the {target_org} organization on HuggingFace or the organization does not exist."
    )

def is_logged_into_wandb(training_config: TrainingConfig) -> bool:
    """Checks if the user is logged into Weights & Biases."""
    import wandb

    if wandb.api.api_key is None:
        print("No Weights & Biases API key found. You might need to run ./setup.sh or set the WANDB_API_KEY environment variable.")
        return False

    api = wandb.Api()
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
