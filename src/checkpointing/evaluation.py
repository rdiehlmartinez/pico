"""
Utilities for checkpointing evaluation-related states (i.e. evaluation results, etc.)

We save the evaluation results in a JSON file at the step-specific evaluation results directory.
"""

import os
import json
from huggingface_hub import upload_folder

# typing imports
from typing import Dict, Any
from src.config import CheckpointingConfig
from lightning.fabric import Fabric


def save_evaluation_results(
    checkpointing_config: CheckpointingConfig,
    fabric: Fabric,
    evaluation_results: Dict[str, Any],
    gradient_step: int,
) -> None:
    """Save evaluation results to disk and optionally to HuggingFace Hub.

    The evaluation results are saved in the following directory structure:
    {checkpointing_config.runs_dir}/
        └── {checkpointing_config.run_name}/
            └── {checkpointing_config.eval_results_dir}/
                └── step_{gradient_step}.json

    Args:
        checkpointing_config: Configuration object containing checkpoint settings
        fabric: Lightning Fabric instance
        evaluation_results: Dictionary containing evaluation metrics
        gradient_step: Current training gradient step (i.e. number of learning steps taken)
    """

    # Only save on rank 0 to avoid conflicts
    if fabric.global_rank != 0:
        return

    run_dir = os.path.join(checkpointing_config.runs_dir, checkpointing_config.run_name)
    eval_results_dir = os.path.join(
        run_dir, checkpointing_config.evaluation.eval_results_dir
    )

    os.makedirs(eval_results_dir, exist_ok=True)

    curr_eval_results_path = os.path.join(
        eval_results_dir, f"step_{gradient_step}.json"
    )

    # save out as json
    with open(curr_eval_results_path, "w") as f:
        json.dump(evaluation_results, f)

    if checkpointing_config.save_checkpoint_repo_id is not None:
        upload_folder(
            folder_path=eval_results_dir,
            path_in_repo=checkpointing_config.evaluation.eval_results_dir,
            repo_id=checkpointing_config.save_checkpoint_repo_id,
            commit_message=f"Saving Evaluation Results -- Step {gradient_step}",
            revision=checkpointing_config.run_name,
            token=os.getenv("HF_TOKEN"),
        )
