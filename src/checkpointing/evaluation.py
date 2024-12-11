"""
Utilities for checkpointing evaluation-related states (i.e. evaluation results, etc.)

We save the evaluation results in a JSON file at the step-specific evaluation results directory.
"""

import os
import json
from huggingface_hub import upload_folder

# typing imports
from typing import Dict, Any
from src.config import EvaluationConfig
from lightning.fabric import Fabric


def save_evaluation_results(
    evaluation_config: EvaluationConfig,
    fabric: Fabric,
    evaluation_results: Dict[str, Any],
    step: int,
) -> None:
    """Save evaluation results to disk and optionally to HuggingFace Hub.

    The evaluation results are saved in the following directory structure:
    {training_config.runs_dir}/
        └── {evaluation_config.run_name}/
            └── {evaluation_config.eval_results_dir}/
                └── step_{step}.json

    Args:
        evaluation_config: Configuration object containing evaluation settings
        fabric: Lightning Fabric instance
        evaluation_results: Dictionary containing evaluation metrics
        step: Current training step
    """

    # Only save on rank 0 to avoid conflicts

    run_dir = os.path.join(evaluation_config.runs_dir, evaluation_config.run_name)
    eval_results_dir = os.path.join(run_dir, evaluation_config.eval_results_dir)

    if fabric.global_rank == 0:
        os.makedirs(eval_results_dir, exist_ok=True)

        curr_eval_results_path = os.path.join(eval_results_dir, f"step_{step}.json")

        # save out as json
        with open(curr_eval_results_path, "w") as f:
            json.dump(evaluation_results, f)

        if evaluation_config.save_checkpoint_repo_id is not None:
            upload_folder(
                folder_path=eval_results_dir,
                path_in_repo=evaluation_config.eval_results_dir,
                repo_id=evaluation_config.save_checkpoint_repo_id,
                commit_message=f"Saving Evaluation Results -- Step {step}",
                revision=evaluation_config.run_name,
                token=os.getenv("HF_TOKEN"),
            )

    fabric.barrier()
