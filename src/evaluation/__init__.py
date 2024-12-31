"""
Pico Evaluation Package

This package implements the evaluation pipeline for the Pico language model. It provides
functionality to evaluate model performance using various metrics and handles the complete
evaluation workflow.

We recommend that each evaluation metric should have its own config, and should be
implemented as a module in the `evaluation/tasks` directory that exposes a `run_<metric_name>` function.

NOTE: Out of the box we only support Paloma, but the structure is designed to be flexible and
you are meant to add whatever metrics you want. One of the main reasons we store out
the model in the HuggingFace format is so that its easy to use third-party evaluation
libraries/frameworks.
"""

import os
from .tasks.paloma import run_paloma_evaluation

# typing imports
from src.config import EvaluationConfig, CheckpointingConfig
from lightning.fabric import Fabric


def run_evaluation(
    evaluation_config: EvaluationConfig,
    checkpointing_config: CheckpointingConfig,
    fabric: Fabric,
) -> None:
    """Run model evaluation using specified metrics in `evaluation_config`.

    This function orchestrates the complete evaluation pipeline by:
    1. Resolving the model checkpoint path (either specified or latest) to load the model from;
        during training, this is the path to the latest checkpoint in the run directory.
    2. Iterating over each evaluation metric, and running the corresponding evaluation function.
        NOTE: we suggest you follow the pattern of the Paloma evaluation function, and implement
        your own evaluation function for each metric in the `evaluation/tasks` directory.
    3. Aggregating results across all metrics in a dictionary, and returning it.

    Args:
        evaluation_config (EvaluationConfig): Configuration object containing:
            - metrics (List[str]): Metrics to evaluate; each metric should have its
                own config. Currently supported: ["paloma"];
            - paloma (PalomaConfig): Configuration for Paloma evaluation
                - max_length (int): Maximum sequence length
                - limit_eval_examples (Optional[int]): Number of examples to evaluate
        checkpointing_config (CheckpointingConfig): Configuration object containing:
        fabric (Fabric): Lightning Fabric instance

    Returns:
        Dict[str, float]: Dictionary mapping metric names to their values
            Example: {"paloma": 3.45}

    Raises:
        ValueError: If an unsupported evaluation metric is requested

    Example:
        results = run_evaluation(
            EvaluationConfig(
                run_name="experiment_1",
                metrics=["paloma"],
                paloma=PalomaConfig(max_length=2048, batch_size=16)
            )
        )

    """

    if fabric.global_rank != 0:
        # NOTE: by default we only want to run evaluation on a single process; evaluation tasks
        # will typically be run using third-party libraries. These libraries should be in charge of
        # handling the distributed evaluation.
        return None

    if checkpointing_config.evaluation.load_checkpoint_path is not None:
        model_path = checkpointing_config.evaluation.load_checkpoint_path
    else:
        run_name = checkpointing_config.run_name
        model_path = f"{os.getcwd()}/{checkpointing_config.runs_dir}/{run_name}/{checkpointing_config.checkpoints_dir}/latest"
    os.makedirs(model_path, exist_ok=True)

    evaluation_results = {}

    for metric in evaluation_config.metrics:
        # NOTE: add your own metrics here
        if metric == "paloma":
            paloma_result = run_paloma_evaluation(model_path, evaluation_config.paloma)
        else:
            raise ValueError(f"Metric {metric} not supported")

        evaluation_results[metric] = paloma_result

    return evaluation_results
