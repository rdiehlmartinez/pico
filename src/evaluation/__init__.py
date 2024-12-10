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
from src.config import EvaluationConfig
from lightning.fabric import Fabric


def run_evaluation(evaluation_config: EvaluationConfig, fabric: Fabric) -> None:
    """Run model evaluation using specified metrics in `evaluation_config`.

    This function orchestrates the complete evaluation pipeline by:
    1. Resolving the model checkpoint path (either specified or latest)
    2. Running possible setup steps for the evaluation metric
    3. Executing each requested evaluation metric (e.g. Paloma)
    4. Aggregating results across all metrics

    Args:
        evaluation_config (EvaluationConfig): Configuration object containing:
            - checkpoint_path (Optional[str]): Specific checkpoint to evaluate
                If None, uses the latest checkpoint
            - run_name (str): Name of the evaluation run
            - evaluation_metrics (List[str]): Metrics to evaluate; each metric should have its
                own config. Currently supported: ["paloma"];
            - paloma (PalomaConfig): Configuration for Paloma evaluation
                - max_length (int): Maximum sequence length
                - limit_eval_examples (Optional[int]): Number of examples to evaluate
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
                evaluation_metrics=["paloma"],
                paloma=PalomaConfig(max_length=2048, batch_size=16)
            )
        )

    Note:
        The function automatically handles checkpoint resolution, directory
        creation, and cleanup of temporary files. For each metric, it ensures
        proper setup and teardown of evaluation environment.
    """

    if fabric.global_rank != 0:
        fabric.barrier()
        return

    if evaluation_config.checkpoint_path is not None:
        model_path = evaluation_config.checkpoint_path
    else:
        run_name = evaluation_config.run_name
        model_path = f"{os.getcwd()}/{evaluation_config.runs_dir}/{run_name}/{evaluation_config.checkpoints_dir}/latest"
    os.makedirs(model_path, exist_ok=True)

    evaluation_results = {}

    for metric in evaluation_config.evaluation_metrics:
        # NOTE: add your own metrics here
        if metric == "paloma":
            paloma_result = run_paloma_evaluation(model_path, evaluation_config.paloma)
        else:
            raise ValueError(f"Metric {metric} not supported")

        evaluation_results[metric] = paloma_result

    fabric.barrier()

    return evaluation_results
