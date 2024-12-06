"""
Evaluation Module for Pico Language Model

This module implements the evaluation pipeline for the Pico language model. It provides
functionality to evaluate model performance using various metrics and handles the complete
evaluation workflow.

NOTE: out of the box we only support Paloma, but the structure is designed to be flexible and
you are meant to add whatever metrics you want. One of the main reasons we store out
the model in the HuggingFace format is so that its easy to use third-party evaluation
libraries/frameworks.

Main Workflow:
1. Setup evaluation configuration
2. Run evaluation using specified metrics
3. Process and aggregate results
4. Clean up temporary files and workspaces
"""

from datasets import load_dataset
import evaluate

import os
from config import EvaluationConfig, PalomaEvaluationConfig

from . import RUNS_DIR, CHECKPOINT_DIR


########################################################
#
# Paloma Evaluation
#
########################################################

"""
Paloma is a comprehensive evaluation benchmark for large language models (LLMs) that focuses 
on measuring perplexity across diverse text domains. 

To evaluate on Paloma, we use the huggingface evaluation framework.

For more details, see: https://huggingface.co/datasets/allenai/paloma
"""

PALOMA_SUB_CONFIGS = [
    "4chan_meta_sep",
    "c4_100_domains",
    "c4_en",
    "dolma_100_programing_languages",
    "dolma_100_subreddits",
    "dolma-v1_5",
    "falcon-refinedweb",
    "gab",
    "m2d2_s2orc_unsplit",
    "m2d2_wikipedia_unsplit",
    "manosphere_meta_sep",
    "mc4",
    "ptb",
    "redpajama",
    "twitterAAE_HELM_fixed",
    "wikitext_103",
]


def run_paloma_evaluation(
    model_path: str, paloma_config: PalomaEvaluationConfig
) -> None:
    """Run Perplexity evaluation on the Paloma evaluation dataset. We use the HuggingFace
    evaluate library to load in and compute the perplexity metric.

    Args:
        model_path (str): Path to the model checkpoint to be evaluated
        paloma_config (PalomaEvaluationConfig): Configuration for Paloma evaluation
    """

    # Load in custom perplexity metric (this is just a fork of the normal perplexity metric
    # that makes it possible to pass `trust_remote_code=True` to the `compute` method)
    perplexity = evaluate.load("pico-lm/perplexity")

    perplexity_results = {}
    perplexity_counts = {}

    for sub_config in PALOMA_SUB_CONFIGS:
        dataset = load_dataset("allenai/paloma", sub_config, split="val")["text"]
        perplexity_result = perplexity.compute(
            model_id=model_path,
            predictions=dataset,
            add_start_token=False,
            max_length=paloma_config.max_length,
            batch_size=paloma_config.batch_size,
            trust_remote_code=True,
        )
        perplexity_results[sub_config] = perplexity_result["mean_perplexity"]
        perplexity_counts[sub_config] = len(dataset)

    # return micro average perplexity
    return sum(
        perplexity_results[sub_config] * perplexity_counts[sub_config]
        for sub_config in PALOMA_SUB_CONFIGS
    ) / sum(perplexity_counts.values())


########################################################
#
# Evaluation
#
########################################################


def run_evaluation(evaluation_config: EvaluationConfig) -> None:
    """Run model evaluation using specified metrics.

    This function orchestrates the complete evaluation pipeline by:
    1. Resolving the model checkpoint path (either specified or latest)
    2. Running possible setup steps for the evaluation metric
    3. Executing each requested evaluation metric
    4. Aggregating results across all metrics

    Args:
        evaluation_config (EvaluationConfig): Configuration object containing:
            - checkpoint_path (Optional[str]): Specific checkpoint to evaluate
                If None, uses the latest checkpoint
            - run_name (str): Name of the evaluation run
            - evaluation_metrics (List[str]): Metrics to evaluate
                Currently supported: ["paloma"]
            - paloma (PalomaConfig): Configuration for Paloma evaluation
                - max_length (int): Maximum sequence length
                - limit_eval_examples (Optional[int]): Number of examples to evaluate

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

    if evaluation_config.checkpoint_path is not None:
        model_path = evaluation_config.checkpoint_path
    else:
        run_name = evaluation_config.run_name
        model_path = f"{os.getcwd()}/{RUNS_DIR}/{run_name}/{CHECKPOINT_DIR}/latest"
    os.makedirs(model_path, exist_ok=True)

    evaluation_results = {}

    for metric in evaluation_config.evaluation_metrics:
        if metric == "paloma":
            paloma_result = run_paloma_evaluation(model_path, evaluation_config.paloma)
        else:
            raise ValueError(f"Metric {metric} not supported")

        evaluation_results[metric] = paloma_result

    return evaluation_results
