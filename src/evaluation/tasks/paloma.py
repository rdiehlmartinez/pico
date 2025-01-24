"""
Paloma is a comprehensive evaluation benchmark for large language models (LLMs) that focuses
on measuring perplexity across diverse text domains.

To evaluate on Paloma, we use the huggingface evaluation framework.

For more details, see: https://huggingface.co/datasets/allenai/paloma
"""

import evaluate
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar, enable_progress_bar

# typing imports
from src.config.evaluation_config import PalomaEvaluationConfig


def run_paloma_evaluation(
    model_path: str,
    paloma_config: PalomaEvaluationConfig,
) -> None:
    """Run Perplexity evaluation on the Paloma evaluation dataset.

    We use the HuggingFace evaluate library to load in and compute the perplexity metric.

    Args:
        model_path (str): Path to the model checkpoint to be evaluated
        paloma_config (PalomaEvaluationConfig): Configuration for Paloma evaluation
    """

    disable_progress_bar()

    perplexity = evaluate.load("pico-lm/perplexity")
    dataset = load_dataset(
        paloma_config.dataset_name, split=paloma_config.dataset_split
    )["text"]
    perplexity_result = perplexity.compute(
        model_id=model_path,
        predictions=dataset,
        add_start_token=False,
        max_length=paloma_config.max_length,
        batch_size=paloma_config.batch_size,
        trust_remote_code=True,
    )

    mean_perplexity = perplexity_result["mean_perplexity"]

    enable_progress_bar()
    return mean_perplexity
