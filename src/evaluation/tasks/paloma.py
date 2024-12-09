"""
Paloma is a comprehensive evaluation benchmark for large language models (LLMs) that focuses
on measuring perplexity across diverse text domains.

To evaluate on Paloma, we use the huggingface evaluation framework.

For more details, see: https://huggingface.co/datasets/allenai/paloma
"""

import evaluate
from datasets import load_dataset
from src.config.evaluation_config import PalomaEvaluationConfig

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
        dataset = load_dataset("allenai/paloma", sub_config, split="val")["text"][:5]
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
