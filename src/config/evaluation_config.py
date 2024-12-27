"""
Evaluation Config

Specifies the hyperparameters for the evaluation process, i.e. what metrics to compute, etc.
"""

from dataclasses import dataclass, field
from typing import Optional, List

from src.config._constants import MAX_SEQ_LEN, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS


@dataclass
class PalomaEvaluationConfig:
    max_length: int = MAX_SEQ_LEN
    batch_size: int = BATCH_SIZE // GRADIENT_ACCUMULATION_STEPS


@dataclass
class EvaluationConfig:
    # Evaluation metrics to compute: by default, we compute the perplexity of the model
    metrics: Optional[List[str]] = field(default_factory=lambda: ["paloma"])

    # NOTE: Add other evaluation configs here
    # Each evaluation metric should have its own config
    paloma: PalomaEvaluationConfig = field(default_factory=PalomaEvaluationConfig)
