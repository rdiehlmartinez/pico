"""
Evaluation Config

Specifies the hyperparameters for the evaluation process, i.e. what metrics to compute, etc.
"""

from dataclasses import dataclass, field
from typing import Optional, List

from ._constants import MAX_SEQ_LEN, RUNS_DIR, EVAL_RESULTS_DIR, CHECKPOINT_DIR


@dataclass
class PalomaEvaluationConfig:
    max_length: int = MAX_SEQ_LEN
    batch_size: int = 16


@dataclass
class EvaluationConfig:
    # Name of the run to evaluate
    run_name: Optional[str] = None

    # Path to load a checkpoint from a local path
    checkpoint_path: Optional[str] = None

    # HuggingFace Hub Configs - set to None to not push to HuggingFace Hub
    # Should be in the format of <(username or )>/<repo_name>, e.g. pico-lm/pico-7b
    save_checkpoint_repo_id: Optional[str] = "pico-lm/demo"

    # Evaluation metrics to compute: by default, we compute the perplexity of the model
    evaluation_metrics: List[str] = field(default_factory=lambda: ["paloma"])

    # NOTE: Add other evaluation configs here
    # Each evaluation metric should have its own config
    paloma: PalomaEvaluationConfig = field(default_factory=PalomaEvaluationConfig)

    # Directories used to store training runs, checkpoints, logs, and evaluation results
    runs_dir: str = RUNS_DIR
    checkpoints_dir: str = CHECKPOINT_DIR
    eval_results_dir: str = EVAL_RESULTS_DIR
