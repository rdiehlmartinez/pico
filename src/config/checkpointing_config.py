"""
Checkpointing Config

Specifies the hyperparameters for the checkpointing process; checkpointing is used to save
the model and optimizer states, as well as the learning dynamics metrics.
"""

from dataclasses import dataclass, field
from typing import Optional, List

from ._constants import (
    RUNS_DIR,
    CHECKPOINTS_DIR,
    LOGS_DIR,
    FABRIC_CHECKPOINT_DIR,
    EVAL_RESULTS_DIR,
    MAX_SEQ_LEN,
    LEARNING_DYNAMICS_DIR,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
)


@dataclass
class TrainingCheckpointingConfig:
    load_checkpoint_path: Optional[str] = None
    load_latest_checkpoint: bool = False


@dataclass
class EvaluationCheckpointingConfig:
    load_checkpoint_path: Optional[str] = None
    eval_results_dir: str = EVAL_RESULTS_DIR


@dataclass
class LearningDynamicsCheckpointingConfig:
    # Suffixes of the layers to compute learning dynamics for
    layer_suffixes: List[str] = field(
        default_factory=lambda: [
            "attention.v_proj",
            "attention.o_proj",
            "feed_forward.w_2",
        ]
    )

    # Sequence index at which to extract hidden states; by default, we extract the hidden states
    # at the last token of the sequence
    sequence_idx: int = MAX_SEQ_LEN - 1

    # size of the sub-batch used for extracting learning dynamics states
    sub_batch_size: int = BATCH_SIZE // GRADIENT_ACCUMULATION_STEPS

    # Path to the evaluation data batch - used across learning dynamics checkpointing for consistency
    eval_data_batch: Optional[str] = "pico-lm/pretokenized-eval-batch"


@dataclass
class CheckpointingConfig:
    # Name of the run
    run_name: Optional[str] = None

    runs_dir: str = RUNS_DIR
    checkpoints_dir: str = CHECKPOINTS_DIR
    logs_dir: str = LOGS_DIR
    fabric_checkpoint_dir: str = FABRIC_CHECKPOINT_DIR
    learning_dynamics_dir: str = LEARNING_DYNAMICS_DIR
    # How often to save checkpoints
    save_every_n_steps: int = 20

    # Should be in the format of <(username or )>/<repo_name>, e.g. pico-lm/pico-7b
    save_checkpoint_repo_id: Optional[str] = "pico-lm/demo"

    training: TrainingCheckpointingConfig = field(
        default_factory=TrainingCheckpointingConfig
    )
    evaluation: EvaluationCheckpointingConfig = field(
        default_factory=EvaluationCheckpointingConfig
    )
    learning_dynamics: LearningDynamicsCheckpointingConfig = field(
        default_factory=LearningDynamicsCheckpointingConfig
    )
