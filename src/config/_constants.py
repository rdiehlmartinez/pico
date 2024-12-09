"""
Constants used throughout the codebase
"""

# Basic Training Constants used throughout the codebase
VOCAB_SIZE = 50304
MAX_SEQ_LEN = 2048
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = (
    1  # NOTE: If you need to make the batch size fit in memory, you can play with this.
)

# Directories used to store training runs, checkpoints, logs, and evaluation results
RUNS_DIR = "runs"
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
FABRIC_CHECKPOINT_DIR = "fabric_state"
EVAL_RESULTS_DIR = "eval_results"
