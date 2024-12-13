"""
Constants used throughout the codebase
"""

# Basic Training Constants used throughout the codebase
VOCAB_SIZE = 50304
MAX_SEQ_LEN = 2048
BATCH_SIZE = 2
# NOTE: If you need to make the batch size fit in memory, you can play with this.
GRADIENT_ACCUMULATION_STEPS = 1

# Directories used to store training runs, checkpoints, logs, and evaluation results
RUNS_DIR = "runs"
CHECKPOINTS_DIR = "checkpoints"
LOGS_DIR = "logs"
FABRIC_CHECKPOINT_DIR = "fabric_state"
LEARNING_DYNAMICS_DIR = "learning_dynamics"
EVAL_RESULTS_DIR = "eval_results"
