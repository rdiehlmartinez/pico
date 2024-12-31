"""
Constants used throughout the codebase
"""

# Basic Training Constants used throughout the codebase
VOCAB_SIZE = 50304
MAX_SEQ_LEN = 2048
BATCH_SIZE = 1024
GRADIENT_ACCUMULATION_STEPS = 128

# Directories used to store training runs, checkpoints, logs, and evaluation results
RUNS_DIR = "runs"
CHECKPOINTS_DIR = "checkpoints"
LOGS_DIR = "logs"
FABRIC_CHECKPOINT_DIR = "fabric_state"
FABRIC_CHECKPOINT_FILENAME = "checkpoint.pt"
LEARNING_DYNAMICS_DIR = "learning_dynamics"
EVAL_RESULTS_DIR = "eval_results"
