"""
Pico Config Package

The modules of this package are where you can specify the hyperparameters for the Pico model,
the dataset, the training process, evaluation, etc.

As with anything else in Pico, we've designed for the configuration setup to be as flexible
as possible. By default the configs are implemented as vanilla dataclasses -- this makes it easy to
switch to different config management systems if you want, like hydra.

Some things to NOTE:
- All hyperparameters are initialized with default values, which can be overridden.
- The default vocab size is set to the size of the OLMo tokenizer.
"""

# ruff: noqa: F401

# For convenience, we export the config classes here
from .training_config import TrainingConfig
from .data_config import DataConfig
from .model_config import ModelConfig
from .evaluation_config import EvaluationConfig
