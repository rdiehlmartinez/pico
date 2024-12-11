"""
Data Config

Specifies the hyperparameters for the dataset, dataloader, and tokenizer.
"""

from dataclasses import dataclass, field

from ._constants import VOCAB_SIZE, BATCH_SIZE, MAX_SEQ_LEN, GRADIENT_ACCUMULATION_STEPS


@dataclass
class _DatasetConfig:
    name: str = "pico-lm/pretokenized-dolma"


@dataclass
class _DataLoaderConfig:
    batch_size: int = BATCH_SIZE // GRADIENT_ACCUMULATION_STEPS
    max_seq_len: int = MAX_SEQ_LEN


class _TokenizerConfig:
    name: str = "allenai/OLMo-7B-0724-hf"
    vocab_size: int = VOCAB_SIZE


@dataclass
class DataConfig:
    dataset: _DatasetConfig = field(default_factory=_DatasetConfig)
    dataloader: _DataLoaderConfig = field(default_factory=_DataLoaderConfig)
    tokenizer: _TokenizerConfig = field(default_factory=_TokenizerConfig)
