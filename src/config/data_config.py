"""
Data Config

Specifies the hyperparameters for the dataset, dataloader, and tokenizer.
"""

from dataclasses import dataclass, field

from ._constants import VOCAB_SIZE, BATCH_SIZE, MAX_SEQ_LEN, GRADIENT_ACCUMULATION_STEPS


@dataclass
class DatasetConfig:
    name: str = "pico-lm/pretokenized-dolma"


@dataclass
class DataLoaderConfig:
    full_batch_size: int = BATCH_SIZE
    sub_batch_size: int = BATCH_SIZE // GRADIENT_ACCUMULATION_STEPS
    max_seq_len: int = MAX_SEQ_LEN


@dataclass
class TokenizerConfig:
    name: str = "allenai/OLMo-7B-0724-hf"
    vocab_size: int = VOCAB_SIZE


@dataclass
class DataConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
