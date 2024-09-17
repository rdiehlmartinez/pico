""" 
A simple, SOTA language model with all the bells and whistles. Just simplified. 

Build on top of this however you like. 

Adapted and simplified from: 
OLMO: https://github.com/allenai/OLMo/blob/main/olmo/model.py
LLAMA 3: https://github.com/meta/llama-3/blob/main/model.py
"""

import math

import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F

from config import PicoConfig
from typing import Tuple, Optional

########################################################
#
# Layer Normalization 
#
########################################################

class RMSNorm(torch.nn.Module):
    def __init__(self, config: PicoConfig):
        super().__init__()
        self.eps = config.norm.eps
        self.weight = nn.Parameter(torch.ones(config.model.d_model))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

########################################################
#
# Positional Embedding
#
########################################################

class RoPE(nn.Module):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).

    Mostly taken from LLaMA 3 implementation.
    """

    _freqs_cis: torch.Tensor = None

    # TODO: implement config 
    def __init__(self, config: PicoConfig):
        """
        Rotary positional embeddings (RoPE). 
        """
        super().__init__()

        self.theta = config.position_emb.theta
        self.dim = config.model.d_model // config.model.n_heads

        max_seq_len = config.model.max_seq_len

        # only gets set once, and then reused for all RoPE instances
        if RoPE._freqs_cis is None:
            RoPE._freqs_cis = self._setup_freqs_cis(max_seq_len, self.theta, self.dim) 

    @classmethod
    def _setup_freqs_cis(cls, seq_len: int, theta: float, dim: int) -> torch.Tensor:
        """
        Sets up the complex frequency tensor that is used to compute the RoPE embeddings. 

        Note other implementations will use cos and sin directly, but using the complex 
        number representation is (probably?) more efficient: 

            e^(theta * i * t) = cos(theta * t) + i * sin(theta * t) [Euler's formula]
        """
        _freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(seq_len,)
        freqs = torch.outer(t, _freqs)
        return torch.polar(torch.ones_like(freqs), freqs)  # complex64

    def get_freqs_cis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshapes the frequency tensor to be broadcastable with the input tensor.
        """
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert RoPE._freqs_cis.shape == (x.shape[1], x.shape[-1])

        # TODO: Check whether this is correct (might be able to remove this)
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return RoPE._freqs_cis.view(*shape)
    

    def apply_rotary_emb(
        self, 
        queries: torch.Tensor,
        keys: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the rotary positional embeddings to the input tensors via complex num multiplication
        """
        queries_ = torch.view_as_complex(queries.float().reshape(*queries.shape[:-1], -1, 2))
        keys_ = torch.view_as_complex(keys.float().reshape(*keys.shape[:-1], -1, 2))
        freqs_cis = self.get_freqs_cis(queries)
        queries_rotated = torch.view_as_real(queries_ * freqs_cis).flatten(3)
        keys_rotated = torch.view_as_real(keys_ * freqs_cis).flatten(3)
        return queries_rotated.type_as(queries), keys_rotated.type_as(keys)

########################################################
#
# Attention
#
########################################################

class Attention(nn.Module):
    def __init__(self, config: PicoConfig):
        super().__init__()

        self.n_heads = config.model.n_heads
        self.n_kv_heads = config.model.n_kv_heads

        self.max_batch_size = config.model.max_batch_size
        self.max_seq_len = config.model.max_seq_len

        self.head_dim = config.model.d_model // self.n_heads

        self.n_rep = self.n_local_heads // self.n_kv_heads

        self.q_proj = nn.Linear(self.head_dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.head_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.head_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.head_dim, bias=False)

        self.rope = RoPE(config)

        # caches for inference; only used if inference_mode is enabled
        self.k_cache = None
        self.v_cache = None

    def repeat_kv(self, original_tensor: torch.Tensor) -> torch.Tensor:
        """
        Repeats the keys and values for the attention mechanism.
        """
        bsz, seq_len, n_kv_heads, head_dim = original_tensor.shape
        if self.n_rep == 1:
            return original_tensor
        return (
            original_tensor[:, :, :, None, :]  # Add a new dimension after n_kv_heads
            .expand(bsz, seq_len, n_kv_heads, self.n_rep, head_dim)  # Expand this new dimension to size n_rep
            .reshape(bsz, seq_len, n_kv_heads * self.n_rep, head_dim)  # Flatten this new dimension into n_kv_heads
        )

    def forward(
        self,
        input: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inference_mode: Optional[bool] = False,
        start_pos: Optional[int] = 0,
    ):
        """
        Args:
            input: input tensor of shape (batch_size, sequence_length, d_model)
            mask: attention mask of shape (batch_size, sequence_length, sequence_length)
            inference_mode: whether to use inference mode; in this mode, we enable use of the cache
            start_pos: start position of the sequence, ignored unless inference_mode is enabled
        """
        bsz, seq_len, _ = input.shape
        _queries, _keys, _values = self.q_proj(input), self.k_proj(input), self.v_proj(input)

        _queries = _queries.view(bsz, seq_len, self.n_heads, self.head_dim)
        _keys = _keys.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        _values = _values.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        # apply rotary positional embeddings
        _queries, _keys = self.rope.apply_rotary_emb(_queries, _keys)

        if inference_mode and start_pos > 0:
            if self.k_cache is None and self.v_cache is None:
                self.k_cache = torch.zeros((self.max_batch_size, self.max_seq_len, self.n_kv_heads, self.head_dim))
                self.v_cache = torch.zeros((self.max_batch_size, self.max_seq_len, self.n_kv_heads, self.head_dim))

            self.cache_k[:bsz, start_pos : start_pos + seq_len] = _keys
            self.cache_v[:bsz, start_pos : start_pos + seq_len] = _values

            keys = self.cache_k[:bsz, : start_pos + seq_len]
            values = self.cache_v[:bsz, : start_pos + seq_len]
        else: 
            keys = _keys
            values = _values

        # repeat k/v heads if n_kv_heads < n_heads
        keys = self.repeat_kv(keys)  # (bs, (cache_len) + seq_len, n_heads, head_dim)
        values = self.repeat_kv(values)  # (bs, (cache_len) + seq_len, n_heads, head_dim)

        queries = _queries.transpose(1, 2)  # (bs, n_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_heads, (cache_len) + seq_len, head_dim)
        values = values.transpose(1, 2)  # (bs, n_heads, (cache_len) + seq_len, head_dim)

        scores = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_heads, seq_len, (cache_len) + seq_len)
        scores = F.softmax(scores.float(), dim=-1).type_as(queries)
        output = torch.matmul(scores, values)  # (bs, n_heads, seq_len, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.wo(output)

########################################################
#
# SwiGLU (Combines MLP and Activation)
#
########################################################

class SwiGLU(nn.Module):
    def __init__(self, config: PicoConfig):
        super().__init__()

        model_dim = config.model.d_model
        act_hidden_dim = config.activation.act_hidden_dim # usually 4 * d_model 

        self.w_0 = nn.Linear(model_dim, act_hidden_dim, bias=False)
        self.w_1 = nn.Linear(model_dim, act_hidden_dim, bias=False)    
        self.w_2 = nn.Linear(act_hidden_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.w_2(F.silu(self.w_0(x)) * self.w_1(x))

########################################################
#
# PicoBlock and the Pico Model 
#
########################################################

class PicoBlock(nn.Module):
    def __init__(self, config: PicoConfig):
        super().__init__()

        self.attention = Attention(config)
        self.feed_forward = SwiGLU(config)
        self.attention_norm = RMSNorm(config)
        self.swiglu_norm = RMSNorm(config)

    def forward(
        self,
        input: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inference_mode: Optional[bool] = False,
        start_pos: Optional[int] = 0,
    ):
        h = input + self.attention(self.attention_norm(input), mask, inference_mode, start_pos)
        out = h + self.feed_forward(self.swiglu_norm(h))
        return out

class Pico(nn.Module):

    def __init__(self, config: PicoConfig):
        super().__init__()
        self.config = config

        self.vocab_size = config.tokenizer.vocab_size
        self.n_layers = config.model.n_layers

        self.embedding_proj = nn.Embedding(self.vocab_size, config.model.d_model)
        
        self.layers = nn.ModuleList([PicoBlock(config) for _ in range(self.n_layers)])

        self.output_norm = RMSNorm(config.model.d_model, eps=config.norm.eps)

        # NOTE: the de-embedding projection is not tied to the embedding projection
        self.de_embedding_proj = nn.Linear(config.model.d_model, self.vocab_size, bias=False)

    def forward(
            self, 
            input_ids: torch.Tensor, 
            inference_mode: Optional[bool] = False, 
            start_pos: Optional[int] = 0
        ):
        seq_len = input_ids.shape[-1]

        h = self.embedding_proj(input_ids)

        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)

            if inference_mode:
                # when inference, we only want to attend to the new tokens (after start_pos)
                mask = torch.hstack([torch.zeros((seq_len, start_pos)), mask]).type_as(h)

        for layer in self.layers:
            h = layer(h, mask, inference_mode, start_pos)
        h = self.output_norm(h)
        output = self.de_embedding_proj(h).float()
        return output
