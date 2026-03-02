# mt_attention_preln.py
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MTEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, dropout: float = 0.1, debug: bool = False):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.debug = debug
        self._printed = False

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.wte(idx)
        x = self.drop(x)
        if self.debug and not self._printed:
            print(f"[MTEmbedding] idx={tuple(idx.shape)} -> x={tuple(x.shape)} (no absolute pos)")
            self._printed = True
        return x


def skew(QEr: torch.Tensor) -> torch.Tensor:
    """
    Skewing procedure (Music Transformer):
      QEr: (B, H, T, 2T-1)  donde la última dim indexa (j-i) + (T-1)
      -> S_rel: (B, H, T, T) alineado por (i,j)

    Implementación estándar: pad + reshape + slice.
    """
    B, H, T, M = QEr.shape
    assert M == 2 * T - 1, f"Esperaba 2T-1, recibido {M} con T={T}"

    x = F.pad(QEr, (1, 0))         # (B,H,T,2T)
    x = x.view(B, H, 2 * T, T)     # (B,H,2T,T)
    x = x[:, :, 1:, :]             # (B,H,2T-1,T)
    x = x.view(B, H, T, 2 * T - 1) # (B,H,T,2T-1)
    x = x[:, :, :, :T]             # (B,H,T,T)
    return x


class RelativeMaskedMHA_PreLN(nn.Module):
    """
    scores = (QK^T + S_rel) / sqrt(d_h)
    con S_rel calculado via skew(QEr) y embeddings relativos (2*max_seq_len-1).
    """
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1, bias: bool = True, debug: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.debug = debug
        self._printed = False

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

        # Embeddings relativos para distancias en [-(L-1), ..., 0, ..., +(L-1)]
        # tamaño = 2*max_seq_len - 1
        self.rel_emb = nn.Embedding(2 * max_seq_len - 1, self.d_head)

        # Máscara causal: True = futuro prohibido
        causal = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, T, D = x.shape
        if T > self.max_seq_len:
            raise ValueError(f"T={T} > max_seq_len={self.max_seq_len}")

        if self.debug and not self._printed:
            print(f"[RelMHA] x={tuple(x.shape)} heads={self.n_heads} d_head={self.d_head} max_seq_len={self.max_seq_len}")
            self._printed = True

        qkv = self.qkv(x)
        q, k, v = qkv.split(D, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,T,Dh)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # 1) content: QK^T
        content = q @ k.transpose(-2, -1)  # (B,H,T,T)

        # 2) relative: QEr^T -> skew
        # Para una longitud T, necesitamos 2T-1 embeddings centrados en 0
        # Índice r = (j-i) + (T-1)
        center = self.max_seq_len - 1
        start = center - (T - 1)
        end   = center + (T - 1) + 1
        Er = self.rel_emb.weight[start:end]             # (2T-1, Dh)

        QEr = torch.matmul(q, Er.transpose(0, 1))       # (B,H,T,2T-1)
        rel = skew(QEr)                                 # (B,H,T,T)

        scores = (content + rel) / math.sqrt(self.d_head)

        # causal mask
        scores = scores.masked_fill(self.causal_mask[:T, :T][None, None, :, :], float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        y = self.out_drop(self.out_proj(y))

        return (y, attn) if return_attn else (y, None)


class PreLNResidual(nn.Module):
    """
    Pre-LN wrapper: y = x + Dropout( sublayer(LN(x)) )
    """
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: nn.Module, return_attn: bool = False):
        x_norm = self.ln(x)
        out, attn = sublayer(x_norm, return_attn=return_attn)
        return x + self.drop(out), attn