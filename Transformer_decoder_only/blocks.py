from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Config de bloques
# =========================
@dataclass
class MTConfig:
    vocab_size: int
    d_model: int
    n_heads: int
    max_seq_len: int            # equivalente a block_size
    dropout: float = 0.1
    bias: bool = True
    d_ff: Optional[int] = None  # None => 4*d_model
    debug: bool = False


# =========================
# Embedding
# =========================
class MTEmbedding(nn.Module):
    """
    Music Transformer style:
    - token embedding + dropout
    - NO positional encoding absoluto aquí
    """
    def __init__(self, vocab_size: int, d_model: int, dropout: float, debug: bool = False):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.debug = debug
        self._printed = False

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.wte(idx)   # (B,T,D)
        x = self.drop(x)
        if self.debug and not self._printed:
            print(f"[MTEmbedding] idx={tuple(idx.shape)} -> x={tuple(x.shape)} (no absolute pos)")
            self._printed = True
        return x


# =========================
# Skewing
# =========================
def skew(QEr: torch.Tensor) -> torch.Tensor:
    """
    Skewing procedure (Music Transformer):
      QEr: (B, H, T, 2T-1)   con índice r = (j-i) + (T-1)
      -> S_rel: (B, H, T, T) alineado por (i,j)

    Implementación estándar: pad + reshape + slice.
    """
    B, H, T, M = QEr.shape
    assert M == 2 * T - 1, f"Esperaba 2T-1, recibido {M} con T={T}"

    x = F.pad(QEr, (1, 0))     # (B,H,T,2T)
    x = x.view(B, H, 2 * T, T)      # (B,H,2T,T)
    x = x[:, :, 1:, :]              # (B,H,2T-1,T)
    x = x.view(B, H, T, 2 * T - 1)  # (B,H,T,2T-1)
    x = x[:, :, :, :T]              # (B,H,T,T)
    return x


# =========================
# Relative Masked Multi-Head Self-Attention (global)
# =========================
class RelativeMaskedMHA(nn.Module):
    """
    Con el objetivo de poder
    informar al meanismo de atención como de lejos estan dos posiciones dentro de una secuencia, se
    introduce S_rel (distancia relativa):
        scores = (QK^T + S_rel) / sqrt(d_head)
    donde S_rel = skew(Q * Er^T) con Er de tamaño (2T-1, d_head)

    Masked causal: se ocultan los valores futuros para que no puedan ser tenidos en cuenta
    """
    def __init__(self, cfg: MTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)

        self.attn_drop = nn.Dropout(cfg.dropout)
        self.out_drop = nn.Dropout(cfg.dropout)

        # embeddings relativos (2*L-1) para max_seq_len=L
        self.rel_emb = nn.Embedding(2 * cfg.max_seq_len - 1, self.d_head)

        # máscara causal: True = futuro prohibido
        causal = torch.triu(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal, persistent=False)

        self._printed = False

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x: (B,T,D)
        returns:
          y: (B,T,D)
          attn: (B,H,T,T) si return_attn
        """
        B, T, D = x.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"T={T} > max_seq_len={self.cfg.max_seq_len}")

        if self.cfg.debug and not self._printed:
            print(f"[RelMHA] x={tuple(x.shape)} heads={self.n_heads} d_head={self.d_head} max_seq_len={self.cfg.max_seq_len}")
            self._printed = True

        qkv = self.qkv(x)                 # (B,T,3D)
        q, k, v = qkv.split(D, dim=-1)    # (B,T,D)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,T,Dh)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # contenido
        content = q @ k.transpose(-2, -1)  # (B,H,T,T)

        # relativo: Er de tamaño (2T-1, Dh) centrado en 0
        center = self.cfg.max_seq_len - 1
        start = center - (T - 1)
        end = center + (T - 1) + 1
        Er = self.rel_emb.weight[start:end]             # (2T-1, Dh)

        QEr = torch.matmul(q, Er.transpose(0, 1))       # (B,H,T,2T-1)
        rel = skew(QEr)                                 # (B,H,T,T)

        scores = (content + rel) / math.sqrt(self.d_head)

        # máscara causal
        scores = scores.masked_fill(self.causal_mask[:T, :T][None, None, :, :], float("-inf"))

        attn = torch.softmax(scores, dim=-1)            # (B,H,T,T)
        attn = self.attn_drop(attn)

        y = attn @ v                                    # (B,H,T,Dh)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        y = self.out_drop(self.out_proj(y))

        if self.cfg.debug:
            # métrica no-trivial (fila t=10)
            trow = min(10, T - 1)
            row = attn[0, 0, trow, :trow + 1]
            ent = -(row * row.clamp_min(1e-12).log()).sum().item()
            print(f"[RelMHA] attn entropy(head0,t={trow})={ent:.4f} row_sum={row.sum().item():.4f}")

        return (y, attn) if return_attn else (y, None)


# =========================
# Feed Forward (FFN)
# =========================
class FeedForward(nn.Module):
    """
    FFN:
      Linear -> GELU -> Dropout -> Linear
    """
    def __init__(self, cfg: MTConfig):
        super().__init__()
        d_ff = cfg.d_ff if cfg.d_ff is not None else 4 * cfg.d_model
        self.fc1 = nn.Linear(cfg.d_model, d_ff, bias=cfg.bias)
        self.fc2 = nn.Linear(d_ff, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

        self.cfg = cfg
        self._printed = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.debug and not self._printed:
            print(f"[FFN] x={tuple(x.shape)} d_ff={self.fc1.out_features}")
            self._printed = True

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


# =========================
# Bloque completo Music Transformer (Pre-LN)
# =========================
class MusicTransformerBlockPreLN(nn.Module):
    """
    Pre-LN:
      x = x + Dropout( Attn( LN(x) ) )
      x = x + Dropout( FFN(  LN(x) ) )
    """
    def __init__(self, cfg: MTConfig):
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)

        self.attn = RelativeMaskedMHA(cfg)
        self.ffn = FeedForward(cfg)

        self.resid_drop1 = nn.Dropout(cfg.dropout)
        self.resid_drop2 = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        # Attention sublayer
        attn_out, attn_w = self.attn(self.ln1(x), return_attn=return_attn)
        x = x + self.resid_drop1(attn_out)

        # FFN sublayer
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.resid_drop2(ffn_out)

        return (x, attn_w) if return_attn else x