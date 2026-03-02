# mt_block_preln.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from mt_attention_preln import RelativeMaskedMHA_PreLN


class FeedForward(nn.Module):
    """
    FFN del Transformer: Linear -> ReLU/GELU -> Linear (con dropout).
    Music Transformer suele usar arquitectura estándar de Transformer.
    Usamos GELU (también vale ReLU).
    """
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1, bias: bool = True, debug: bool = False):
        super().__init__()
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.fc1 = nn.Linear(d_model, self.d_ff, bias=bias)
        self.fc2 = nn.Linear(self.d_ff, d_model, bias=bias)
        self.drop = nn.Dropout(dropout)
        self.debug = debug
        self._printed = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.debug and not self._printed:
            print(f"[FFN] x={tuple(x.shape)} d_ff={self.d_ff}")
            self._printed = True
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class PreLNResidual(nn.Module):
    """
    Pre-LN residual wrapper:
      x = x + Dropout( sublayer(LN(x)) )
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: nn.Module, **kwargs):
        x_norm = self.ln(x)
        out = sublayer(x_norm, **kwargs)
        # sublayer puede devolver (y, attn) o solo y
        if isinstance(out, tuple):
            y, extra = out
            return x + self.drop(y), extra
        else:
            return x + self.drop(out), None


class MusicTransformerBlockPreLN(nn.Module):
    """
    Bloque completo:
      1) Relative masked MHA + Pre-LN residual
      2) FFN + Pre-LN residual
    """
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1, d_ff: int = None,
                 bias: bool = True, debug: bool = False):
        super().__init__()
        self.attn = RelativeMaskedMHA_PreLN(d_model, n_heads, max_seq_len, dropout=dropout, bias=bias, debug=debug)
        self.ffn = FeedForward(d_model, d_ff=d_ff, dropout=dropout, bias=bias, debug=debug)

        self.resid1 = PreLNResidual(d_model, dropout)
        self.resid2 = PreLNResidual(d_model, dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        # attention sublayer
        x, attn = self.resid1(x, self.attn, return_attn=return_attn)
        # ffn sublayer
        x, _ = self.resid2(x, self.ffn)
        return (x, attn) if return_attn else x