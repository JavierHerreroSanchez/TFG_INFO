import torch
import torch.nn as nn

from masked_mha import MaskedMultiHeadSelfAttention, ResidualConnectionPostLN
from feed_forward import FeedForward


class DecoderBlockPostLN(nn.Module):
    """
    Un bloque Transformer decoder-only (sin cross-attention):
      1) Masked MHA
      2) Add & Norm (Post-LN)
      3) FFN
      4) Add & Norm (Post-LN)
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, d_ff: int = None, debug: bool = True):
        super().__init__()
        self.attn = MaskedMultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            bias=True,
            debug=debug
        )
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, debug=debug)

        self.resid1 = ResidualConnectionPostLN(d_model=d_model, dropout=dropout)
        self.resid2 = ResidualConnectionPostLN(d_model=d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) attention + add&norm
        attn_out = self.attn(x)
        x = self.resid1(x, attn_out)

        # 2) ffn + add&norm
        ffn_out = self.ffn(x)
        x = self.resid2(x, ffn_out)

        return x