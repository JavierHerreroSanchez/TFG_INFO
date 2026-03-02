from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import MTConfig, MTEmbedding, MusicTransformerBlockPreLN


@dataclass
class MTModelConfig:
    vocab_size: int
    block_size: int
    n_layer: int = 6
    d_model: int = 256
    n_heads: int = 8
    dropout: float = 0.1
    d_ff: Optional[int] = None
    bias: bool = True
    tie_weights: bool = True
    use_final_ln: bool = True
    debug: bool = False


class MusicTransformerGPT(nn.Module):
    """
    Decoder-only autoregresivo (pretraining):
      idx (B,T) -> logits (B,T,V)
    """
    def __init__(self, cfg: MTModelConfig):
        super().__init__()
        self.cfg = cfg

        bcfg = MTConfig(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            max_seq_len=cfg.block_size,
            dropout=cfg.dropout,
            bias=cfg.bias,
            d_ff=cfg.d_ff,
            debug=cfg.debug,
        )

        self.embed = MTEmbedding(cfg.vocab_size, cfg.d_model, cfg.dropout, debug=cfg.debug)
        self.blocks = nn.ModuleList([MusicTransformerBlockPreLN(bcfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model) if cfg.use_final_ln else nn.Identity()

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.lm_head.weight = self.embed.wte.weight

        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[MT] init | layers={cfg.n_layer} d_model={cfg.d_model} heads={cfg.n_heads} "
              f"block={cfg.block_size} vocab={cfg.vocab_size} params={n_params:,}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if idx.dtype != torch.long:
            raise TypeError(f"idx debe ser torch.long, recibido {idx.dtype}")
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"T={T} > block_size={self.cfg.block_size}")

        x = self.embed(idx)        # (B,T,D)
        for blk in self.blocks:
            x = blk(x)             # (B,T,D)

        x = self.ln_f(x)
        logits = self.lm_head(x)   # (B,T,V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx