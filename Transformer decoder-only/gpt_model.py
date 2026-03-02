from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from embed_encod import EmbeddingWithPosition
from decoder_block_postln import DecoderBlockPostLN


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int = 6
    d_model: int = 256
    n_heads: int = 8
    dropout: float = 0.1
    d_ff: Optional[int] = None          # None -> 4*d_model (en FeedForward)
    use_sinusoidal_pos: bool = False
    tie_weights: bool = True
    use_final_ln: bool = True
    debug: bool = True                  # prints once


class GPT(nn.Module):
    """
    GPT (decoder-only) para language modeling autoregresivo:
      idx (B,T) -> logits (B,T,V)

    Estructura:
      idx -> Embedding+Pos -> N * DecoderBlockPostLN -> (optional LN) -> Linear vocab
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self._printed = False

        # 1) Embedding + Positional encoding
        self.embed = EmbeddingWithPosition(
            vocab_size=cfg.vocab_size,
            n_embd=cfg.d_model,
            block_size=cfg.block_size,
            dropout=cfg.dropout,
            use_sinusoidal=cfg.use_sinusoidal_pos,
            debug=cfg.debug,
        )

        # 2) Stack de bloques decoder
        self.blocks = nn.ModuleList([
            DecoderBlockPostLN(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                dropout=cfg.dropout,
                d_ff=cfg.d_ff,
                debug=cfg.debug,
            )
            for _ in range(cfg.n_layer)
        ])

        # 3) LN final (opcional; con Post-LN no es imprescindible, pero suele venir bien)
        self.ln_f = nn.LayerNorm(cfg.d_model) if cfg.use_final_ln else nn.Identity()

        # 4) LM head: proyecta a vocab
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying (como GPT): lm_head.weight == token_embedding.weight
        if cfg.tie_weights:
            # embeddings.py: TokenEmbedding usa self.wte (nn.Embedding)
            self.lm_head.weight = self.embed.tok_emb.wte.weight

        # init suave (opcional, pero ayuda a empezar estable)
        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[GPT] init | layers={cfg.n_layer} d_model={cfg.d_model} heads={cfg.n_heads} "
              f"block={cfg.block_size} vocab={cfg.vocab_size} params={n_params:,}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        idx: (B,T) int64
        targets: (B,T) int64 (next token)
        returns:
          logits: (B,T,V)
          loss:  scalar o None
        """
        if idx.dtype != torch.long:
            raise TypeError(f"idx debe ser torch.long, recibido {idx.dtype}")

        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"T={T} > block_size={self.cfg.block_size}")

        if self.cfg.debug and not self._printed:
            print(f"[GPT.forward] idx={tuple(idx.shape)} targets={'yes' if targets is not None else 'no'}")
            self._printed = True

        x = self.embed(idx)  # (B,T,D)
        for blk in self.blocks:
            x = blk(x)        # (B,T,D)

        x = self.ln_f(x)      # (B,T,D)
        logits = self.lm_head(x)  # (B,T,V)

        loss = None
        if targets is not None:
            # IMPORTANT: no softmax aquí; cross_entropy espera logits “sin normalizar”
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """
        Generación autoregresiva (sanity check).
        idx: (B,T0)
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]  # recorta contexto
            logits, _ = self(idx_cond)

            # solo último paso
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx