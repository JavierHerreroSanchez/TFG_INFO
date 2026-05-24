"""
Define componentes del modelo Transformer usado para generacion musical simbolica.

Estas clases encapsulan la arquitectura neuronal que despues se reutiliza en preentrenamiento, fine-tuning y generacion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.blocks import MTConfig, MTEmbedding, MusicTransformerBlockPreLN

# =============================================================================
# Configuración de alto nivel del modelo (GPT decoder-only)
# -----------------------------------------------------------------------------
# En esta dataclass se definen los hiperparámetros “macro” del modelo:
# - tamaño del vocabulario y longitud máxima (block_size)
# - profundidad (n_layer) y dimensiones internas (d_model, n_heads, d_ff)
# - decisiones de arquitectura (atar pesos, LayerNorm final)
# Esta capa de configuración permite instanciar el modelo de forma reproducible.
# =============================================================================
@dataclass
class MTModelConfig:
    """Encapsula la arquitectura MTModelConfig usada por los experimentos."""

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

# =============================================================================
# MusicTransformerGPT (decoder-only autoregresivo)
# -----------------------------------------------------------------------------
# Se implementa un Transformer únicamente con la parte del decoder:
#   idx (B, T) -> embeddings (B, T, D) -> bloques (B, T, D) -> logits (B, T, V)
# El entrenamiento de pretraining se plantea como modelado autorregresivo:
# predecimos el siguiente token (next-token prediction) usando máscara causal
# (implementada dentro de la atención relativa del bloque).
# =============================================================================
class MusicTransformerGPTlike(nn.Module):
    """
    Modelo decoder-only autorregresivo para música,
    inspirado en la idea de combinar:
      - una pila de bloques tipo Music Transformer (atención relativa + máscara causal),
      - con el objetivo clásico de GPT: predecir el siguiente token en la secuencia.

    La interfaz principal es:
      idx (B, T) -> logits (B, T, V)
    y opcionalmente devuelve la loss.
    """
    def __init__(self, cfg: MTModelConfig):
        """
        Implementa la logica de   init   dentro del pipeline del TFG.

        Parametros principales: cfg.
        """

        super().__init__()
        self.cfg = cfg

        # Conversión de la configuración macro (MTModelConfig) en la configuración
        # de bloque (MTConfig) usada por los componentes importados desde `blocks`.
        # Aquí fijamos max_seq_len = block_size para el rango de embeddings relativos.
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

        # 1) Embedding de tokens (sin positional encoding absoluto).
        self.embed = MTEmbedding(cfg.vocab_size, cfg.d_model, cfg.dropout, debug=cfg.debug)

        # 2) Pila de bloques decoder (Pre-LN) con atención relativa y FFN.
        self.blocks = nn.ModuleList([MusicTransformerBlockPreLN(bcfg) for _ in range(cfg.n_layer)])

        # 3) Normalización final (opcional). En muchos Transformers, un LN final
        #    mejora estabilidad y calidad; queda configurable.
        self.ln_f = nn.LayerNorm(cfg.d_model) if cfg.use_final_ln else nn.Identity()

        # 4) Proyección a vocabulario (capa “LM head”): (B, T, D) -> (B, T, V).
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # En caso de usar weight tying, se comparten los pesos del embedding de entrada
        # con la proyección de salida para reducir el número de parámetros.
        if cfg.tie_weights:
            self.lm_head.weight = self.embed.wte.weight

        # Inicialización explícita de pesos (estilo GPT: Normal(0, 0.02)).
        self.apply(self._init_weights)

        # Resumen informativo: número de parámetros y configuración principal.
        n_params = sum(p.numel() for p in self.parameters())
        print(
            f"[MT] init | layers={cfg.n_layer} d_model={cfg.d_model} heads={cfg.n_heads} "
            f"block={cfg.block_size} vocab={cfg.vocab_size} params={n_params:,}"
        )

    # =============================================================================
    # Inicialización de pesos
    # -----------------------------------------------------------------------------
    # Se aplica una inicialización gaussiana a capas lineales y embeddings, y
    # ponemos a cero los sesgos. Este patrón es habitual en implementaciones
    # tipo GPT para estabilizar el arranque del entrenamiento.
    # =============================================================================
    def _init_weights(self, module):
        """
        Implementa la logica de  init weights dentro del pipeline del TFG.

        Parametros principales: module.
        """

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # =============================================================================
    # Forward (entrenamiento / inferencia)
    # -----------------------------------------------------------------------------
    # En el forward:
    #  - verificamos tipos y longitudes
    #  - embebemos tokens
    #  - pasamos por la pila de bloques decoder
    #  - proyectamos a logits de vocabulario
        # Si `targets` está presente, se calcula cross-entropy a nivel de token,
    # tal y como se hace en next-token prediction (lenguaje/música).
    # =============================================================================
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # En PyTorch, los índices de nn.Embedding deben ser enteros (torch.long).
        """
        Implementa la logica de forward dentro del pipeline del TFG.

        Parametros principales: idx, targets.
        """

        if idx.dtype != torch.long:
            raise TypeError(f"idx debe ser torch.long, recibido {idx.dtype}")
        B, T = idx.shape

        # Restringimos la longitud al block_size; el bloque de atención relativa
        # también depende de este máximo para sus embeddings de distancia.
        if T > self.cfg.block_size:
            raise ValueError(f"T={T} > block_size={self.cfg.block_size}")

        # 1) Embedding: (B, T) -> (B, T, D)
        x = self.embed(idx)

        # 2) Decoder stack: la forma (B, T, D) se mantiene a lo largo de los bloques.
        for blk in self.blocks:
            x = blk(x)

        # 3) Normalización final (si se habilita).
        x = self.ln_f(x)

        # 4) Logits sobre el vocabulario: (B, T, D) -> (B, T, V)
        logits = self.lm_head(x)

        # 5) Loss opcional (entrenamiento): cross-entropy por token.
        #    Aplanamos (B, T, V) -> (B*T, V) y (B, T) -> (B*T).
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    # =============================================================================
    # Generación autorregresiva
    # -----------------------------------------------------------------------------
    # En esta función generamos tokens iterativamente:
    #  - recortamos el contexto al último block_size
    #  - obtenemos logits del último timestep
    #  - aplicar temperatura y (opcionalmente) top-k sampling
    #  - muestreamos el siguiente token y lo concatenamos al contexto
    # Este es el patrón estándar de muestreo en modelos GPT.
    # =============================================================================
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """
        Genera tokens de forma autorregresiva sin calcular gradientes.

        En cada paso se recorta el contexto a `block_size`, se calculan los logits
        del último token y se muestrea el siguiente id aplicando temperatura y,
        opcionalmente, top-k sampling.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Conserva el contexto más reciente para respetar block_size.
            idx_cond = idx[:, -self.cfg.block_size:]

            # Inferencia: logits para todo el contexto; tomamos el último paso.
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Top-k sampling: restringimos la masa de probabilidad a los k tokens
            # más probables antes del softmax.
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            # Conversión a distribución y muestreo del siguiente id.
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            # Actualizamos la secuencia concatenando el token generado.
            idx = torch.cat([idx, next_id], dim=1)

        return idx
