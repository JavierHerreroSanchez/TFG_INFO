"""
Define componentes del modelo Transformer usado para generación musical simbólica.

Estas clases encapsulan la arquitectura neuronal que después se reutiliza en
preentrenamiento, fine-tuning y generación.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Configuración de hiperparámetros (bloques tipo Music Transformer)
# -----------------------------------------------------------------------------
# En esta dataclass centralizamos los parámetros del modelo (dimensiones,
# número de cabezas, longitud máxima, etc.) para que todos los submódulos
# compartan una misma fuente.
# =============================================================================
@dataclass
class MTConfig:
    """Agrupa la configuración de MTConfig para hacer reproducibles los experimentos."""

    vocab_size: int
    d_model: int
    n_heads: int
    max_seq_len: int            # Longitud máxima de secuencia (equivalente a block_size)
    dropout: float = 0.1        # Establecemos a 0.1, siendo el dropout típico
    bias: bool = True
    d_ff: Optional[int] = None  # Si es None, se aplica la heurística típica d_ff = 4*d_model
    debug: bool = False


# =============================================================================
# Embedding de tokens (sin positional encoding absoluto)
# -----------------------------------------------------------------------------
# En el Music Transformer (Huang et al., 2018), la información de posición puede
# incorporarse mediante atención relativa; por ello, aquí únicamente embebemos
# tokens y aplicar dropout, sin sumar codificación posicional absoluta.
# =============================================================================
class MTEmbedding(nn.Module):
    """
    Convierte ids de tokens en vectores de dimensión d_model.
    Sigue el planteamiento del Music Transformer: embedding + dropout,
    evitando un positional encoding absoluto en esta etapa para no solapar el
    mecanismo de atención relativa.
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


# =============================================================================
# Skewing (reindexado absoluto-relativo -> absoluto-absoluto)
# -----------------------------------------------------------------------------
# Este procedimiento corresponde al “skewing” descrito en Music Transformer
# (Huang et al., 2018, Sección 3.4.1). Su objetivo es transformar una matriz
# indexada por (posición_absoluta, distancia_relativa) en otra indexada por
# (posición_absoluta, posición_absoluta), alineando cada logit relativo con el
# par (i, j) correcto sin materializar un tensor O(T^2 * d_head).
# =============================================================================
def skew(QEr: torch.Tensor) -> torch.Tensor:
    """
    Aplica el “skewing” del Music Transformer para alinear logits relativos.

    Entradas:
      QEr: (B, H, T, 2T-1)
        - Para cada query i, la última dimensión recorre distancias relativas r
          codificadas como r = (j - i) + (T - 1).

    Salida:
      S_rel: (B, H, T, T)
        - Matriz alineada por pares absolutos (i, j), lista para sumarse a QK^T.

    Implementación (estándar en el paper):
      pad -> reshape -> slice -> reshape -> crop
    """
    B, H, T, M = QEr.shape
    assert M == 2 * T - 1, f"Esperaba 2T-1, recibido {M} con T={T}"

    # 1) Columna dummy a la izquierda para habilitar el corrimiento.
    x = F.pad(QEr, (1, 0)).contiguous()  # (B, H, T, 2T)

    # 2) Reinterpretamos para desplazar los elementos por filas/columnas.
    x = x.view(B, H, 2 * T, T)  # (B, H, 2T, T)

    # 3) Eliminamos la primera fila “extra” para completar el corrimiento.
    x = x[:, :, 1:, :].contiguous()  # (B, H, 2T-1, T)

    # 4) Volvemos a la forma original en la que la última dim es “2T-1”.
    x = x.view(B, H, T, 2 * T - 1)  # (B, H, T, 2T-1)

    # 5) Conserva las T columnas que corresponden a j en [0, T-1].
    x = x[:, :, :, :T]  # (B, H, T, T)

    return x


# =============================================================================
# Relative Masked Multi-Head Self-Attention (global, causal)
# -----------------------------------------------------------------------------
# Autoatención con posiciones relativas (Music Transformer).
# La puntuación de atención combina:
#   - contenido: QK^T
#   - término relativo: S_rel = skew(Q * Er^T)
# y máscara causal para evitar atención al futuro (modelo autorregresivo).
# =============================================================================
class RelativeMaskedMHA(nn.Module):
    """
    En este módulo incorporamos atención relativa para que el modelo aprenda
    dependencias en función de la distancia entre posiciones, algo crucial en música
    (ritmo, repetición, motivo).

    Seguimos el esquema del Music Transformer (Huang et al., 2018):
      scores = (QK^T + S_rel) / sqrt(d_head),
    donde:
      S_rel = skew(Q * Er^T),
    y Er contiene embeddings para distancias relativas en el rango [-T+1, T-1]
    (almacenadas como 2T-1 embeddings).
    Además, aplica máscara causal para garantizar generación autorregresiva.
    """
    def __init__(self, cfg: MTConfig):

        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads

        # Proyección conjunta para Q, K, V
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)

        self.attn_drop = nn.Dropout(cfg.dropout)
        self.out_drop = nn.Dropout(cfg.dropout)

        # Embeddings relativos para distancias en [-L+1, L-1], con L = max_seq_len o T.
        self.rel_emb = nn.Embedding(2 * cfg.max_seq_len - 1, self.n_heads * self.d_head)

        # Máscara causal: True indica posiciones prohibidas (futuro).
        causal = torch.triu(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal, persistent=False)

        self._printed = False

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x: (B, T, D)

        Devuelve:
          y: (B, T, D)
          attn: (B, H, T, T) si return_attn=True (útil para debugear/inspección/visualización)
        """
        B, T, D = x.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"T={T} excede max_seq_len={self.cfg.max_seq_len}")

        if self.cfg.debug and not self._printed:
            print(
                f"[RelMHA] x={tuple(x.shape)} heads={self.n_heads} d_head={self.d_head} max_seq_len={self.cfg.max_seq_len}")
            self._printed = True

        # 1) Cálculo de Q, K, V y reordenación a (B, H, T, Dh).
        qkv = self.qkv(x)  # (B, T, 3D)
        q, k, v = qkv.split(D, dim=-1)  # (B, T, D) cada uno

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, Dh)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, Dh)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, Dh)

        # 2) Término de contenido (estándar): QK^T.
        content = q @ k.transpose(-2, -1)  # (B, H, T, T)

        # 3) Término relativo (Music Transformer):
        center = self.cfg.max_seq_len - 1
        start = center - (T - 1)
        end = center + (T - 1) + 1

        Er_flat = self.rel_emb.weight[start:end]  # (2T-1, H*Dh)
        Er = Er_flat.view(2 * T - 1, self.n_heads, self.d_head).permute(1, 2, 0)  # (H,2T-1,Dh)

        # Proyección de Q contra Er^T -> (B, H, T, 2T-1) y aplicación de skew.
        QEr = torch.matmul(q, Er)  # (B,H,T,2T-1)
        rel = skew(QEr)  # (B,H,T,T)

        # 4) Escalamos y sumamos ambos términos, como en el Transformer.
        scores = (content + rel) / math.sqrt(self.d_head)

        # 5) Enmascaramos el futuro (causal mask) para generación autorregresiva.
        scores = scores.masked_fill(self.causal_mask[:T, :T][None, None, :, :], float("-inf"))

        # 6) Softmax para probabilidades de atención + dropout.
        attn = torch.softmax(scores, dim=-1)  # (B, H, T, T)
        attn = self.attn_drop(attn)

        # 7) Combinamos valores: Attn * V y proyectamos de vuelta a (B, T, D).
        y = attn @ v  # (B, H, T, Dh)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        y = self.out_drop(self.out_proj(y))

        if self.cfg.debug:
            # Como comprobación no trivial, medimos entropía de una fila de atención.
            trow = min(10, T - 1)
            row = attn[0, 0, trow, :trow + 1]
            ent = -(row * row.clamp_min(1e-12).log()).sum().item()
            print(f"[RelMHA] attn entropy(head0,t={trow})={ent:.4f} row_sum={row.sum().item():.4f}")

        return (y, attn) if return_attn else (y, None)


# =============================================================================
# Feed Forward (FFN)
# -----------------------------------------------------------------------------
# Este sub-bloque replica el FFN del Transformer estándar (Vaswani et al., 2017):
#    Linear(d_model -> d_ff) -> activación (GELU aquí) -> Dropout -> Linear(d_ff -> d_model)
# En el contexto del Music Transformer, mantiene la misma función: expandir y
# remezclar características por posición de forma punto-a-punto.
# =============================================================================
class FeedForward(nn.Module):
    """
    Bloque FFN por posición del Transformer.

    A diferencia de la atención, que mezcla información entre posiciones, el FFN
    actúa de manera independiente en cada timestep: aplica dos proyecciones lineales
    (con una no linealidad intermedia) sobre la dimensión de características.
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


# =============================================================================
# Bloque completo tipo Music Transformer (Pre-LN)
# -----------------------------------------------------------------------------
# Variante Pre-LayerNorm (común en implementaciones modernas):
#   x = x + Dropout( Attn( LN(x) ) )
#   x = x + Dropout(  FFN( LN(x) ) )
# Esta formulación suele estabilizar el entrenamiento en Transformers profundos.
# =============================================================================
class MusicTransformerBlockPreLN(nn.Module):
    """
    En este bloque combinamos:
      1) Autoatención (con posiciones relativas y máscara causal)
      2) Feed-Forward (FFN)
    y envolvemos ambos submódulos con residual connections.

    El esquema Pre-LN (normalizar antes de cada subcapa) suele mejorar
    la estabilidad numérica y el flujo de gradiente en comparación con Post-LN.
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
        # Subcapa 1: atención (primero normalizamos, luego atendemos, luego residual).

        attn_out, attn_w = self.attn(self.ln1(x), return_attn=return_attn)
        x = x + self.resid_drop1(attn_out)

        # Subcapa 2: FFN (mismo patrón: LN -> FFN -> dropout -> residual).
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.resid_drop2(ffn_out)

        return (x, attn_w) if return_attn else x
