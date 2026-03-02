import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """
    Máscara causal (T,T) booleana.
    True = PROHIBIDO (mirar al futuro)
    False = permitido
    """
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)


class MaskedMultiHeadSelfAttention(nn.Module):
    """
    Implementación manual de Masked Multi-Head Self-Attention (decoder-only).

    Entrada:
      x: (B, T, D)
      B -> batch size: secuencias procesadas a la vez
      T -> longitud de la secuencia (nº de tokens)
      D -> dimension de la secuencia

    Salida:
      y: (B, T, D)
      (opcional) attn_weights: (B, H, T, T)
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, bias: bool = True, debug: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model debe ser divisible por n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout
        self.debug = debug
        self._printed = False

        # Proyecciones Q, K, V y salida
        self.wq = nn.Linear(d_model, d_model, bias=bias)
        self.wk = nn.Linear(d_model, d_model, bias=bias)
        self.wv = nn.Linear(d_model, d_model, bias=bias)
        self.wo = nn.Linear(d_model, d_model, bias=bias)

        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """
        x: (B, T, D)
        """
        if x.dim() != 3:
            raise ValueError(f"x debe ser (B,T,D). Recibido: {tuple(x.shape)}")

        B, T, D = x.shape
        if D != self.d_model:
            raise ValueError(f"D={D} no coincide con d_model={self.d_model}")

        if self.debug and not self._printed:
            print(f"[MHA] x={tuple(x.shape)} | heads={self.n_heads} d_head={self.d_k} | dropout={self.dropout}")
            self._printed = True

        # 1) Q, K, V
        Q = self.wq(x)  # (B,T,D)
        K = self.wk(x)  # (B,T,D)
        V = self.wv(x)  # (B,T,D)

        # 2) reshape a heads: (B,T,D) -> (B,H,T,d_head)
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # 3) scores = Q K^T / sqrt(d_head)  => (B,H,T,T)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 4) máscara causal (no mirar al futuro)
        causal_mask = build_causal_mask(T, device=x.device)  # (T,T) True=prohibido
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # 5) softmax (normaliza filas -> suma 1)
        attn = F.softmax(scores, dim=-1)  # (B,H,T,T)
        attn = self.attn_drop(attn)

        # 6) aplicar a V: (B,H,T,T) @ (B,H,T,d_head) -> (B,H,T,d_head)
        y = attn @ V

        # 7) juntar heads: (B,H,T,d_head) -> (B,T,D)
        y = y.transpose(1, 2).contiguous().view(B, T, D)

        # 8) proyección final
        y = self.wo(y)

        if self.debug:
            # checks ligeros
            if torch.isnan(y).any():
                print("[MHA][WARN] NaNs en salida")
            # (sin dropout) cada fila de attn debería sumar ~1
            row_sum = attn[0, 0, 0].sum().item()  # una fila ejemplo
            print(f"[MHA] y={tuple(y.shape)} | attn_row_sum(example)={row_sum:.6f}")

        if return_attn:
            return y, attn
        return y




class ResidualConnectionPostLN(nn.Module):
    """
    N(x + Dropout(sublayer_out))
    """
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.dropout(sublayer_out))