import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    FFN clásico del Transformer:
      x -> Linear(d_model -> d_ff) -> GELU -> Dropout -> Linear(d_ff -> d_model) -> Dropout

    d_ff suele ser 4*d_model (como en GPT/minGPT).
    """
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1, debug: bool = True):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.dropout = dropout

        self.fc1 = nn.Linear(d_model, self.d_ff)
        self.fc2 = nn.Linear(self.d_ff, d_model)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        self.debug = debug
        self._printed = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        if self.debug and not self._printed:
            print(f"[FFN] x={tuple(x.shape)} | d_ff={self.d_ff} | dropout={self.dropout}")
            self._printed = True

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop1(x)
        x = self.fc2(x)

        return x