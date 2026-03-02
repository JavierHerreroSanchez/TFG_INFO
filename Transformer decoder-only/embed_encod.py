import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Embedding de tokens: ids (B,T) -> (B,T,D)
    Estilo GPT/minGPT: nn.Embedding sin escalado obligatorio.
    """
    def __init__(self, vocab_size: int, n_embd: int):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, T) int64
        return self.wte(idx)  # (B, T, D)


class LearnedPositionalEmbedding(nn.Module):
    """
    Positional embedding aprendible (GPT): posiciones 0..block_size-1.
    """
    def __init__(self, block_size: int, n_embd: int):
        super().__init__()
        self.block_size = block_size
        self.wpe = nn.Embedding(block_size, n_embd)

    def forward(self, T: int, device: torch.device) -> torch.Tensor:
        if T > self.block_size:
            raise ValueError(f"T={T} > block_size={self.block_size}. "
                             f"Recorta ventana o aumenta block_size.")
        pos = torch.arange(0, T, dtype=torch.long, device=device)  # (T,)
        return self.wpe(pos)  # (T, D)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Alternativa: sinusoidal (no aprendible).
    GPT típico usa learned
    """
    def __init__(self, n_embd: int, block_size: int):
        super().__init__()
        pe = torch.zeros(block_size, n_embd)  # (T, D)
        position = torch.arange(0, block_size, dtype=torch.float32).unsqueeze(1)  # (T,1)
        div_term = torch.exp(torch.arange(0, n_embd, 2, dtype=torch.float32) * (-math.log(10000.0) / n_embd))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, T: int, device: torch.device) -> torch.Tensor:
        return self.pe[:T, :].to(device)  # (T, D)


class EmbeddingWithPosition(nn.Module):
    """
    Une token embedding + positional encoding, devolviendo (B,T,D).
    Incluye dropout y prints de debug opcionales.
    """
    def __init__(self, vocab_size: int, n_embd: int, block_size: int, dropout: float = 0.1,
                 use_sinusoidal: bool = False, debug: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size
        self.debug = debug
        self._printed = False

        self.tok_emb = TokenEmbedding(vocab_size, n_embd)
        if use_sinusoidal:
            self.pos_emb = SinusoidalPositionalEncoding(n_embd, block_size)
            self.pos_type = "sinusoidal"
        else:
            self.pos_emb = LearnedPositionalEmbedding(block_size, n_embd)
            self.pos_type = "learned"

        self.drop = nn.Dropout(dropout)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.dtype != torch.long:
            raise TypeError(f"idx debe ser torch.long (int64). dtype={idx.dtype}")

        B, T = idx.shape
        if self.debug and (not self._printed):
            print(f"[Embed+Pos] idx: shape={tuple(idx.shape)} dtype={idx.dtype} "
                  f"| vocab={self.vocab_size} n_embd={self.n_embd} block={self.block_size} pos={self.pos_type}")
            self._printed = True

        tok = self.tok_emb(idx)  # (B,T,D)
        pos = self.pos_emb(T, idx.device)  # (T,D)

        x = tok + pos.unsqueeze(0)  # (B,T,D)
        x = self.drop(x)

        if self.debug and (B > 0):
            # prints “ligeros” para ver que no hay NaNs y rangos razonables
            print(f"[Embed+Pos] out: shape={tuple(x.shape)} "
                  f"| mean={x.mean().item():.4f} std={x.std().item():.4f} "
                  f"| min={x.min().item():.4f} max={x.max().item():.4f}")
        return x