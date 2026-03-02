import json
from pathlib import Path

import torch
import torch.nn as nn

from embed_encod import EmbeddingWithPosition
from masked_mha import MaskedMultiHeadSelfAttention, build_causal_mask


# =========================
# CONFIG
# =========================
JSON_PATH = Path(r"../tokenizer/tokens_json_bpe/clean_midi/.38 Special/Caught Up In You.json")
TOKEN_FIELD = "ids"                                  # "ids" o "ids_encoded"
VOCAB_SIZE = 30000                                   # <- tu vocab real

BLOCK_SIZE = 2048     # para test rápido; luego 1024/2048
D_MODEL = 512        # dimensión embedding/modelo
N_HEADS = 8
DROPOUT = 0.1        # en eval() no afecta

SPLIT_T = 128        # para test causalidad (prefijo)
SEED = 123
DEVICE = "cpu"       # o "cuda" si tienes GPU y quieres probar
# =========================


def load_tokens(path: Path, field: str) -> list[int]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    tokens = obj[field]
    if not isinstance(tokens, list) or (len(tokens) > 0 and not isinstance(tokens[0], int)):
        raise TypeError(f"Campo '{field}' no es lista de ints.")
    return tokens


def build_batch_from_tokens(tokens: list[int], T: int, batch_size: int = 2) -> torch.Tensor:
    """
    Devuelve idx (B,T) con 2 crops: inicio y final.
    """
    if len(tokens) < T:
        raise ValueError(f"Secuencia demasiado corta: len={len(tokens)} < T={T}")

    x1 = torch.tensor(tokens[:T], dtype=torch.long).unsqueeze(0)
    x2 = torch.tensor(tokens[-T:], dtype=torch.long).unsqueeze(0)
    idx = torch.cat([x1, x2], dim=0)  # (2,T)
    return idx


def max_forbidden_attention(attn_weights: torch.Tensor) -> float:
    """
    attn_weights: (B,H,T,T)
    Devuelve el máximo weight en posiciones prohibidas (futuro).
    """
    B, H, T, _ = attn_weights.shape
    causal = build_causal_mask(T, device=attn_weights.device)  # True=prohibido
    forbidden = attn_weights[0, 0][causal]  # head 0, batch 0
    return float(forbidden.max().item()) if forbidden.numel() > 0 else 0.0


def prefix_invariance_test(model_fn, idx: torch.Tensor, split_t: int, vocab_size: int):
    """
    idx1 e idx2 iguales en prefijo; idx2 distinto en futuro.
    Si causalidad está bien, la salida en el prefijo debe coincidir.
    """
    idx1 = idx.clone()
    idx2 = idx.clone()

    if split_t < idx.size(1):
        torch.manual_seed(SEED)
        idx2[:, split_t:] = torch.randint(
            low=0, high=vocab_size, size=idx2[:, split_t:].shape, dtype=torch.long, device=idx2.device
        )

    with torch.no_grad():
        y1 = model_fn(idx1)
        y2 = model_fn(idx2)

    diff = (y1[:, :split_t, :] - y2[:, :split_t, :]).abs().max().item()
    print(f"[CAUSALITY] max|diff| en prefijo (<= {split_t}) = {diff:.10f}")
    # tolerancia pequeña por numérico
    assert diff < 1e-5, "Falla causalidad: el prefijo cambia al alterar el futuro."


def main():
    torch.manual_seed(SEED)

    tokens = load_tokens(JSON_PATH, TOKEN_FIELD)
    print(f"[INFO] file={JSON_PATH.name} | len={len(tokens)} | first10={tokens[:10]}")

    # Ventana T
    T = min(BLOCK_SIZE, len(tokens))
    if T < 4:
        raise ValueError("Secuencia demasiado corta para el test.")

    idx = build_batch_from_tokens(tokens, T=T, batch_size=2).to(DEVICE)
    print(f"[INFO] idx shape={tuple(idx.shape)} | dtype={idx.dtype} | device={idx.device}")

    # 1) Embedding + Positional Encoding
    embed = EmbeddingWithPosition(
        vocab_size=VOCAB_SIZE,
        n_embd=D_MODEL,
        block_size=BLOCK_SIZE,
        dropout=DROPOUT,
        use_sinusoidal=False,
        debug=True
    ).to(DEVICE)

    # 2) Masked Multi-Head Self-Attention (manual)
    attn = MaskedMultiHeadSelfAttention(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        dropout=DROPOUT,
        bias=True,
        debug=True
    ).to(DEVICE)

    # 3) Add & Norm (Post-LN): LN(x + attn(x))
    ln_post = nn.LayerNorm(D_MODEL).to(DEVICE)

    # Poner en eval para que dropout no meta ruido durante validación
    embed.eval()
    attn.eval()
    ln_post.eval()

    # ----- Forward pass parcial -----
    with torch.no_grad():
        x = embed(idx)                 # (B,T,D)
        attn_out, w = attn(x, return_attn=True)  # attn_out (B,T,D), w (B,H,T,T)
        y = ln_post(x + attn_out)      # (B,T,D)

    print(f"[OUT] x (embed+pos) shape={tuple(x.shape)}")
    print(f"[OUT] attn_out shape={tuple(attn_out.shape)}")
    print(f"[OUT] y (Post-LN) shape={tuple(y.shape)}")

    # ----- Checks -----
    m = max_forbidden_attention(w)
    print(f"[CHECK] max attn weight en futuro (debería ~0): {m:.12f}")
    assert m < 1e-8, "La máscara causal no está anulando bien el futuro."

    # Test de invariancia del prefijo (muy útil)
    def pipeline(idx_in: torch.Tensor) -> torch.Tensor:
        x_ = embed(idx_in)
        a_ = attn(x_)
        return ln_post(x_ + a_)  # Post-LN

    prefix_invariance_test(pipeline, idx[:1, :], split_t=min(SPLIT_T, T // 2), vocab_size=VOCAB_SIZE)

    print("\n[DONE] Forward parcial (Embed+Pos -> Masked MHA -> Post-LN) OK ✅")


if __name__ == "__main__":
    main()