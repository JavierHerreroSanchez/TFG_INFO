import json
from pathlib import Path
import torch

from mt_attention_preln import MTEmbedding, RelativeMaskedMHA_PreLN, PreLNResidual

# =======================
# CONFIG
# =======================
JSON_PATH = Path(r"C:\Users\hersa\PycharmProjects\TFG_INFO\tokenizer\tokens_json_bpe\clean_midi\.38 Special\Caught Up In You.json")
TOKEN_FIELD = "ids"          # o "ids_encoded" si procede
VOCAB_SIZE = 30000

T = 128                      # ventana para el test
D_MODEL = 256
N_HEADS = 8
MAX_SEQ_LEN = 512            # máximo permitido por el módulo de atención

DROPOUT = 0.0                # IMPORTANTÍSIMO para test (sin ruido)
DEVICE = "cpu"               # o "cuda"
SEED = 123
# =======================


def load_ids(path: Path, field: str) -> list[int]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    ids = obj[field]
    if not isinstance(ids, list) or (len(ids) > 0 and not isinstance(ids[0], int)):
        raise TypeError(f"'{field}' no es lista de ints en {path}")
    return ids


def main():
    torch.manual_seed(SEED)

    ids = load_ids(JSON_PATH, TOKEN_FIELD)
    print(f"[INFO] file={JSON_PATH.name} | len={len(ids)} | first10={ids[:10]}")

    if len(ids) < T:
        raise ValueError(f"Secuencia demasiado corta: len={len(ids)} < T={T}")

    # batch (1,T)
    idx = torch.tensor(ids[:T], dtype=torch.long).unsqueeze(0).to(DEVICE)

    # check rango (muy útil)
    mx = int(idx.max().item())
    mn = int(idx.min().item())
    print(f"[INFO] token range in window: min={mn} max={mx} | vocab_size={VOCAB_SIZE}")
    if mn < 0 or mx >= VOCAB_SIZE:
        raise ValueError(f"Hay tokens fuera de rango [0, vocab_size): min={mn} max={mx}")

    emb = MTEmbedding(vocab_size=VOCAB_SIZE, d_model=D_MODEL, dropout=DROPOUT, debug=True).to(DEVICE)
    attn = RelativeMaskedMHA_PreLN(d_model=D_MODEL, n_heads=N_HEADS, max_seq_len=MAX_SEQ_LEN,
                                  dropout=DROPOUT, bias=True, debug=True).to(DEVICE)
    preln = PreLNResidual(d_model=D_MODEL, dropout=DROPOUT).to(DEVICE)

    emb.eval(); attn.eval(); preln.eval()

    # -----------------------
    # Forward: x -> PreLN(attn)
    # -----------------------
    with torch.no_grad():
        x = emb(idx)  # (1,T,D)
        y, w = preln(x, attn, return_attn=True)  # y: (1,T,D), w: (1,H,T,T)
    # métricas no triviales de atención (head0, token t=10)
    t_row = min(10, T - 1)
    row = w[0, 0, t_row, :t_row + 1]  # solo pasado permitido
    entropy = -(row * (row.clamp_min(1e-12).log())).sum().item()

    topv, topi = torch.topk(row, k=min(5, row.numel()))
    print(f"[INFO] attn row t={t_row} sum={row.sum().item():.6f} entropy={entropy:.6f}")
    print(f"[INFO] top attn weights: {list(zip(topi.tolist(), topv.tolist()))}")

    # stats de tensores
    print(f"[INFO] x mean/std: {x.mean().item():.6f} / {x.std().item():.6f}")
    print(f"[INFO] y mean/std: {y.mean().item():.6f} / {y.std().item():.6f}")
    print(f"[OUT] x={tuple(x.shape)} y={tuple(y.shape)} attn={tuple(w.shape)}")

    # -----------------------
    # CHECK 1: máscara causal -> futuro debe ser 0
    # -----------------------
    future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=DEVICE), diagonal=1)  # j>i
    max_future = w[0, 0][future].max().item()  # batch0 head0
    print(f"[CHECK] max attn weight (future, head0) = {max_future:.12f}")
    assert max_future < 1e-8, "Falla máscara causal: hay atención al futuro."

    # -----------------------
    # CHECK 2: invariancia del prefijo
    # -----------------------
    split_t = T // 2
    idx2 = idx.clone()
    idx2[:, split_t:] = torch.randint(0, VOCAB_SIZE, idx2[:, split_t:].shape, device=DEVICE, dtype=torch.long)

    with torch.no_grad():
        y1, _ = preln(emb(idx),  attn, return_attn=False)
        y2, _ = preln(emb(idx2), attn, return_attn=False)

    diff = (y1[:, :split_t, :] - y2[:, :split_t, :]).abs().max().item()
    print(f"[CHECK] prefix invariance max|diff| (<= {split_t}) = {diff:.10f}")
    assert diff < 1e-6, "Falla causalidad: el prefijo cambia al modificar el futuro."

    print("\n[DONE] Embedding + Relative Masked Self-Attention (Pre-LN) OK ✅")


if __name__ == "__main__":
    main()