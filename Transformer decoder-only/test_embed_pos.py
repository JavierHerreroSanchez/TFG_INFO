import json
from pathlib import Path
import torch

from embed_encod import EmbeddingWithPosition

# ====== PARAMETROS DE PRUEBA ======
JSON_PATH = Path(r"../tokenizer/tokens_json_bpe/clean_midi/.38 Special/Caught Up In You.json")
TOKEN_FIELD = "ids"  # o "ids_encoded" según el JSON
VOCAB_SIZE = 30000
BLOCK_SIZE = 1024
N_EMBD = 512

# Dónde guardar resultados
OUT_DIR = Path("../debug_outputs").resolve()
OUT_TXT = OUT_DIR / "embed_pos_output.txt"
OUT_PT  = OUT_DIR / "embed_pos_output.pt"  # opcional (recomendado)
# ========================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    obj = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    ids = obj[TOKEN_FIELD]

    print(f"[INFO] file={JSON_PATH.name} | len={len(ids)} | first10={ids[:10]}")

    # Creamos un batch B=2 recortando ventanas (para test)
    T = min(BLOCK_SIZE, len(ids) - 1)
    if T <= 0:
        raise ValueError("La secuencia es demasiado corta para crear una ventana.")
    x1 = torch.tensor(ids[:T], dtype=torch.long).unsqueeze(0)   # (1,T)
    x2 = torch.tensor(ids[-T:], dtype=torch.long).unsqueeze(0)  # (1,T)
    idx = torch.cat([x1, x2], dim=0)                            # (2,T)

    emb = EmbeddingWithPosition(
        vocab_size=VOCAB_SIZE,
        n_embd=N_EMBD,
        block_size=BLOCK_SIZE,
        dropout=0.1,
        use_sinusoidal=False,  # GPT típico
        debug=True
    )

    out = emb(idx)  # (B,T,D)
    out_cpu = out.detach().cpu()

    print(f"[DONE] out shape={tuple(out_cpu.shape)}")

    # 1) Guardar en texto (LEGIBLE)
    # Evitar truncado: profile="full"
    torch.set_printoptions(profile="full", linewidth=200, precision=6, sci_mode=False)

    print(f"[WRITE] Volcando tensor completo a TXT: {OUT_TXT}")
    with OUT_TXT.open("w", encoding="utf-8") as f:
        f.write(f"file={JSON_PATH}\n")
        f.write(f"token_field={TOKEN_FIELD}\n")
        f.write(f"shape={tuple(out_cpu.shape)} dtype={out_cpu.dtype}\n\n")
        f.write(str(out_cpu))
        f.write("\n")

    # Restaurar printoptions por defecto (para no afectar otros prints)
    torch.set_printoptions(profile="default")

    # 2) Guardar en binario (para recargar rápido y explorar sin TXT gigante)
    print(f"[WRITE] Guardando tensor en PT: {OUT_PT}")
    torch.save(out_cpu, OUT_PT)

    print(f"[OK] TXT: {OUT_TXT.resolve()}")
    print(f"[OK] PT : {OUT_PT.resolve()}")

if __name__ == "__main__":
    main()