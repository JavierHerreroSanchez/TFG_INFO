from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]

TOKENS_DIR = PROJECT_ROOT / "data" / "interim" / "tokenized_finetuning_v3"
OUT_CSV = PROJECT_ROOT / "data" / "interim" / "indexes" / "index_finetuning_v3.csv"

TOKEN_FIELD_CANDIDATES = ("ids", "ids_encoded")


def extract_ids(obj: dict) -> list[int] | None:
    for field in TOKEN_FIELD_CANDIDATES:
        ids = obj.get(field, None)
        if isinstance(ids, list):
            return ids
    return None

def main() -> None:
    if not TOKENS_DIR.exists():
        raise FileNotFoundError(f"No existe TOKENS_DIR: {TOKENS_DIR}")

    json_paths = sorted(p for p in TOKENS_DIR.rglob("*.json") if p.is_file())
    print(f"[INFO] JSON encontrados: {len(json_paths)}")
    print(f"[INFO] TOKENS_DIR: {TOKENS_DIR}")
    print(f"[INFO] OUT_CSV: {OUT_CSV}")

    rows = []
    bad = 0

    for i, json_path in enumerate(json_paths, start=1):
        try:
            obj = json.loads(json_path.read_text(encoding="utf-8"))
            ids = extract_ids(obj)

            if not ids:
                bad += 1
                print(f"[WARN] sin ids o vacío: {json_path}")
                continue

            rows.append({
                "path": str(json_path.resolve()),
                "length": int(len(ids)),
                "min_id": int(min(ids)),
                "max_id": int(max(ids)),
            })

        except Exception as e:
            bad += 1
            print(f"[WARN] no se pudo leer {json_path}: {type(e).__name__}: {e}")

        if i % 5000 == 0:
            print(f"[INFO] procesados {i}/{len(json_paths)}")

    df = pd.DataFrame(rows, columns=["path", "length", "min_id", "max_id"])
    df = df.sort_values("path").reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print("\n=== RESUMEN ===")
    print(f"Filas válidas: {len(df)}")
    print(f"JSON problemáticos: {bad}")
    print(f"CSV guardado en: {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()