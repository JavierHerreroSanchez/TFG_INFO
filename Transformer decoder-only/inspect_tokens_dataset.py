from __future__ import annotations

import json
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =========================
# CONFIG
# =========================

# Windows ejemplo: Path(r"C:\Users\...\tokenizer\tokens_json")
TOKENS_DIR = Path(r"../tokenizer/tokens_json_bpe").resolve()


# - "ids"        -> tokens base (lo normal)
# - "ids_encoded"-> si se guardó otra variante (p.ej. BPE)
TOKEN_FIELD = "ids"

# Si se quiere escanear solo una muestra rápida, se pone un número (p.ej. 200).
# Si quieres escanear todas: None
LIMIT_FILES: Optional[int] = None

# Verificación opcional de rango (None si no se quiere chequearlo todavía)
VOCAB_SIZE: Optional[int] = None

# Salida
OUT_DIR = Path(r"../debug_dataset").resolve()
INDEX_CSV = OUT_DIR / "index.csv"
BAD_FILES_TXT = OUT_DIR / "bad_files.txt"


# =========================
# Código
# =========================

@dataclass
class FileStats:
    path: Path
    length: int
    min_id: int
    max_id: int


def iter_json_files(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"No existe TOKENS_DIR: {root}")
    files = sorted([p for p in root.rglob("*.json") if p.is_file()])
    return files


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_token_list(obj: Dict[str, Any], field: str) -> List[int]:
    if field not in obj:
        raise KeyError(f"Falta el campo '{field}'. Campos disponibles: {list(obj.keys())[:20]}")
    tokens = obj[field]
    if not isinstance(tokens, list):
        raise TypeError(f"El campo '{field}' no es una lista, es: {type(tokens)}")
    # Lista vacía: lo consideramos válido pero lo marcamos en stats como length=0
    if len(tokens) == 0:
        return []
    # Comprobación tipo elementos (primero rápido)
    if not isinstance(tokens[0], int):
        raise TypeError(f"El primer elemento de '{field}' no es int: {type(tokens[0])}")
    # Comprobación fuerte (por si hay floats/str sueltos)
    for i, t in enumerate(tokens[:2000]):  # limitamos para no gastar mucho
        if not isinstance(t, int):
            raise TypeError(f"Elemento no-int en '{field}' en pos {i}: {t} ({type(t)})")
    return tokens


def summarize_lengths(lengths: List[int]) -> str:
    arr = np.asarray(lengths, dtype=np.int64)
    if arr.size == 0:
        return "sin datos"
    p = np.percentile(arr, [0, 25, 50, 75, 90, 95, 99, 100]).astype(int)
    return (
        f"min={p[0]} | p25={p[1]} | p50={p[2]} | p75={p[3]} | "
        f"p90={p[4]} | p95={p[5]} | p99={p[6]} | max={p[7]}"
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = iter_json_files(TOKENS_DIR)
    print(f"[INFO] TOKENS_DIR = {TOKENS_DIR}")
    print(f"[INFO] JSON encontrados = {len(files)}")

    if LIMIT_FILES is not None:
        files = files[:LIMIT_FILES]
        print(f"[INFO] LIMIT_FILES activo -> escaneo {len(files)} archivos")

    ok_stats: List[FileStats] = []
    bad: List[Tuple[Path, str]] = []
    empty_count = 0

    global_min = None
    global_max = None

    for idx, path in enumerate(files, start=1):
        try:
            obj = load_json(path)
            tokens = extract_token_list(obj, TOKEN_FIELD)

            if len(tokens) == 0:
                empty_count += 1
                # guardamos stats "vacías" con min/max dummy
                ok_stats.append(FileStats(path=path, length=0, min_id=0, max_id=0))
                continue

            mn = int(min(tokens))
            mx = int(max(tokens))

            if VOCAB_SIZE is not None and mx >= VOCAB_SIZE:
                raise ValueError(f"max_id={mx} >= vocab_size={VOCAB_SIZE}")

            if mn < 0:
                raise ValueError(f"min_id={mn} < 0 (IDs negativos)")

            ok_stats.append(FileStats(path=path, length=len(tokens), min_id=mn, max_id=mx))

            global_min = mn if global_min is None else min(global_min, mn)
            global_max = mx if global_max is None else max(global_max, mx)

            # prints de progreso
            if idx <= 3:
                print(f"[SAMPLE {idx}] {path.name}")
                print(f"         len={len(tokens)} | min={mn} | max={mx}")
                print(f"         first10={tokens[:10]}")
                print(f"         last10 ={tokens[-10:]}")
            if idx % 500 == 0:
                print(f"[SCAN] {idx}/{len(files)} | ok={len(ok_stats)} | bad={len(bad)} | empty={empty_count}")

        except Exception as e:
            bad.append((path, str(e)))

    # Resumen final
    lengths = [s.length for s in ok_stats if s.length > 0]
    print("\n========== RESUMEN ==========")
    print(f"[OK] archivos OK  = {len(ok_stats)}")
    print(f"[BAD] archivos BAD = {len(bad)}")
    print(f"[EMPTY] archivos vacíos = {empty_count}")
    if lengths:
        print(f"[LEN] {summarize_lengths(lengths)}")
        print(f"[ID] global_min={global_min} | global_max={global_max} | vocab_est~{(global_max + 1) if global_max is not None else 'N/A'}")
    else:
        print("[LEN] no hay secuencias no vacías")

    # Guardar index.csv (para usar luego en dataset/entrenamiento)
    print(f"\n[OUT] guardando index: {INDEX_CSV}")
    with INDEX_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "length", "min_id", "max_id"])
        for s in ok_stats:
            w.writerow([str(s.path), s.length, s.min_id, s.max_id])

    # Guardar bad_files.txt
    print(f"[OUT] guardando bad files: {BAD_FILES_TXT}")
    with BAD_FILES_TXT.open("w", encoding="utf-8") as f:
        for p, err in bad:
            f.write(f"{p}\n  -> {err}\n\n")

    print("\n[DONE] Inspección terminada.")


if __name__ == "__main__":
    main()