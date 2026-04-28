from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

INDEX_CSV = Path(r"/data/interim/indexes/index_pretraining.csv")
TOKENS_DIR = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\data\interim\tokenized_json_bpe").resolve()
ANCHOR = r"data\interim\tokenized_json_bpe"

CACHE_DIR = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\data\processed\pretraining_check").resolve()
OUT_BIN = CACHE_DIR / "train_check.bin"

TOKEN_FIELD = "ids"
ADD_EOS = True
EOS_ID = 2
VOCAB_SIZE = 30000
USE_UINT16 = True

VAL_RATIO = 0.01
TEST_RATIO = 0.01
SEED = 100454434

# Si quieres revisar solo una parte para depurar más rápido:
LIMIT_FILES = None  # por ejemplo 5000, o None para todos


# =============================================================================
# UTILIDADES
# =============================================================================

def choose_np_dtype(use_uint16: bool, vocab_size: int):
    if use_uint16:
        if vocab_size >= 65535:
            raise ValueError("VOCAB_SIZE no cabe en uint16; usa uint32.")
        return np.uint16
    return np.uint32


def rebase_path(abs_path: str, tokens_dir: Path, anchor: str) -> Path:
    s = abs_path.replace("\\", "/")
    a = anchor.replace("\\", "/")
    pos = s.find(a)
    if pos == -1:
        return Path(abs_path)
    rel_part = s[pos + len(a):].lstrip("/")
    return tokens_dir / rel_part


def resolve_json_paths(index_csv: Path, tokens_dir: Path, anchor: str) -> List[Path]:
    if not index_csv.exists():
        raise FileNotFoundError(f"INDEX_CSV no existe: {index_csv}")

    df = pd.read_csv(index_csv)
    if "path" not in df.columns:
        raise ValueError("index_pretraining.csv debe tener columna 'path'.")

    raw_paths = df["path"].tolist()

    # 1) Uso directo
    paths1 = [Path(p) for p in raw_paths]
    exist1 = [p for p in paths1 if p.exists()]
    if len(exist1) > 0:
        print(f"[DATA] paths OK (tal cual): {len(exist1)}")
        return exist1

    print("[DATA][WARN] 0 paths existentes usando rutas absolutas del CSV. Intento rebase...")

    # batch_2) Rebase
    if tokens_dir.exists():
        paths2 = [rebase_path(p, tokens_dir, anchor) for p in raw_paths]
        exist2 = [p for p in paths2 if p.exists()]
        if len(exist2) > 0:
            print(f"[DATA] paths OK (rebase): {len(exist2)}")
            return exist2

    print("[DATA][WARN] 0 paths existentes tras rebase. Fallback: escaneo TOKENS_DIR...")

    # batch_3) Scan
    if not tokens_dir.exists():
        raise FileNotFoundError(f"TOKENS_DIR no existe: {tokens_dir}")

    scan = sorted([p for p in tokens_dir.rglob("*.json") if p.is_file()])
    print(f"[DATA] paths OK (scan): {len(scan)}")
    return scan


def split_train_val_test(paths: List[Path], val_ratio: float, test_ratio: float, seed: int):
    rng = random.Random(seed)
    p = paths[:]
    rng.shuffle(p)
    n = len(p)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    test_paths = p[:n_test]
    val_paths = p[n_test:n_test + n_val]
    train_paths = p[n_test + n_val:]
    return train_paths, val_paths, test_paths


# =============================================================================
# VALIDACIÓN DE JSONS
# =============================================================================

def validate_token_json(path: Path, token_field: str) -> Tuple[bool, str, int]:
    """
    Devuelve:
      - ok: si el fichero es válido
      - reason: descripción del error o "ok"
      - n_tokens: longitud de ids si es válido, 0 si no
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        return False, f"UnicodeDecodeError: {e}", 0
    except Exception as e:
        return False, f"ReadError: {e}", 0

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return False, f"JSONDecodeError: {e}", 0
    except Exception as e:
        return False, f"JSONLoadError: {e}", 0

    if not isinstance(obj, dict):
        return False, f"Formato inesperado: type={type(obj).__name__}, se esperaba dict", 0

    if token_field not in obj:
        return False, f"Falta campo '{token_field}'", 0

    ids = obj[token_field]
    if not isinstance(ids, list):
        return False, f"'{token_field}' no es una lista", 0

    if len(ids) == 0:
        return False, f"'{token_field}' está vacío", 0

    for i, x in enumerate(ids[:20]):  # comprobación rápida de los primeros
        if not isinstance(x, int):
            return False, f"'{token_field}' contiene no-int en posición {i}: {type(x).__name__}", 0

    return True, "ok", len(ids)


def filter_valid_json_paths(paths: List[Path], token_field: str) -> Tuple[List[Path], List[Tuple[Path, str]]]:
    valid = []
    bad = []

    for i, path in enumerate(paths, start=1):
        ok, reason, n_tokens = validate_token_json(path, token_field)

        if ok:
            valid.append(path)
        else:
            bad.append((path, reason))
            print(f"[BAD {len(bad)}] {path} -> {reason}")

        if i % 2000 == 0:
            print(f"[CHECK] revisados {i}/{len(paths)} | válidos={len(valid)} | bad={len(bad)}")

    return valid, bad


# =============================================================================
# CONSTRUCCIÓN DEL MEMMAP
# =============================================================================

def build_memmap(files: List[Path], out_bin: Path, token_field: str, dtype, add_eos: bool, eos_id: int) -> int:
    if len(files) == 0:
        raise ValueError("No hay ficheros válidos para construir el memmap.")

    total = 0
    for p in files:
        obj = json.loads(p.read_text(encoding="utf-8"))
        ids = obj[token_field]
        total += len(ids) + (1 if add_eos else 0)

    if total <= 0:
        raise ValueError("Total tokens = 0.")

    out_bin.parent.mkdir(parents=True, exist_ok=True)

    mm = np.memmap(out_bin, mode="w+", dtype=dtype, shape=(total,))
    w = 0

    for i, p in enumerate(files, start=1):
        obj = json.loads(p.read_text(encoding="utf-8"))
        ids = obj[token_field]

        arr = np.asarray(ids, dtype=dtype)
        mm[w:w + len(arr)] = arr
        w += len(arr)

        if add_eos:
            mm[w] = np.asarray([eos_id], dtype=dtype)[0]
            w += 1

        if i % 2000 == 0:
            print(f"[MEMMAP] {i}/{len(files)} escritos | tokens={w:,}")

    mm.flush()
    print(f"[OK] Memmap creado: {out_bin}")
    print(f"[OK] Total tokens escritos: {w:,}")
    return w


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("[INFO] Resolviendo rutas de JSON...")
    paths = resolve_json_paths(INDEX_CSV, TOKENS_DIR, ANCHOR)

    if LIMIT_FILES is not None:
        paths = paths[:LIMIT_FILES]
        print(f"[INFO] LIMIT_FILES activo: {len(paths)} archivos")

    print(f"[INFO] Total paths encontrados: {len(paths)}")

    train_files, val_files, test_files = split_train_val_test(paths, VAL_RATIO, TEST_RATIO, SEED)
    print(f"[INFO] split -> train={len(train_files)} val={len(val_files)} test={len(test_files)}")

    print("\n[INFO] Validando JSONs de train...")
    valid_train, bad_train = filter_valid_json_paths(train_files, TOKEN_FIELD)

    print("\n================ RESUMEN VALIDACIÓN ================")
    print(f"[INFO] train_files originales: {len(train_files)}")
    print(f"[INFO] train_files válidos:    {len(valid_train)}")
    print(f"[INFO] train_files erróneos:   {len(bad_train)}")

    if bad_train:
        print("\n[INFO] Primeros errores encontrados:")
        for path, reason in bad_train[:20]:
            print(f"  - {path} -> {reason}")

        bad_report = CACHE_DIR / "bad_json_report.txt"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        bad_report.write_text(
            "\n".join(f"{path}\t{reason}" for path, reason in bad_train),
            encoding="utf-8"
        )
        print(f"\n[INFO] Reporte completo guardado en: {bad_report}")

    if not valid_train:
        print("\n[ERROR] No hay archivos válidos. No se construirá el memmap.")
        return

    dtype = choose_np_dtype(USE_UINT16, VOCAB_SIZE)

    print("\n[INFO] Construyendo memmap solo con archivos válidos...")
    build_memmap(
        files=valid_train,
        out_bin=OUT_BIN,
        token_field=TOKEN_FIELD,
        dtype=dtype,
        add_eos=ADD_EOS,
        eos_id=EOS_ID,
    )


if __name__ == "__main__":
    main()