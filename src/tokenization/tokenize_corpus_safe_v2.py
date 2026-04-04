from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================
# Este script está pensado para ejecutarse desde PyCharm o desde
# la raíz del proyecto, pero calcula las rutas de forma robusta
# a partir de la ubicación del propio archivo.
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]   # TFG_INFO/

DATA_RAW = PROJECT_ROOT / "data" / "raw"
TOKENS_DIR = PROJECT_ROOT / "data" / "interim" / "tokenized_json_bpe"
BAD_LIST = PROJECT_ROOT / "tokenizer" / "bad_midis.txt"

# Ajustar aquí qué roots se quieren tokenizar
DATASETS_TO_SCAN = [
    DATA_RAW / "maestro-v3.0.0",
    DATA_RAW / "ariamidi",
]

# Script que tokeniza un solo archivo
TOKENIZE_ONE = THIS_FILE.parent / "tokenize_one.py"

# Campo esperado en el JSON de salida
TOKEN_FIELD_CANDIDATES = ("ids", "ids_encoded")

# Si True, no reintenta MIDIs que ya estén en bad_midis.txt
SKIP_PREVIOUS_BAD = True


# ============================================================
# UTILIDADES
# ============================================================
def find_midis(roots: Iterable[Path]) -> list[Path]:
    midi_paths: list[Path] = []
    for root in roots:
        if not root.exists():
            print(f"[WARN] No existe: {root}")
            continue
        midi_paths.extend(root.rglob("*.mid"))
        midi_paths.extend(root.rglob("*.midi"))
    return sorted({p.resolve() for p in midi_paths})


def load_bad_set(path: Path) -> set[str]:
    if not path.exists():
        return set()

    bad = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        # guardamos solo la ruta, aunque la línea lleve tabs/motivo
        bad_path = line.split("\t", 1)[0].strip()
        if bad_path:
            bad.add(bad_path)
    return bad


def append_bad(path: Path, midi_path: Path, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{midi_path}\t{reason}\n")


def json_has_valid_tokens(json_path: Path) -> bool:
    """
    Consideramos válido un JSON si:
    - existe
    - se puede parsear
    - contiene alguno de los campos esperados
    - ese campo es una lista no vacía
    """
    if not json_path.exists() or not json_path.is_file():
        return False

    try:
        obj = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    for field in TOKEN_FIELD_CANDIDATES:
        ids = obj.get(field, None)
        if isinstance(ids, list) and len(ids) > 0:
            return True

    return False


def dataset_relative_path(midi_path: Path) -> Path:
    """
    Devuelve una ruta relativa que conserve el nombre del dataset de origen,
    por ejemplo:
      data/raw/ariamidi/foo/bar.mid
        -> ariamidi/foo/bar.mid

      data/raw/maestro-v3.0.0/2004/file.mid
        -> maestro-v3.0.0/2004/file.mid
    """
    for root in DATASETS_TO_SCAN:
        try:
            rel_inside_root = midi_path.relative_to(root)
            return Path(root.name) / rel_inside_root
        except ValueError:
            continue

    # Fallback razonable
    try:
        return midi_path.relative_to(DATA_RAW)
    except ValueError:
        return Path(midi_path.name)


def build_output_json_path(midi_path: Path) -> Path:
    rel = dataset_relative_path(midi_path)
    return (TOKENS_DIR / rel).with_suffix(".json")


def run_tokenize_one(midi_path: Path, out_json: Path) -> tuple[int, str]:
    """
    Ejecuta tokenize_one.py y devuelve:
      (returncode, short_reason)
    """
    out_json.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [sys.executable, str(TOKENIZE_ONE), str(midi_path), str(out_json)],
            check=False,
        )
        return result.returncode, f"returncode={result.returncode}"
    except Exception as e:
        return 9999, f"exception={type(e).__name__}:{e}"


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    if not TOKENIZE_ONE.exists():
        raise FileNotFoundError(f"No existe tokenize_one.py en: {TOKENIZE_ONE}")

    TOKENS_DIR.mkdir(parents=True, exist_ok=True)
    BAD_LIST.parent.mkdir(parents=True, exist_ok=True)

    midi_paths = find_midis(DATASETS_TO_SCAN)
    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[INFO] DATA_RAW     = {DATA_RAW}")
    print(f"[INFO] TOKENS_DIR   = {TOKENS_DIR}")
    print(f"[INFO] BAD_LIST     = {BAD_LIST}")
    print(f"[INFO] TOKENIZE_ONE = {TOKENIZE_ONE}")
    print(f"[INFO] MIDIs encontrados: {len(midi_paths)}")

    previous_bad = load_bad_set(BAD_LIST) if SKIP_PREVIOUS_BAD else set()

    skipped_ok = 0
    skipped_bad = 0
    repaired = 0
    ok_new = 0
    bad_new = 0

    for i, midi_path in enumerate(midi_paths, start=1):
        out_json = build_output_json_path(midi_path)
        midi_key = str(midi_path.resolve())

        if SKIP_PREVIOUS_BAD and midi_key in previous_bad:
            skipped_bad += 1
            if skipped_bad % 500 == 0:
                print(f"[INFO] ya marcados como bad saltados: {skipped_bad}")
            continue

        # Reanudar bien: solo saltamos si el JSON existente es válido
        if json_has_valid_tokens(out_json):
            skipped_ok += 1
            if skipped_ok % 2000 == 0:
                print(f"[INFO] ya tokenizados válidos saltados: {skipped_ok}")
            continue

        # Si existe pero está corrupto/vacío, lo borramos y rehacemos
        if out_json.exists():
            try:
                out_json.unlink()
                repaired += 1
            except Exception:
                pass

        print(f"[DOING] {i}/{len(midi_paths)} -> {midi_path}")

        returncode, reason = run_tokenize_one(midi_path, out_json)

        if returncode == 0 and json_has_valid_tokens(out_json):
            ok_new += 1
            if ok_new % 500 == 0:
                print(f"[INFO] OK nuevos: {ok_new}")
        else:
            bad_new += 1

            # si se creó un JSON corrupto, mejor lo quitamos
            if out_json.exists() and not json_has_valid_tokens(out_json):
                try:
                    out_json.unlink()
                except Exception:
                    pass

            append_bad(BAD_LIST, midi_path, reason)
            print(f"[SKIP] {reason} file={midi_path}")

    print("\n=== RESUMEN ===")
    print(f"OK nuevos:                  {ok_new}")
    print(f"Rehechos por JSON inválido: {repaired}")
    print(f"Saltados ya tokenizados:    {skipped_ok}")
    print(f"Saltados por bad previo:    {skipped_bad}")
    print(f"BAD nuevos:                 {bad_new}")
    print(f"Bad list:                   {BAD_LIST.resolve()}")


if __name__ == "__main__":
    main()