from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from miditok.utils import get_score_programs
from symusic import Score

from src.tokenization.deprecated.tokenizer_train import load_bpe_tokenizer


# ============================================================
# RUTAS
# ============================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "pretraining_raw"
TOKENS_DIR = PROJECT_ROOT / "data" / "interim" / "tokenized_json_bpe"
BAD_LIST = PROJECT_ROOT / "tokenizer" / "bad_midis.txt"
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "tokenizer_REMI_BPE_v3.json"

DATASETS_TO_SCAN = [
    DATA_RAW / "maestro-v3.0.0",
    DATA_RAW / "ariamidi",
]

TOKEN_FIELD_CANDIDATES = ("ids", "ids_encoded")

# Ajusta esto según tu CPU:
# prueba con os.cpu_count() // 2, o 6, 8, 10...
MAX_WORKERS = max(1, (os.cpu_count() or 8) - 2)

# Variables globales por worker
TOKENIZER = None


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
        bad_path = line.split("\t", 1)[0].strip()
        if bad_path:
            bad.add(bad_path)
    return bad


def append_bad_lines(path: Path, lines: list[str]) -> None:
    if not lines:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def json_has_valid_tokens(json_path: Path) -> bool:
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
    for root in DATASETS_TO_SCAN:
        try:
            rel_inside_root = midi_path.relative_to(root)
            return Path(root.name) / rel_inside_root
        except ValueError:
            continue

    try:
        return midi_path.relative_to(DATA_RAW)
    except ValueError:
        return Path(midi_path.name)


def build_output_json_path(midi_path: Path) -> Path:
    rel = dataset_relative_path(midi_path)
    return (TOKENS_DIR / rel).with_suffix(".json")


# ============================================================
# INIT por worker
# ============================================================
def worker_init():
    global TOKENIZER
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"No existe el tokenizador: {TOKENIZER_PATH}")

    TOKENIZER = load_bpe_tokenizer(TOKENIZER_PATH.parent, TOKENIZER_PATH.name)


# ============================================================
# TRABAJO por MIDI
# ============================================================
def process_one(midi_str: str) -> tuple[str, str, str]:
    """
    Devuelve:
      ("ok", midi_path, "")
      ("skip", midi_path, "already_done")
      ("bad", midi_path, "motivo")
    """
    global TOKENIZER

    midi_path = Path(midi_str)
    out_json = build_output_json_path(midi_path)

    try:
        if json_has_valid_tokens(out_json):
            return ("skip", str(midi_path), "already_done")

        if out_json.exists():
            try:
                out_json.unlink()
            except Exception:
                pass

        score = Score(midi_path)
        tokens = TOKENIZER.encode(score)

        out_json.parent.mkdir(parents=True, exist_ok=True)

        save_programs = not TOKENIZER.config.use_programs
        programs = None
        if save_programs and get_score_programs is not None:
            programs = get_score_programs(score)

        TOKENIZER.save_tokens(tokens, out_json, programs)

        if not json_has_valid_tokens(out_json):
            try:
                out_json.unlink()
            except Exception:
                pass
            return ("bad", str(midi_path), "invalid_output_json")

        return ("ok", str(midi_path), "")

    except Exception as e:
        try:
            if out_json.exists() and not json_has_valid_tokens(out_json):
                out_json.unlink()
        except Exception:
            pass
        return ("bad", str(midi_path), f"{type(e).__name__}:{e}")


# ============================================================
# MAIN
# ============================================================
def main():
    TOKENS_DIR.mkdir(parents=True, exist_ok=True)
    BAD_LIST.parent.mkdir(parents=True, exist_ok=True)

    previous_bad = load_bad_set(BAD_LIST)
    midi_paths = find_midis(DATASETS_TO_SCAN)

    todo = [p for p in midi_paths if str(p) not in previous_bad]

    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[INFO] TOKENIZER    = {TOKENIZER_PATH}")
    print(f"[INFO] TOKENS_DIR    = {TOKENS_DIR}")
    print(f"[INFO] BAD_LIST      = {BAD_LIST}")
    print(f"[INFO] MIDIs totales = {len(midi_paths)}")
    print(f"[INFO] A procesar    = {len(todo)}")
    print(f"[INFO] WORKERS       = {MAX_WORKERS}")

    ok = 0
    skip = 0
    bad = 0
    bad_lines: list[str] = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=worker_init) as ex:
        futures = [ex.submit(process_one, str(p)) for p in todo]

        for i, fut in enumerate(as_completed(futures), start=1):
            status, midi_path, reason = fut.result()

            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                bad += 1
                bad_lines.append(f"{midi_path}\t{reason}")

            if i % 1000 == 0:
                print(f"[INFO] {i}/{len(todo)} | ok={ok} skip={skip} bad={bad}")

            if len(bad_lines) >= 500:
                append_bad_lines(BAD_LIST, bad_lines)
                bad_lines.clear()

    append_bad_lines(BAD_LIST, bad_lines)

    print("\n=== RESUMEN ===")
    print(f"OK:   {ok}")
    print(f"SKIP: {skip}")
    print(f"BAD:  {bad}")


if __name__ == "__main__":
    main()