# tokenize_corpus_safe.py
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = (PROJECT_ROOT / "data/pretraining_raw").resolve()
OUT_ROOT = (PROJECT_ROOT / "data/tokenized_json_bpe_v2").resolve()
TOKENS_DIR = (OUT_ROOT / "tokenized_json_bpe").resolve()
BAD_LIST = OUT_ROOT / "bad_midis.txt"

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TOKENS_DIR.mkdir(parents=True, exist_ok=True)

    midi_paths = sorted(list(DATA_ROOT.rglob("*.mid")) + list(DATA_ROOT.rglob("*.midi")))
    print(f"[INFO] MIDIs encontrados: {len(midi_paths)}")
    print(f"[INFO] data_root: {DATA_ROOT}")
    print(f"[INFO] out_dir: {TOKENS_DIR}")

    bad, ok = tokenize_corpus(midi_paths)

    BAD_LIST.write_text("\n".join(bad), encoding="utf-8")
    print("\n=== RESUMEN ===")
    print(f"OK nuevos: {ok}")
    print(f"BAD: {len(bad)}")
    print(f"Bad list: {BAD_LIST.resolve()}")


def tokenize_corpus(midi_paths: list[Path]) -> tuple[list[Any], int]:
    bad = []
    ok = 0

    for i, midi_path in enumerate(midi_paths, start=1):
        rel = midi_path.relative_to(DATA_ROOT)
        out_json = (TOKENS_DIR / rel).with_suffix(".json")
        out_json.parent.mkdir(parents=True, exist_ok=True)

        # reanudar: si ya existe, saltar
        if out_json.exists():
            continue

        print(f"[DOING] {i}/{len(midi_paths)} {midi_path}")

        r = subprocess.run(
            [sys.executable, "tokenize_one.py", str(midi_path), str(out_json)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if r.returncode == 0:
            ok += 1
            if ok % 500 == 0:
                print(f"[INFO] OK nuevos: {ok}")
        else:
            bad.append(f"{midi_path}\treturncode={r.returncode}")
            print(f"[SKIP] returncode={r.returncode} file={midi_path}")
    return bad, ok


if __name__ == "__main__":
    main()
