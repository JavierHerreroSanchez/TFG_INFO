from __future__ import annotations

from pathlib import Path
from multiprocessing import get_context
from symusic import Score

from tokenizer_train import load_bpe_tokenizer, list_midi_files, ensure_trained_bpe_tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = (PROJECT_ROOT / "data").resolve()
OUT_ROOT = (PROJECT_ROOT / "tokenizer").resolve()
TOKENIZER_FILENAME = "tokenizer_REMI_BPE.json"

TOKENS_DIR = (OUT_ROOT / "tokens_json_bpe").resolve()
BAD_LIST = (OUT_ROOT / "bad_midis.txt").resolve()


def output_path_for(midi_path: Path, data_root: Path, tokens_dir: Path) -> Path:
    rel = midi_path.resolve().relative_to(data_root.resolve())
    return (tokens_dir / rel).with_suffix(".json")


def worker(midi_path_str: str, out_json_str: str, tokenizer_filename: str) -> None:
    """
    Trabajo real del hijo. Si peta, el proceso devolverá exitcode != 0.
    """
    midi_path = Path(midi_path_str)
    out_json = Path(out_json_str)

    tokenizer = load_bpe_tokenizer(OUT_ROOT, tokenizer_filename)
    score = Score(midi_path)
    tokens = tokenizer.encode(score)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save_tokens(tokens, out_json, programs=None)


def run_one(midi_path: Path, out_json: Path) -> int:
    """
    Lanza un proceso (spawn) que tokeniza 1 MIDI.
    Devuelve exitcode (0 OK).
    """
    ctx = get_context("spawn")  # Windows
    p = ctx.Process(target=worker, args=(str(midi_path), str(out_json), TOKENIZER_FILENAME))
    p.start()
    p.join()
    return 1 if p.exitcode is None else p.exitcode


def tokenize_corpus(data_root: Path = DATA_ROOT, tokens_dir: Path = TOKENS_DIR) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tokens_dir.mkdir(parents=True, exist_ok=True)

    tok_path = ensure_trained_bpe_tokenizer()
    print(f"[INFO] Tokenizer: {tok_path}", flush=True)

    midi_paths = list_midi_files(data_root)
    print(f"[INFO] MIDIs encontrados: {len(midi_paths)}", flush=True)
    print(f"[INFO] data_root: {data_root.resolve()}", flush=True)
    print(f"[INFO] out_dir: {tokens_dir.resolve()}", flush=True)

    bad: list[str] = []
    ok = 0

    for i, midi_path in enumerate(midi_paths, start=1):
        out_json = output_path_for(midi_path, data_root, tokens_dir)

        if out_json.exists():
            continue

        out_json.parent.mkdir(parents=True, exist_ok=True)

        print(f"[DOING] {i}/{len(midi_paths)} {midi_path}", flush=True)
        code = run_one(midi_path, out_json)

        if code == 0:
            ok += 1
            if ok % 200 == 0:
                print(f"[INFO] OK nuevos: {ok}", flush=True)
        else:
            bad.append(f"{midi_path}\texitcode={code}")
            print(f"[SKIP] exitcode={code} file={midi_path}", flush=True)

    BAD_LIST.write_text("\n".join(bad), encoding="utf-8")

    print("\n=== RESUMEN ===", flush=True)
    print(f"OK nuevos: {ok}", flush=True)
    print(f"BAD: {len(bad)}", flush=True)
    print(f"Bad list: {BAD_LIST}", flush=True)


def main() -> None:
    # PyCharm: Run y ya está
    tokenize_corpus()


if __name__ == "__main__":
    main()