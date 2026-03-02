# tokenizer_train.py


from __future__ import annotations

from pathlib import Path
from typing import Literal

from miditok import REMI, TokenizerConfig

# --- Configuración (ajusta aquí) ---
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = (PROJECT_ROOT / "data").resolve()
OUT_ROOT = (PROJECT_ROOT / "tokenizer").resolve()

TOKENIZER_FILENAME = "tokenizer_REMI_BPE.json"
VOCAB_SIZE = 30000
TRAIN_LIMIT: int | None = None
ENCODE_IDS_SPLIT: Literal["bar", "beat", "no"] = "bar"


def list_midi_files(root: Path) -> list[Path]:
    root = root.resolve()
    return sorted(list(root.rglob("*.mid")) + list(root.rglob("*.midi")))


def tokenizer_path(out_root: Path = OUT_ROOT, filename: str = TOKENIZER_FILENAME) -> Path:
    return (out_root.resolve() / filename).resolve()


def build_config() -> TokenizerConfig:
    return TokenizerConfig(
        num_velocities=16,
        use_chords=True,
        use_programs=True,
        encode_ids_split=ENCODE_IDS_SPLIT,
    )


def ensure_trained_bpe_tokenizer(
    data_root: Path = DATA_ROOT,
    out_root: Path = OUT_ROOT,
    filename: str = TOKENIZER_FILENAME,
    vocab_size: int = VOCAB_SIZE,
    train_limit: int | None = TRAIN_LIMIT,
) -> Path:
    
    data_root = data_root.resolve()
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    tok_path = tokenizer_path(out_root, filename)
    if tok_path.exists():
        print(f"[INFO] El tokenizer ya se encuentra entrenado y creado en {tok_path}", flush=True)
        return tok_path

    midi_paths = list_midi_files(data_root)
    if train_limit is not None:
        midi_paths = midi_paths[:train_limit]

    config = build_config()
    tokenizer = REMI(config)

    print(f"[INFO] Entrenando BPE con {len(midi_paths)} MIDIs | vocab_size={vocab_size}", flush=True)
    tokenizer.train(vocab_size=vocab_size, model="BPE", files_paths=midi_paths)
    tokenizer.save(out_root, None, filename)

    print(f"[OK] Tokenizer guardado en: {tok_path}", flush=True)
    return tok_path


def load_bpe_tokenizer(out_root: Path = OUT_ROOT, filename: str = TOKENIZER_FILENAME) -> REMI:
    
    path = tokenizer_path(out_root, filename)
    return REMI(params=path)


def main() -> None:
    ensure_trained_bpe_tokenizer()


if __name__ == "__main__":
    main()