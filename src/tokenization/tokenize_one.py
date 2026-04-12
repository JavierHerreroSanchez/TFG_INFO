from pathlib import Path
import sys
import traceback

from miditok.utils import get_score_programs
from symusic import Score

from src.tokenization.deprecated.tokenizer_train import load_bpe_tokenizer


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]  # TFG_INFO/
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "tokenizer_REMI_BPE_v3.json"


def main():
    if len(sys.argv) < 3:
        raise ValueError("Uso: python tokenize_one.py <midi_path> <out_json>")

    midi_path = Path(sys.argv[1]).resolve()
    out_json = Path(sys.argv[2]).resolve()

    if not midi_path.exists():
        raise FileNotFoundError(f"No existe el MIDI: {midi_path}")

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"No existe el tokenizador BPE: {TOKENIZER_PATH}")

    # Si load_bpe_tokenizer espera (root, filename), le pasamos ambos de forma robusta
    tokenizer = load_bpe_tokenizer(TOKENIZER_PATH.parent, TOKENIZER_PATH.name)

    score = Score(midi_path)
    tokens = tokenizer.encode(score)

    out_json.parent.mkdir(parents=True, exist_ok=True)

    save_programs = not tokenizer.config.use_programs
    programs = None
    if save_programs and get_score_programs is not None:
        programs = get_score_programs(score)

    tokenizer.save_tokens(tokens, out_json, programs)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)