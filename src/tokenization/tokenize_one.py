#Código para tokenizar un solo MIDI, usado por un proceso padre de manera iterativa para tokenizar
#el corpus sin incurrir en problemas de memoria
from pathlib import Path
import sys

from miditok import REMI, TokenizerConfig
from miditok.utils import get_score_programs
from tokenizer_train import load_bpe_tokenizer
from symusic import Score

def main():
    midi_path = Path(sys.argv[1])
    out_json = Path(sys.argv[2])

    TOKENIZER_FILENAME = "../../tokenizer/tokenizer_REMI_BPE.json"
    OUT_ROOT = Path("").resolve()

    tokenizer = load_bpe_tokenizer(OUT_ROOT, TOKENIZER_FILENAME)

    score = Score(midi_path)

    tokens = tokenizer.encode(score)

    out_json.parent.mkdir(parents=True, exist_ok=True)

    save_programs = not tokenizer.config.use_programs
    programs = None
    if save_programs and get_score_programs is not None:
        programs = get_score_programs(score)

    tokenizer.save_tokens(tokens, out_json, programs)

if __name__ == "__main__":
    main()
