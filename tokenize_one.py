# worker_encode_and_save_json.py
from pathlib import Path
import sys

from miditok import REMI, TokenizerConfig
from miditok.utils import get_score_programs
from symusic import Score

def main():
    midi_path = Path(sys.argv[1])
    out_json = Path(sys.argv[2])

    config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
    tokenizer = REMI(config)

    score = Score(midi_path)

    tokens = tokenizer.encode(score)  # ✅ API correcta

    # tokseq puede ser TokSequence o lista[TokSequence]; guardamos en formato MidiTok
    # save_tokens guarda solo ids en un JSON con key "ids" (formato oficial). :contentReference[oaicite:3]{index=3}
    out_json.parent.mkdir(parents=True, exist_ok=True)

    save_programs = not tokenizer.config.use_programs
    programs = None
    if save_programs and get_score_programs is not None:
        programs = get_score_programs(score)

    tokenizer.save_tokens(tokens, out_json, programs)

if __name__ == "__main__":
    main()
