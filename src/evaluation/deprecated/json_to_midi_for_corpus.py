

from pathlib import Path
from miditok import REMI

TOKENIZER_PATH = Path(r"/tokenizer/tokenizer_REMI_BPE_v3.json")
INPUT_JSON = Path(
    r"/data/interim/tokenized_json_bpe/maestro-v3.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.json")
OUTPUT_MIDI = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\ARIA_CHECK.mid")

def main():
    tokenizer = REMI(params=TOKENIZER_PATH)

    # Cargar con el loader oficial de MidiTok
    seq = tokenizer.load_tokens(INPUT_JSON)

    print(f"type(seq): {type(seq)}")
    if hasattr(seq, "are_ids_encoded"):
        print(f"are_ids_encoded: {seq.are_ids_encoded}")

    score = tokenizer.decode(seq)
    OUTPUT_MIDI.parent.mkdir(parents=True, exist_ok=True)
    score.dump_midi(OUTPUT_MIDI)

    print(f"[OK] MIDI guardado en: {OUTPUT_MIDI}")

if __name__ == "__main__":
    main()