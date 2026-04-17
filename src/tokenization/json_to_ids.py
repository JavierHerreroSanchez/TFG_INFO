from pathlib import Path
from miditok import REMI

TOKENIZER_PATH = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\tokenizer\tokenizer_REMI_BPE_v3.json")
JSON_PATH = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\data\interim\tokenized_json_bpe_v2\maestro-v3.0.0\2004\MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.json")


def main():
    tokenizer = REMI(params=TOKENIZER_PATH)

    # Carga el JSON como TokSequence
    seq = tokenizer.load_tokens(JSON_PATH)

    # Si los ids están BPE-encoded, descompónlos a ids base
    if getattr(seq, "are_ids_encoded", False):
        tokenizer.decode_token_ids(seq)

    # Completa los tokens legibles a partir de los ids base
    tokenizer.complete_sequence(seq, complete_bytes=False)

    print(f"Longitud ids en JSON: {len(seq.ids)}")
    print("-" * 100)

    for i, (tok_id, tok_txt) in enumerate(zip(seq.ids, seq.tokens)):
        print(f"{i:06d} | {tok_id:6d} | {tok_txt}")


if __name__ == "__main__":
    main()