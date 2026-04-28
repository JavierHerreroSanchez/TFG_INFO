from pathlib import Path
import json

from miditok import REMI, TokSequence
from tokenizers import Tokenizer

JSON_PATH = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\data\interim\tokenized_json_bpe\maestro-v3.0.0\2004\MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.json")
TOKENIZER_PATH = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\tokenizer\tokenizer_REMI_BPE_v3.json")
OUT_MID = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\output\test_decode.mid")

# ------------------------------------------------------------
# 1) tokenizer musical REMI
# ------------------------------------------------------------
music_tokenizer = REMI(params=TOKENIZER_PATH)

# ------------------------------------------------------------
# batch_2) ids BPE del json
# ------------------------------------------------------------
obj = json.loads(JSON_PATH.read_text(encoding="utf-8"))
ids_bpe = obj["ids"]

print(f"[DEBUG] n_ids={len(ids_bpe)} min={min(ids_bpe)} max={max(ids_bpe)}")

# ------------------------------------------------------------
# batch_3) cargar tokenizer HF-BPE embebido
# ------------------------------------------------------------
tok_obj = json.loads(TOKENIZER_PATH.read_text(encoding="utf-8"))
hf_model_json = tok_obj["_model"]
hf_bpe = Tokenizer.from_str(hf_model_json)

# vocab del modelo BPE: string_piece -> id
hf_vocab = hf_bpe.get_vocab()

# inversa: id -> string_piece
id_to_piece = {v: k for k, v in hf_vocab.items()}

# ------------------------------------------------------------
# 4) ids BPE -> piezas BPE (NO usar decode aquí)
# ------------------------------------------------------------
pieces = [id_to_piece[i] for i in ids_bpe]

print(f"[DEBUG] n_pieces={len(pieces)}")
print(f"[DEBUG] primeras pieces={pieces[:20]}")

# concatenar piezas BPE en una sola secuencia de símbolos
merged = "".join(pieces)

print(f"[DEBUG] merged ejemplo={merged[:300]}")

# ------------------------------------------------------------
# 5) separar por metaspace ▁
# Cada símbolo restante representa un id base del vocab REMI.
# ------------------------------------------------------------
segments = [seg for seg in merged.split("▁") if seg]

print(f"[DEBUG] n_segments={len(segments)}")
print(f"[DEBUG] primeros segments={segments[:10]}")

# construir ids base REMI
base_ids = []
for seg in segments:
    for ch in seg:
        if ch not in hf_vocab:
            raise KeyError(f"Carácter no encontrado en vocab BPE: {repr(ch)}")
        base_ids.append(hf_vocab[ch])

print(f"[DEBUG] n_base_ids={len(base_ids)}")
print(f"[DEBUG] primeros base_ids={base_ids[:30]}")

# opcional: ver primeros tokens musicales
preview_tokens = [music_tokenizer[i] for i in base_ids[:30]]
print(f"[DEBUG] primeros tokens base={preview_tokens}")

# ------------------------------------------------------------
# 6) decodificar a MIDI
# ------------------------------------------------------------
seq = TokSequence(ids=base_ids)
midi = music_tokenizer.decode(seq)

OUT_MID.parent.mkdir(parents=True, exist_ok=True)

if hasattr(midi, "dump_midi"):
    midi.dump_midi(OUT_MID)
elif hasattr(midi, "write"):
    midi.write(OUT_MID)
else:
    raise TypeError(f"No sé guardar el objeto {type(midi)}")

print(f"[OK] MIDI guardado en: {OUT_MID}")