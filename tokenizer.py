# tokenizer.py
from pathlib import Path
from miditok import REMI, TokenizerConfig

DATA_ROOT = Path("./data").resolve()
OUT_ROOT = Path("./tokenizer").resolve()
TOKENIZER_FILENAME = "tokenizer_REMI_BPE.json"

# Ajusta a tu gusto
VOCAB_SIZE = 30000          # debe ser > vocab base
TRAIN_LIMIT = None          # p.ej. 5000 para entrenar más rápido, o None para todos

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tok_path = (OUT_ROOT / TOKENIZER_FILENAME).resolve()

    # Si ya existe, no reentrenes (evitas cambios de ids)
    if tok_path.exists():
        print(f"[INFO] Tokenizer ya existe: {tok_path}")
        return

    midi_paths = sorted(list(DATA_ROOT.rglob("*.mid")) + list(DATA_ROOT.rglob("*.midi")))
    if TRAIN_LIMIT is not None:
        midi_paths = midi_paths[:TRAIN_LIMIT]

    print(f"[INFO] Entrenando BPE con {len(midi_paths)} MIDIs | vocab_size={VOCAB_SIZE}")

    # Config base (igual que el tuyo)
    config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True, encode_ids_split="bar")
    tokenizer = REMI(config)

    # Entrenar vocabulario con HuggingFace tokenizers (BPE)
    tokenizer.train(vocab_size=VOCAB_SIZE, model="BPE", files_paths=midi_paths)

    # Guardar tokenizer ya entrenado
    tokenizer.save(OUT_ROOT, None, TOKENIZER_FILENAME)
    print(f"[OK] Guardado tokenizer BPE en: {tok_path}")
if __name__ == "__main__":
    main()