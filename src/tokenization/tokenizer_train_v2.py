from pathlib import Path
import random
from typing import Literal
from miditok import REMI, TokenizerConfig

# =========================
# CONFIGURACIÓN
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MAESTRO_ROOT = (PROJECT_ROOT / "data/pretraining_raw" / "maestro-v3.0.0").resolve()
ARIA_ROOT = (PROJECT_ROOT / "data/pretraining_raw" / "ariamidi").resolve()

OUT_ROOT = (PROJECT_ROOT / "tokenizer").resolve()
TOKENIZER_FILENAME = "tokenizer_REMI_BPE_v4.json"

VOCAB_SIZE = 30000
ENCODE_IDS_SPLIT: Literal["bar", "beat", "no"] = "bar"
SEED = 1453

# Usar todo MAESTRO y solo una muestra de ARIA
USE_ALL_MAESTRO = True
ARIA_SAMPLE_SIZE = 80000        #30000 con el v2

def list_midi_files(root: Path) -> list[Path]:
    """Devuelve todos los .mid y .midi bajo root."""
    return sorted(list(root.rglob("*.mid")) + list(root.rglob("*.midi")))


def sample_files(paths: list[Path], n: int, seed: int) -> list[Path]:
    """Muestra aleatoria reproducible sin reemplazo."""
    if n >= len(paths):
        return paths[:]
    rng = random.Random(seed)
    return rng.sample(paths, n)


def build_training_file_list() -> list[Path]:
    """Construye la lista final de MIDIs para entrenar BPE."""
    maestro_files = list_midi_files(MAESTRO_ROOT)
    aria_files = list_midi_files(ARIA_ROOT)

    if USE_ALL_MAESTRO:
        selected_maestro = maestro_files
    else:
        selected_maestro = []

    selected_aria = sample_files(aria_files, ARIA_SAMPLE_SIZE, SEED)

    train_files = sorted(selected_maestro + selected_aria)

    print(f"[INFO] MAESTRO encontrados: {len(maestro_files)}")
    print(f"[INFO] ARIA encontrados: {len(aria_files)}")
    print(f"[INFO] MAESTRO usados: {len(selected_maestro)}")
    print(f"[INFO] ARIA usados: {len(selected_aria)}")
    print(f"[INFO] Total para train BPE: {len(train_files)}")

    return train_files


def build_tokenizer() -> REMI:
    """Crea el tokenizer base REMI."""
    config = TokenizerConfig(
        pitch_range=(21, 109),
        beat_res={(0, 4): 8, (4, 12): 4},
        encode_ids_split=ENCODE_IDS_SPLIT,

        num_velocities=8,
        use_velocities=True,
        use_note_duration_programs=[0],

        use_chords=True,
        chord_tokens_with_root_note=True,
        # usar chord_maps por defecto
        # no tocar chord_unknown

        use_rests=True,
        beat_res_rest={(0, 1): 8, (1, 2): 4, (2, 12): 2},

        use_time_signatures=True,
        use_tempos=False,

        use_programs=False,
        one_token_stream_for_programs=False,
        program_changes=False,

        use_pitch_intervals=False,
        # max_pitch_interval y pitch_intervals_max_time_dist no aplican aquí

        use_sustain_pedals=False,
        use_pitch_bends=False,

        ac_polyphony_track=False,
        ac_polyphony_bar=True,
        ac_polyphony_min=1,
        ac_polyphony_max=6,

        ac_pitch_class_bar=True,

        ac_note_density_track=False,
        ac_note_density_bar=True,
        ac_note_density_bar_max=18,

        ac_note_duration_bar=True,
        ac_note_duration_track=False,

        ac_repetition_track=True,
        ac_repetition_track_num_bins=8,
        ac_repetition_track_num_consec_bars=4,
    )
    return REMI(config)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tok_path = (OUT_ROOT / TOKENIZER_FILENAME).resolve()

    if tok_path.exists():
        print(f"[INFO] El tokenizer ya existe: {tok_path}")
        print("[INFO] Bórralo o cambia TOKENIZER_FILENAME si quieres reentrenarlo.")
        return

    train_files = build_training_file_list()
    if not train_files:
        raise RuntimeError("No se han encontrado archivos MIDI para entrenar el tokenizer.")

    tokenizer = build_tokenizer()

    print(
        f"[INFO] Entrenando tokenizer BPE | vocab_size={VOCAB_SIZE} "
        f"| encode_ids_split={ENCODE_IDS_SPLIT}"
    )
    tokenizer.train(
        vocab_size=VOCAB_SIZE,
        model="BPE",
        files_paths=train_files,
    )

    tokenizer.save(OUT_ROOT, None, TOKENIZER_FILENAME)
    print(f"[OK] Tokenizer guardado en: {tok_path}")


if __name__ == "__main__":
    main()