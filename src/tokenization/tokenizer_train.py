"""
Entrenamiento y carga del tokenizador REMI+BPE usado en el proyecto.

El tokenizador transforma archivos MIDI simbólicos en secuencias de tokens REMI y
posteriormente aplica BPE para reducir la longitud de las secuencias. La misma
configuración se reutiliza en pretraining, fine-tuning, generación y evaluación.
"""

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
TOKENIZER_FILENAME = "tokenizer_REMI_BPE_v5.json"

VOCAB_SIZE = 18000
ENCODE_IDS_SPLIT: Literal["bar", "beat", "no"] = "bar"
SEED = 1453

# Se usa todo MAESTRO y una muestra reproducible de ARIA.
USE_ALL_MAESTRO = True
ARIA_SAMPLE_SIZE = 100000

CUSTOM_CHORDS = {
    # triadas base
    "min": (0, 3, 7),
    "maj": (0, 4, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),

    # séptimas base
    "7dom": (0, 4, 7, 10),
    "7min": (0, 3, 7, 10),
    "7maj": (0, 4, 7, 11),
    "7halfdim": (0, 3, 6, 10),
    "7dim": (0, 3, 6, 9),
    "7aug": (0, 4, 8, 11),

    # novenas base
    "9maj": (0, 4, 7, 10, 14),
    "9min": (0, 4, 7, 10, 13),

    # duplicación de octava en triadas
    "min8": (0, 3, 7, 12),
    "maj8": (0, 4, 7, 12),
    "dim8": (0, 3, 6, 12),
    "aug8": (0, 4, 8, 12),
    "sus28": (0, 2, 7, 12),
    "sus48": (0, 5, 7, 12),

    # duplicación de octava en séptimas frecuentes
    "7dom8": (0, 4, 7, 10, 12),
    "7min8": (0, 3, 7, 10, 12),
    "7maj8": (0, 4, 7, 11, 12),

    # voicings abiertos muy frecuentes en piano
    "maj_open": (0, 7, 12, 16),
    "min_open": (0, 7, 12, 15),
}


def list_midi_files(root: Path) -> list[Path]:
    """Devuelve todos los .mid y .midi bajo root."""
    return sorted(list(root.rglob("*.mid")) + list(root.rglob("*.midi")))


def tokenizer_path(out_root: Path = OUT_ROOT, filename: str = TOKENIZER_FILENAME) -> Path:
    """Devuelve la ruta absoluta del tokenizador guardado."""
    return (out_root.resolve() / filename).resolve()


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

        num_velocities=16,
        use_velocities=True,

        chord_maps=CUSTOM_CHORDS,
        use_chords=True,
        chord_tokens_with_root_note=True,
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
        use_pitchdrum_tokens= False,
        ac_note_density_track=False,
        ac_note_density_bar=True,
        ac_note_density_bar_max=18,

        ac_note_duration_bar=True,
        ac_note_duration_track=False,
    )
    return REMI(config)


def load_bpe_tokenizer(out_root: Path = OUT_ROOT, filename: str = TOKENIZER_FILENAME) -> REMI:
    """Carga desde disco el tokenizador REMI+BPE entrenado."""
    return REMI(params=tokenizer_path(out_root, filename))


def main() -> None:
    """Punto de entrada del script cuando se ejecuta desde consola."""

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tok_path = (OUT_ROOT / TOKENIZER_FILENAME).resolve()

    if tok_path.exists():
        print(f"[INFO] El tokenizer ya existe: {tok_path}")
        print("[INFO] Eliminarlo o cambiar TOKENIZER_FILENAME para reentrenarlo.")
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


# Ejecucion directa del script.
if __name__ == "__main__":
    main()
