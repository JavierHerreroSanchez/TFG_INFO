from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pretty_midi


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

# Cambia esta ruta a tu MIDI real
MIDI_PATH = Path(r"/data/raw_old\clean_midi\Bach Johann Sebastian\Toccata and Fugue in D minor, BWV 565.mid")

# Frames por segundo del piano roll
FS = 100

# Rango de pitch a mostrar
PITCH_MIN = 21
PITCH_MAX = 108

# Si True, fusiona todos los instrumentos en una sola visualización
MERGE_INSTRUMENTS = True

# Si quieres guardar la imagen, pon aquí una ruta; si no, déjalo en None
SAVE_PNG_PATH = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\output/generated_pianoroll.png")
# SAVE_PNG_PATH = None


# =============================================================================
# CÓDIGO
# =============================================================================

def load_piano_roll(
    midi_path: Path,
    fs: int = 100,
    pitch_min: int = 21,
    pitch_max: int = 108,
    merge_instruments: bool = True,
):
    pm = pretty_midi.PrettyMIDI(str(midi_path))

    instrument_names = []

    if merge_instruments:
        roll = pm.get_piano_roll(fs=fs)
        for inst in pm.instruments:
            if inst.is_drum:
                instrument_names.append("Drums")
            else:
                instrument_names.append(pretty_midi.program_to_instrument_name(inst.program))
    else:
        rolls = []
        for inst in pm.instruments:
            r = inst.get_piano_roll(fs=fs)
            rolls.append(r)

            if inst.is_drum:
                instrument_names.append("Drums")
            else:
                instrument_names.append(pretty_midi.program_to_instrument_name(inst.program))

        if len(rolls) == 0:
            roll = np.zeros((128, 1), dtype=np.float32)
        else:
            max_t = max(r.shape[1] for r in rolls)
            padded = []
            for r in rolls:
                if r.shape[1] < max_t:
                    pad = np.zeros((128, max_t - r.shape[1]), dtype=r.dtype)
                    r = np.concatenate([r, pad], axis=1)
                padded.append(r)
            roll = np.sum(padded, axis=0)

    roll = roll[pitch_min:pitch_max + 1]
    return roll, instrument_names


def plot_piano_roll(
    roll: np.ndarray,
    fs: int,
    pitch_min: int,
    pitch_max: int,
    title: str,
    out_png: Path | None = None,
):
    if roll.size == 0:
        raise ValueError("El piano roll está vacío.")

    duration_sec = roll.shape[1] / fs

    fig_w = max(12, duration_sec / 2.5)
    fig_h = 6

    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(
        roll,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[0, duration_sec, pitch_min, pitch_max + 1],
    )
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Pitch MIDI")
    plt.title(title)
    plt.colorbar(label="Velocidad / intensidad acumulada")
    plt.tight_layout()

    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"[OK] Imagen guardada en: {out_png.resolve()}")

    plt.show()


def main():
    if not MIDI_PATH.exists():
        raise FileNotFoundError(f"No existe el MIDI: {MIDI_PATH.resolve()}")

    roll, instrument_names = load_piano_roll(
        midi_path=MIDI_PATH,
        fs=FS,
        pitch_min=PITCH_MIN,
        pitch_max=PITCH_MAX,
        merge_instruments=MERGE_INSTRUMENTS,
    )

    print("=" * 90)
    print("VISUALIZACIÓN MIDI")
    print("=" * 90)
    print(f"MIDI cargado      : {MIDI_PATH.resolve()}")
    print(f"Instrumentos      : {instrument_names if instrument_names else 'ninguno'}")
    print(f"Shape piano roll  : {roll.shape} = (pitches, frames)")
    print(f"Duración aprox    : {roll.shape[1] / FS:.2f} s")

    plot_piano_roll(
        roll=roll,
        fs=FS,
        pitch_min=PITCH_MIN,
        pitch_max=PITCH_MAX,
        title=f"Piano roll - {MIDI_PATH.name}",
        out_png=SAVE_PNG_PATH,
    )


if __name__ == "__main__":
    main()