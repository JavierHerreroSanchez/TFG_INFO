from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import librosa
import librosa.display

# Opción A: usando pretty_midi.fluidsynth()
USE_PRETTY_MIDI_FLUIDSYNTH = True

# Opción B: si prefieres generar WAV con midi2audio fuera de este script, pon False
# y carga directamente un WAV ya renderizado
EXTERNAL_WAV_PATH = None  # Path(r"output/generated.wav")


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

MIDI_PATH = Path(r"../../../output/generation_v2/best_test/generated_from_json2.mid")

# Si pretty_midi.fluidsynth necesita SoundFont explícita, indícala aquí.
# Si no, déjalo en None e inténtalo igualmente.
SOUNDFONT_PATH = None
# Ejemplo:
# SOUNDFONT_PATH = Path(r"C:\ruta\a\soundfont.sf2")

SAMPLE_RATE = 22050

PITCH_MIN = 21
PITCH_MAX = 108
PIANOROLL_FS = 100

SAVE_PIANOROLL_PNG = Path(r"../../../output/generation_v2/generated_pianoroll.png")
SAVE_SPECTROGRAM_PNG = Path(r"../../../output/generation_v2/generated_spectrogram.png")


# =============================================================================
# MÉTRICAS MIDI SIMPLES
# =============================================================================

def summarize_midi(pm: pretty_midi.PrettyMIDI) -> dict:
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        notes.extend(inst.notes)

    notes = sorted(notes, key=lambda n: (n.start, n.pitch))

    if len(notes) == 0:
        return {
            "num_instruments": len(pm.instruments),
            "num_notes": 0,
            "duration_sec": pm.get_end_time(),
        }

    pitches = np.array([n.pitch for n in notes], dtype=np.int32)
    durations = np.array([n.end - n.start for n in notes], dtype=np.float32)
    velocities = np.array([n.velocity for n in notes], dtype=np.int32)
    onsets = np.array([n.start for n in notes], dtype=np.float32)

    ioi = np.diff(onsets) if len(onsets) > 1 else np.array([], dtype=np.float32)

    return {
        "num_instruments": len(pm.instruments),
        "num_notes": len(notes),
        "duration_sec": float(pm.get_end_time()),
        "pitch_min": int(pitches.min()),
        "pitch_max": int(pitches.max()),
        "pitch_mean": float(pitches.mean()),
        "duration_mean_sec": float(durations.mean()),
        "velocity_mean": float(velocities.mean()),
        "ioi_mean_sec": float(ioi.mean()) if len(ioi) > 0 else None,
    }


# =============================================================================
# PIANO ROLL
# =============================================================================

def plot_pianoroll(pm: pretty_midi.PrettyMIDI):
    roll = pm.get_piano_roll(fs=PIANOROLL_FS)
    roll = roll[PITCH_MIN:PITCH_MAX + 1]

    duration_sec = roll.shape[1] / PIANOROLL_FS
    fig_w = max(12, duration_sec / 2.5)

    plt.figure(figsize=(fig_w, 6))
    plt.imshow(
        roll,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[0, duration_sec, PITCH_MIN, PITCH_MAX + 1],
    )
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Pitch MIDI")
    plt.title(f"Piano roll - {MIDI_PATH.name}")
    plt.colorbar(label="Velocidad / intensidad acumulada")
    plt.tight_layout()

    SAVE_PIANOROLL_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(SAVE_PIANOROLL_PNG, dpi=200, bbox_inches="tight")
    print(f"[OK] Piano-roll guardado en: {SAVE_PIANOROLL_PNG.resolve()}")
    plt.show()


# =============================================================================
# MIDI -> AUDIO
# =============================================================================

def synthesize_audio(pm: pretty_midi.PrettyMIDI) -> tuple[np.ndarray, int]:
    """
    Devuelve waveform + sample rate.

    Prioridad:
    1) WAV externo, si se ha indicado.
    batch_2) FluidSynth, si está disponible.
    batch_3) pretty_midi.synthesize() como fallback sin dependencias externas.
    """
    if EXTERNAL_WAV_PATH is not None:
        if not EXTERNAL_WAV_PATH.exists():
            raise FileNotFoundError(f"No existe WAV externo: {EXTERNAL_WAV_PATH.resolve()}")
        y, sr = librosa.load(str(EXTERNAL_WAV_PATH), sr=SAMPLE_RATE, mono=True)
        return y, sr

    if USE_PRETTY_MIDI_FLUIDSYNTH:
        try:
            if SOUNDFONT_PATH is not None:
                y = pm.fluidsynth(fs=SAMPLE_RATE, sf2_path=str(SOUNDFONT_PATH))
            else:
                y = pm.fluidsynth(fs=SAMPLE_RATE)

            print("[INFO] Audio sintetizado con FluidSynth.")
            return y, SAMPLE_RATE

        except Exception as e:
            print(f"[WARN] FluidSynth no disponible: {e}")
            print("[WARN] Se usará pretty_midi.synthesize() como fallback.")

    # Fallback interno de pretty_midi
    y = pm.synthesize(fs=SAMPLE_RATE)
    print("[INFO] Audio sintetizado con pretty_midi.synthesize().")
    return y, SAMPLE_RATE

# =============================================================================
# ESPECTROGRAMA
# =============================================================================

def plot_melspectrogram(y: np.ndarray, sr: int):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(14, 6))
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+batch_2.0f dB")
    plt.title(f"Mel-spectrogram - {MIDI_PATH.name}")
    plt.tight_layout()

    SAVE_SPECTROGRAM_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(SAVE_SPECTROGRAM_PNG, dpi=200, bbox_inches="tight")
    print(f"[OK] Espectrograma guardado en: {SAVE_SPECTROGRAM_PNG.resolve()}")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not MIDI_PATH.exists():
        raise FileNotFoundError(f"No existe el MIDI: {MIDI_PATH.resolve()}")

    pm = pretty_midi.PrettyMIDI(str(MIDI_PATH))

    print("=" * 90)
    print("ANÁLISIS DEL MIDI")
    print("=" * 90)
    print(f"Archivo: {MIDI_PATH.resolve()}")

    summary = summarize_midi(pm)
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nMostrando piano-roll...")
    plot_pianoroll(pm)

    print("\nIntentando sintetizar a audio para espectrograma...")
    y, sr = synthesize_audio(pm)
    print(f"[INFO] Audio sintetizado: {len(y)} muestras | sr={sr}")

    print("\nMostrando mel-spectrogram...")
    plot_melspectrogram(y, sr)


if __name__ == "__main__":
    main()