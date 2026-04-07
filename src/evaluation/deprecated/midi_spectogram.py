import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pretty_midi

MIDI_PATH = Path(r"/output/generation_v2/generated_from_json0.mid")
SOUNDFONT = Path(r"C:\SoundFonts\FluidR3_GM.sf2")  # <-- pon aquí tu sf2
OUT_PNG = MIDI_PATH.with_suffix(".spectrogram.png")

SR = 22050
N_FFT = 2048
HOP = 256

def main():
    pm = pretty_midi.PrettyMIDI(str(MIDI_PATH))
    # Render a audio (mono)
    audio = pm.fluidsynth(fs=SR, sf2_path=str(SOUNDFONT))

    # STFT magnitude
    # (simple implementation)
    window = np.hanning(N_FFT)
    frames = []
    for i in range(0, len(audio) - N_FFT, HOP):
        x = audio[i:i+N_FFT] * window
        X = np.fft.rfft(x)
        frames.append(np.abs(X))
    S = np.stack(frames, axis=1)  # (freq, time)

    # Convert to dB
    S_db = 20 * np.log10(S + 1e-8)

    plt.figure()
    plt.imshow(S_db, aspect="auto", origin="lower")
    plt.xlabel("Frame")
    plt.ylabel("Frequency bin")
    plt.title("Spectrogram (dB)")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    plt.close()
    print("[OK] saved", OUT_PNG)

if __name__ == "__main__":
    main()