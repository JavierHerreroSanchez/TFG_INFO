from __future__ import annotations

"""
Genera espectrogramas "profesionales" de MIDIs SIN SoundFont.

En lugar de FluidSynth, usa pretty_midi.synthesize(), que sintetiza una señal
simple pero consistente. Esto es útil para:
- depurar generaciones,
- comparar estructura temporal,
- obtener descriptores espectrales básicos,
- guardar espectrogramas de alta calidad.

IMPORTANTE:
- La síntesis NO es realista como un piano con SoundFont.
- Sí es consistente y suficiente para visualización técnica y análisis preliminar.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pretty_midi
from scipy.stats import entropy

# ============================================================
# CONFIGURACIÓN PYCHARM
# ============================================================
GENERATED_DIR = Path(r"/output/generation_v2")
OUT_DIR = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\output\generation_v2\midi_spectrogram_eval_simple")

RECURSIVE = True
MAX_FILES: Optional[int] = None  # None = todas

# Audio / espectrograma
SAMPLE_RATE = 22050
N_FFT = 4096              # más resolución frecuencial
HOP_LENGTH = 256          # mejor resolución temporal
N_MELS = 192              # más detalle vertical que 128
FMIN = 27.5               # A0, piano
FMAX = 8000.0             # recorte útil para visualización de piano
TOP_DB = 80.0             # rango dinámico visible
DPI = 220                 # alta resolución

# Visualización
FIG_W = 15
FIG_H = 6
COLORMAP = "magma"

# CSV de features
FEATURE_COLUMNS = [
    "duration_s",
    "rms_mean", "rms_std",
    "spectral_centroid_mean", "spectral_centroid_std",
    "spectral_bandwidth_mean", "spectral_bandwidth_std",
    "spectral_rolloff_mean", "spectral_rolloff_std",
    "spectral_flatness_mean", "spectral_flatness_std",
    "zcr_mean", "zcr_std",
    "mel_energy_mean", "mel_energy_std",
    "spectral_contrast_mean", "spectral_contrast_std",
]

# ============================================================
# UTILIDADES
# ============================================================
def find_midi_files(root: Path, recursive: bool = True) -> List[Path]:
    pats = ["*.mid", "*.midi"]
    files: List[Path] = []
    for pat in pats:
        files.extend(root.rglob(pat) if recursive else root.glob(pat))
    return sorted({p.resolve() for p in files if p.is_file()})


def safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float(default)
    except Exception:
        return float(default)


# ============================================================
# MIDI -> AUDIO (SIN SOUNDFONT)
# ============================================================
def synthesize_midi_simple(midi_path: Path, sample_rate: int) -> np.ndarray:
    pm = pretty_midi.PrettyMIDI(str(midi_path))

    # Síntesis simple interna de pretty_midi
    y = pm.synthesize(fs=sample_rate)
    y = np.asarray(y, dtype=np.float32)

    # Normalización suave
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 1e-9:
        y = 0.98 * y / peak

    return y


# ============================================================
# ESPECTROGRAMA
# ============================================================
def compute_logmel(y: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=TOP_DB)
    return mel_db


def compute_onset_envelope(y: np.ndarray, sr: int) -> np.ndarray:
    return librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)


def save_professional_spectrogram(
    mel_db: np.ndarray,
    y: np.ndarray,
    sr: int,
    png_path: Path,
    title: str,
    extra_info: Optional[Dict[str, str]] = None,
) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)

    duration_s = len(y) / sr if len(y) else 0.0
    times = librosa.times_like(compute_onset_envelope(y, sr), sr=sr, hop_length=HOP_LENGTH)
    onset_env = compute_onset_envelope(y, sr)

    fig = plt.figure(figsize=(FIG_W, FIG_H), constrained_layout=True, dpi=DPI)
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 1.1])

    ax0 = fig.add_subplot(gs[0])
    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        fmin=FMIN,
        fmax=FMAX,
        cmap=COLORMAP,
        ax=ax0,
    )
    cbar = fig.colorbar(img, ax=ax0, format="%+batch_2.0f dB", pad=0.01)
    cbar.set_label("Nivel (dB)", rotation=90)

    subtitle_parts = [f"sr={sr}", f"n_fft={N_FFT}", f"hop={HOP_LENGTH}", f"mels={N_MELS}", f"dur={duration_s:.1f}s"]
    if extra_info:
        subtitle_parts.extend([f"{k}={v}" for k, v in extra_info.items()])

    ax0.set_title(title, fontsize=14, fontweight="bold", loc="left")
    ax0.text(
        0.0, 1.02,
        " | ".join(subtitle_parts),
        transform=ax0.transAxes,
        fontsize=9,
        alpha=0.85,
        va="bottom",
    )
    ax0.grid(False)

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.plot(times, onset_env, linewidth=1.2)
    ax1.set_ylabel("Onset\nstrength")
    ax1.set_xlabel("Tiempo (s)")
    ax1.set_xlim(0, duration_s if duration_s > 0 else 1.0)
    ax1.grid(alpha=0.25)

    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# DESCRIPTORES ESPECTRALES
# ============================================================
def extract_spectral_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    if y.size == 0:
        return {k: np.nan for k in FEATURE_COLUMNS}

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0
    )

    def ms(arr: np.ndarray) -> Tuple[float, float]:
        arr = np.asarray(arr, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan, np.nan
        return float(arr.mean()), float(arr.std())

    centroid_mean, centroid_std = ms(centroid)
    bandwidth_mean, bandwidth_std = ms(bandwidth)
    rolloff_mean, rolloff_std = ms(rolloff)
    flatness_mean, flatness_std = ms(flatness)
    rms_mean, rms_std = ms(rms)
    zcr_mean, zcr_std = ms(zcr)
    mel_energy_mean, mel_energy_std = ms(mel)
    contrast_mean, contrast_std = ms(contrast)

    duration_s = float(len(y) / sr)

    return {
        "duration_s": duration_s,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "spectral_centroid_mean": centroid_mean,
        "spectral_centroid_std": centroid_std,
        "spectral_bandwidth_mean": bandwidth_mean,
        "spectral_bandwidth_std": bandwidth_std,
        "spectral_rolloff_mean": rolloff_mean,
        "spectral_rolloff_std": rolloff_std,
        "spectral_flatness_mean": flatness_mean,
        "spectral_flatness_std": flatness_std,
        "zcr_mean": zcr_mean,
        "zcr_std": zcr_std,
        "mel_energy_mean": mel_energy_mean,
        "mel_energy_std": mel_energy_std,
        "spectral_contrast_mean": contrast_mean,
        "spectral_contrast_std": contrast_std,
    }


# ============================================================
# PIPELINE
# ============================================================
def process_midi_file(midi_path: Path, spec_dir: Path) -> Dict[str, float | str]:
    stem = midi_path.stem
    png_path = spec_dir / f"{stem}_logmel.png"

    y = synthesize_midi_simple(midi_path, SAMPLE_RATE)
    mel_db = compute_logmel(y, SAMPLE_RATE)
    save_professional_spectrogram(
        mel_db=mel_db,
        y=y,
        sr=SAMPLE_RATE,
        png_path=png_path,
        title=stem,
        extra_info={"synth": "pretty_midi_simple"},
    )

    feats = extract_spectral_features(y, SAMPLE_RATE)
    row: Dict[str, float | str] = {
        "file": str(midi_path),
        "spectrogram_path": str(png_path),
        "synth_mode": "pretty_midi_simple",
    }
    row.update(feats)
    return row


def build_feature_table(files: List[Path], spec_dir: Path, tag: str) -> pd.DataFrame:
    rows = []
    total = len(files)
    for i, path in enumerate(files, start=1):
        try:
            rows.append(process_midi_file(path, spec_dir))
        except Exception as e:
            print(f"[{tag}][WARN] fallo en {path.name}: {e}")
        if i % 10 == 0 or i == total:
            print(f"[{tag}] procesados {i}/{total}")
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================
def main():
    print("[INFO] Iniciando generación de espectrogramas sin SoundFont")
    print(f"[INFO] GENERATED_DIR = {GENERATED_DIR}")
    print(f"[INFO] OUT_DIR       = {OUT_DIR}")

    files = find_midi_files(GENERATED_DIR, RECURSIVE)
    if MAX_FILES is not None:
        files = files[:MAX_FILES]

    if len(files) == 0:
        raise RuntimeError("No se encontró ningún MIDI en GENERATED_DIR.")

    spec_dir = OUT_DIR / "generated_spectrograms"
    features_csv = OUT_DIR / "generated_spectral_features.csv"
    summary_json = OUT_DIR / "spectral_summary.json"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_feature_table(files, spec_dir, "generated")
    if df.empty:
        raise RuntimeError("No se pudo procesar ningún MIDI.")

    for p in [features_csv, summary_json]:
        if p.exists():
            p.unlink()

    df.to_csv(features_csv, index=False, encoding="utf-8-sig")

    summary = {
        "generated_dir": str(GENERATED_DIR),
        "n_files": int(len(df)),
        "synthesis_mode": "pretty_midi_simple",
        "sample_rate": SAMPLE_RATE,
        "spectrogram": {
            "type": "log-mel",
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "n_mels": N_MELS,
            "fmin": FMIN,
            "fmax": FMAX,
            "top_db": TOP_DB,
            "dpi": DPI,
            "colormap": COLORMAP,
        },
        "notes": [
            "No usa SoundFont.",
            "Adecuado para visualización técnica y comparación consistente.",
            "No sustituye una evaluación perceptual realista con síntesis de piano."
        ],
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[OK] Archivos generados:")
    print(f"  - {spec_dir}")
    print(f"  - {features_csv}")
    print(f"  - {summary_json}")


if __name__ == "__main__":
    main()
