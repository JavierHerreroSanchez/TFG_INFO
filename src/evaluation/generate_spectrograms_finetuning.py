"""
Evalua las piezas generadas mediante metricas simbolicas, espectrales o graficas.

Los resultados producidos aqui sirven para justificar experimentalmente la calidad del modelo en la memoria del TFG.
"""

from __future__ import annotations

"""
Evaluación espectral de MIDIs generados tras el fine-tuning.

Las referencias del corpus objetivo se segmentan en ventanas de audio con duración
comparable a las piezas generadas. La puntuación mide cercanía espectral al corpus
de referencia y se complementa con métricas globales OA/KLD y un diagnóstico
separado contra las obras completas.

Salidas principales:
- `spectral_features_all.csv`: generados y ventanas de referencia.
- `spectral_reference_full_piece_features.csv`: obras completas, solo diagnóstico.
- `spectral_reference_window_features.csv`: ventanas usadas para scoring.
- `spectral_evaluation.csv`: puntuación por pieza y bloque global compacto.
- `spectral_per_piece_details.json`: detalle de métricas por muestra.
- `spectral_summary.json`: configuración y resumen de ejecución.
"""

import json
import math
import random
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
# CONFIGURACIÓN
# ============================================================
GENERATED_DIR = Path(r"../../output/generation_finetuning_tfg_first")
OUT_DIR = Path(r"../../output/generation_finetuning_tfg_first/midi_spectral_eval_windows")

# Referencias de fine-tuning: subconjunto musical objetivo.
USE_REFERENCE = True
REFERENCE_MODE = "single_dir"
REFERENCE_DIR = Path(r"../../data/finetuning/finetuning_sonatas_aug")
MAESTRO_DIR = None
ARIA_DIR = None

RECURSIVE = True
MAX_GENERATED_FILES: Optional[int] = None
MAX_REFERENCE_FILES = None

# Pool local de referencias por duración.
LOCAL_REF_POOL_SIZE = 100
LOCAL_REF_DURATION_TOL = 0.30  # +/-30%

# ============================================================
# VENTANAS DE REFERENCIA
# ============================================================
# "match_generated": crea longitudes de ventana según las duraciones de los generados.
# "fixed": usa REFERENCE_WINDOW_SECONDS fijo.
REFERENCE_WINDOW_MODE = "match_generated"  # "match_generated" o "fixed"
REFERENCE_WINDOW_SECONDS = 60.0
REFERENCE_WINDOW_BIN_SECONDS = 5.0
MIN_REFERENCE_WINDOW_SECONDS = 10.0
MAX_REFERENCE_WINDOW_SECONDS = 300.0
REFERENCE_WINDOW_STRIDE_FRACTION = 0.50
MAX_WINDOWS_PER_REFERENCE_PER_SIZE = 16
REFERENCE_WINDOW_RANDOM_SEED = 1453

# Evita ventanas prácticamente silenciosas.
MIN_WINDOW_RMS = 1e-5

# Audio / espectrograma
SAMPLE_RATE = 16000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 96
FMIN = 27.5
FMAX = 6000.0
TOP_DB = 80.0

# Parámetros de visualización de los PNG.
DPI = 180
FIG_W = 14
FIG_H = 5.5
COLORMAP = "magma"

# Pesos de la puntuación global.
W_REFERENCE_BASED = 0.75
W_REFERENCE_FREE = 0.25

QUAL_LABELS = [
    (85.0, "muy plausible"),
    (72.0, "plausible"),
    (58.0, "aceptable con anomalías"),
    (45.0, "débil"),
    (-1.0, "fuera de distribución"),
]

# Features espectrales usadas
SPECTRAL_FEATURES = [
    "duration_s",
    "rms_mean", "rms_std",
    "spectral_centroid_mean", "spectral_centroid_std",
    "spectral_bandwidth_mean", "spectral_bandwidth_std",
    "spectral_rolloff_mean", "spectral_rolloff_std",
    "spectral_flatness_mean", "spectral_flatness_std",
    "spectral_contrast_mean", "spectral_contrast_std",
]

REFERENCE_BASED_FEATURES = [
    "rms_mean", "rms_std",
    "spectral_centroid_mean", "spectral_centroid_std",
    "spectral_bandwidth_mean", "spectral_bandwidth_std",
    "spectral_rolloff_mean", "spectral_rolloff_std",
    "spectral_flatness_mean", "spectral_flatness_std",
    "spectral_contrast_mean", "spectral_contrast_std",
]

PER_PIECE_COLUMNS = [
    "file",
    "spectrogram_path",
    "duration_s",
    "spectral_reference_based_score",
    "spectral_reference_free_score",
    "spectral_global_score",
    "spectral_qualitative_label",
    "spectral_reference_matched_count",
    "strengths",
    "issues",
]

GLOBAL_REPORT_COLUMNS = [
    "feature",
    "oa",
    "kld_real_to_gen",
    "real_mean",
    "gen_mean",
    "real_std",
    "gen_std",
    "n_real",
    "n_gen",
]


# ============================================================
# UTILIDADES
# ============================================================
def find_midi_files(root: Path, recursive: bool = True) -> List[Path]:
    """
    Implementa la logica de find midi files dentro del pipeline del TFG.

    Parametros principales: root, recursive.
    """

    pats = ["*.mid", "*.midi"]
    files: List[Path] = []
    for pat in pats:
        files.extend(root.rglob(pat) if recursive else root.glob(pat))
    return sorted({p.resolve() for p in files if p.is_file()})


def safe_float(x, default=np.nan) -> float:
    """
    Implementa la logica de safe float dentro del pipeline del TFG.

    Parametros principales: x, default.
    """

    try:
        v = float(x)
        return v if math.isfinite(v) else float(default)
    except Exception:
        return float(default)


def sanitize_text(parts: List[str]) -> str:
    """
    Implementa la logica de sanitize text dentro del pipeline del TFG.

    Parametros principales: parts.
    """

    return " | ".join(p for p in parts if p)


def finite_values(arr: np.ndarray) -> np.ndarray:
    """
    Implementa la logica de finite values dentro del pipeline del TFG.

    Parametros principales: arr.
    """

    arr = np.asarray(arr, dtype=float)
    return arr[np.isfinite(arr)]


def finite_stats(arr: np.ndarray) -> Tuple[float, float, int]:
    """
    Implementa la logica de finite stats dentro del pipeline del TFG.

    Parametros principales: arr.
    """

    arr = finite_values(arr)
    if arr.size == 0:
        return np.nan, np.nan, 0
    mean = float(arr.mean())
    std = float(arr.std()) if arr.size > 1 else 0.0
    return mean, std, int(arr.size)


def piece_label(score: float) -> str:
    """
    Implementa la logica de piece label dentro del pipeline del TFG.

    Parametros principales: score.
    """

    for th, label in QUAL_LABELS:
        if score >= th:
            return label
    return QUAL_LABELS[-1][1]


def choose_reference_files() -> List[Path]:
    """
    En finetuning no se mezcla MAESTRO+ARIA.
    Se usa exclusivamente el subconjunto de sonatas aumentadas indicado en REFERENCE_DIR.
    """
    if REFERENCE_MODE != "single_dir":
        raise ValueError(
            f"Para la evaluación espectral de finetuning, REFERENCE_MODE debe ser 'single_dir'. "
            f"Recibido: {REFERENCE_MODE}"
        )

    files = find_midi_files(REFERENCE_DIR, RECURSIVE)
    if MAX_REFERENCE_FILES is not None:
        files = files[:MAX_REFERENCE_FILES]

    print(f"[INFO] REFERENCE_MODE=single_dir | n_reference_files={len(files)}")
    return files


# ============================================================
# MIDI -> AUDIO SIMPLE
# ============================================================
def synthesize_midi_simple(midi_path: Path, sample_rate: int) -> np.ndarray:
    """
    Implementa la logica de synthesize midi simple dentro del pipeline del TFG.

    Parametros principales: midi_path, sample_rate.
    """

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    y = pm.synthesize(fs=sample_rate)
    y = np.asarray(y, dtype=np.float32)
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 1e-9:
        y = 0.98 * y / peak
    return y


# ============================================================
# ESPECTROGRAMA
# ============================================================
def compute_logmel(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Implementa la logica de compute logmel dentro del pipeline del TFG.

    Parametros principales: y, sr.
    """

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
    """
    Implementa la logica de compute onset envelope dentro del pipeline del TFG.

    Parametros principales: y, sr.
    """

    return librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)


def save_professional_spectrogram(
    mel_db: np.ndarray,
    y: np.ndarray,
    sr: int,
    png_path: Path,
    title: str,
    extra_info: Optional[Dict[str, str]] = None,
) -> None:
    """
    Guarda resultados intermedios o finales en disco.

    Parametros principales: mel_db, y, sr, png_path, title, extra_info.
    """

    png_path.parent.mkdir(parents=True, exist_ok=True)

    duration_s = len(y) / sr if len(y) else 0.0
    onset_env = compute_onset_envelope(y, sr)
    times = librosa.times_like(onset_env, sr=sr, hop_length=HOP_LENGTH)

    fig = plt.figure(figsize=(FIG_W, FIG_H), constrained_layout=True, dpi=DPI)
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 1.0])

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
    cbar = fig.colorbar(img, ax=ax0, format="%+2.0f dB", pad=0.01)
    cbar.set_label("Nivel (dB)", rotation=90)

    subtitle_parts = [
        f"sr={sr}",
        f"n_fft={N_FFT}",
        f"hop={HOP_LENGTH}",
        f"mels={N_MELS}",
        f"dur={duration_s:.1f}s",
    ]
    if extra_info:
        subtitle_parts.extend([f"{k}={v}" for k, v in extra_info.items()])

    ax0.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=10)
    ax0.text(
        1.0, 1.005,
        " | ".join(subtitle_parts),
        transform=ax0.transAxes,
        fontsize=8,
        alpha=0.85,
        va="bottom",
        ha="right",
    )
    ax0.grid(False)

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.plot(times, onset_env, linewidth=1.1)
    ax1.set_ylabel("Onset")
    ax1.set_xlabel("Tiempo (s)")
    ax1.set_xlim(0, duration_s if duration_s > 0 else 1.0)
    ax1.grid(alpha=0.25)

    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# FEATURES ESPECTRALES
# ============================================================
def extract_spectral_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Implementa la logica de extract spectral features dentro del pipeline del TFG.

    Parametros principales: y, sr.
    """

    if y.size == 0:
        return {k: np.nan for k in SPECTRAL_FEATURES}

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)

    def ms(arr: np.ndarray) -> Tuple[float, float]:
        """
        Implementa la logica de ms dentro del pipeline del TFG.

        Parametros principales: arr.
        """

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
        "spectral_contrast_mean": contrast_mean,
        "spectral_contrast_std": contrast_std,
    }


# ============================================================
# PIPELINE DE PIEZAS COMPLETAS
# ============================================================
def process_midi_file(
    midi_path: Path,
    spec_dir: Optional[Path],
    generate_png: bool,
) -> Dict[str, float | str]:
    """
    Implementa la logica de process midi file dentro del pipeline del TFG.

    Parametros principales: midi_path, spec_dir, generate_png.
    """

    stem = midi_path.stem
    png_path = spec_dir / f"{stem}_logmel.png" if (generate_png and spec_dir is not None) else None

    y = synthesize_midi_simple(midi_path, SAMPLE_RATE)

    if generate_png and png_path is not None:
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
        "spectrogram_path": str(png_path) if png_path is not None else "",
        "synth_mode": "pretty_midi_simple",
    }
    row.update(feats)
    return row


def build_feature_table(
    files: List[Path],
    spec_dir: Optional[Path],
    tag: str,
    generate_png: bool,
) -> pd.DataFrame:
    """
    Construye una estructura auxiliar usada por el resto del flujo.

    Parametros principales: files, spec_dir, tag, generate_png.
    """

    rows = []
    total = len(files)
    for i, path in enumerate(files, start=1):
        try:
            rows.append(process_midi_file(path, spec_dir, generate_png=generate_png))
        except Exception as e:
            print(f"[{tag}][WARN] fallo en {path.name}: {e}")
        if i % 10 == 0 or i == total:
            print(f"[{tag}] procesados {i}/{total}")
    return pd.DataFrame(rows)


# ============================================================
# PIPELINE DE VENTANAS DE REFERENCIA
# ============================================================
def generated_duration_bins(gen_df: pd.DataFrame) -> List[float]:
    """
    Implementa la logica de generated duration bins dentro del pipeline del TFG.

    Parametros principales: gen_df.
    """

    vals = finite_values(gen_df["duration_s"].to_numpy(dtype=float))
    vals = vals[(vals >= MIN_REFERENCE_WINDOW_SECONDS) & (vals <= MAX_REFERENCE_WINDOW_SECONDS)]
    if vals.size == 0:
        return [REFERENCE_WINDOW_SECONDS]

    bins = sorted({
        float(np.clip(
            round(float(v) / REFERENCE_WINDOW_BIN_SECONDS) * REFERENCE_WINDOW_BIN_SECONDS,
            MIN_REFERENCE_WINDOW_SECONDS,
            MAX_REFERENCE_WINDOW_SECONDS,
        ))
        for v in vals
    })
    return bins if bins else [REFERENCE_WINDOW_SECONDS]


def iter_audio_windows(y: np.ndarray, sr: int, window_seconds: float) -> List[Tuple[int, int]]:
    """
    Implementa la logica de iter audio windows dentro del pipeline del TFG.

    Parametros principales: y, sr, window_seconds.
    """

    window_samples = int(round(window_seconds * sr))
    if window_samples <= 0 or y.size < window_samples:
        return []

    stride_samples = max(1, int(round(window_samples * REFERENCE_WINDOW_STRIDE_FRACTION)))
    max_start = int(y.size - window_samples)
    starts = list(range(0, max_start + 1, stride_samples))
    if not starts or starts[-1] != max_start:
        starts.append(max_start)

    return [(s, s + window_samples) for s in starts]


def process_reference_windows_for_file(
    midi_path: Path,
    window_lengths_seconds: List[float],
    rng: random.Random,
) -> List[Dict[str, float | str]]:
    """
    Implementa la logica de process reference windows for file dentro del pipeline del TFG.

    Parametros principales: midi_path, window_lengths_seconds, rng.
    """

    rows: List[Dict[str, float | str]] = []

    y = synthesize_midi_simple(midi_path, SAMPLE_RATE)
    if y.size == 0:
        return rows

    for window_seconds in window_lengths_seconds:
        windows = iter_audio_windows(y, SAMPLE_RATE, window_seconds)
        if not windows:
            continue

        if len(windows) > MAX_WINDOWS_PER_REFERENCE_PER_SIZE:
            windows = sorted(rng.sample(windows, MAX_WINDOWS_PER_REFERENCE_PER_SIZE))

        for start_sample, end_sample in windows:
            y_win = y[start_sample:end_sample]
            if y_win.size == 0:
                continue

            rms = float(np.sqrt(np.mean(np.square(y_win)))) if y_win.size else 0.0
            if rms < MIN_WINDOW_RMS:
                continue

            feats = extract_spectral_features(y_win, SAMPLE_RATE)
            row: Dict[str, float | str] = {
                "file": str(midi_path),
                "spectrogram_path": "",
                "synth_mode": "pretty_midi_simple_window",
                "source_file": str(midi_path),
                "window_start_s": float(start_sample / SAMPLE_RATE),
                "window_end_s": float(end_sample / SAMPLE_RATE),
                "window_target_s": float(window_seconds),
            }
            row.update(feats)
            rows.append(row)

    return rows


def build_reference_window_table(
    files: List[Path],
    window_lengths_seconds: List[float],
    tag: str = "reference_windows",
) -> pd.DataFrame:
    """
    Construye una estructura auxiliar usada por el resto del flujo.

    Parametros principales: files, window_lengths_seconds, tag.
    """

    rows: List[Dict[str, float | str]] = []
    total = len(files)
    rng = random.Random(REFERENCE_WINDOW_RANDOM_SEED)

    for i, path in enumerate(files, start=1):
        try:
            rows.extend(process_reference_windows_for_file(path, window_lengths_seconds, rng))
        except Exception as e:
            print(f"[{tag}][WARN] fallo en {path.name}: {e}")

        if i % 10 == 0 or i == total:
            print(f"[{tag}] procesados {i}/{total} | ventanas={len(rows)}")

    return pd.DataFrame(rows)


# ============================================================
# COMPARACIÓN GLOBAL
# ============================================================
def normalized_hist_pair(a: np.ndarray, b: np.ndarray, bins: int | str = "auto") -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementa la logica de normalized hist pair dentro del pipeline del TFG.

    Parametros principales: a, b, bins.
    """

    a = finite_values(a)
    b = finite_values(b)
    if a.size == 0 or b.size == 0:
        return np.asarray([]), np.asarray([])

    lo = min(np.min(a), np.min(b))
    hi = max(np.max(a), np.max(b))
    if lo == hi:
        lo -= 0.5
        hi += 0.5

    if bins == "auto":
        nb = int(np.clip(np.sqrt(a.size + b.size), 8, 40))
    else:
        nb = int(bins)

    edges = np.linspace(lo, hi, nb + 1)
    ha, _ = np.histogram(a, bins=edges, density=False)
    hb, _ = np.histogram(b, bins=edges, density=False)

    ha = ha.astype(float) + 1e-12
    hb = hb.astype(float) + 1e-12
    ha /= ha.sum()
    hb /= hb.sum()
    return ha, hb


def overlap_area(p: np.ndarray, q: np.ndarray) -> float:
    """
    Implementa la logica de overlap area dentro del pipeline del TFG.

    Parametros principales: p, q.
    """

    if p.size == 0 or q.size == 0:
        return np.nan
    return float(np.minimum(p, q).sum())


def kld_real_to_gen(p_real: np.ndarray, q_gen: np.ndarray) -> float:
    """
    Implementa la logica de kld real to gen dentro del pipeline del TFG.

    Parametros principales: p_real, q_gen.
    """

    if p_real.size == 0 or q_gen.size == 0:
        return np.nan
    return float(entropy(p_real, q_gen))


def global_distribution_report(ref_df: pd.DataFrame, gen_df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Implementa la logica de global distribution report dentro del pipeline del TFG.

    Parametros principales: ref_df, gen_df, features.
    """

    rows = []
    for feat in features:
        if feat not in ref_df.columns or feat not in gen_df.columns:
            continue
        ref_vals = ref_df[feat].to_numpy(dtype=float)
        gen_vals = gen_df[feat].to_numpy(dtype=float)
        p, q = normalized_hist_pair(ref_vals, gen_vals)
        real_mean, real_std, n_real = finite_stats(ref_vals)
        gen_mean, gen_std, n_gen = finite_stats(gen_vals)
        rows.append({
            "feature": feat,
            "oa": overlap_area(p, q),
            "kld_real_to_gen": kld_real_to_gen(p, q),
            "real_mean": safe_float(real_mean),
            "gen_mean": safe_float(gen_mean),
            "real_std": safe_float(real_std),
            "gen_std": safe_float(gen_std),
            "n_real": n_real,
            "n_gen": n_gen,
        })
    out = pd.DataFrame(rows).reindex(columns=GLOBAL_REPORT_COLUMNS)
    if not out.empty:
        out = out.sort_values(["oa", "kld_real_to_gen"], ascending=[False, True]).reset_index(drop=True)
    return out


# ============================================================
# SCORE POR PIEZA
# ============================================================
def select_local_reference_pool(duration_s: float, ref_df: pd.DataFrame) -> pd.DataFrame:
    """
    Implementa la logica de select local reference pool dentro del pipeline del TFG.

    Parametros principales: duration_s, ref_df.
    """

    if ref_df.empty or not np.isfinite(duration_s):
        return ref_df.head(LOCAL_REF_POOL_SIZE)

    d = ref_df["duration_s"].to_numpy(dtype=float)
    mask = np.isfinite(d)
    ref_sub = ref_df.loc[mask].copy()
    if ref_sub.empty:
        return ref_df.head(LOCAL_REF_POOL_SIZE)

    low = duration_s * (1.0 - LOCAL_REF_DURATION_TOL)
    high = duration_s * (1.0 + LOCAL_REF_DURATION_TOL)
    pool = ref_sub[(ref_sub["duration_s"] >= low) & (ref_sub["duration_s"] <= high)].copy()

    if len(pool) < min(8, LOCAL_REF_POOL_SIZE):
        ref_sub["_dist"] = np.abs(ref_sub["duration_s"] - duration_s)
        pool = ref_sub.sort_values("_dist").head(LOCAL_REF_POOL_SIZE).drop(columns=["_dist"], errors="ignore")
    else:
        pool["_dist"] = np.abs(pool["duration_s"] - duration_s)
        pool = pool.sort_values("_dist").head(LOCAL_REF_POOL_SIZE).drop(columns=["_dist"], errors="ignore")

    return pool


def strict_feature_score(x: float, ref_vals: np.ndarray, eps: float = 1e-8) -> float:
    """
    Implementa la logica de strict feature score dentro del pipeline del TFG.

    Parametros principales: x, ref_vals, eps.
    """

    if not np.isfinite(x):
        return np.nan
    ref_vals = finite_values(ref_vals)
    if ref_vals.size == 0:
        return np.nan

    mu = float(ref_vals.mean())
    sigma = float(ref_vals.std()) if ref_vals.size > 1 else 0.0
    sigma = max(sigma, eps)
    z = abs(float(x) - mu) / sigma
    z = min(z, 5.0)
    return float(1.0 / (1.0 + z))


def per_piece_reference_based(gen_row: pd.Series, local_ref_df: pd.DataFrame) -> Dict:
    """
    Implementa la logica de per piece reference based dentro del pipeline del TFG.

    Parametros principales: gen_row, local_ref_df.
    """

    detail_scores = {}
    vals = []
    for feat in REFERENCE_BASED_FEATURES:
        if feat not in local_ref_df.columns:
            continue
        ref_vals = finite_values(local_ref_df[feat].to_numpy(dtype=float))
        if ref_vals.size == 0:
            continue
        x = safe_float(gen_row.get(feat, np.nan))
        s = strict_feature_score(x, ref_vals)
        detail_scores[feat] = {
            "x": x,
            "mu_ref": float(ref_vals.mean()),
            "sigma_ref": float(ref_vals.std()) if ref_vals.size > 1 else 0.0,
            "score": s,
        }
        if np.isfinite(s):
            vals.append(s)

    score = float(np.nanmean(vals) * 100.0) if vals else np.nan

    local_durations = finite_values(local_ref_df["duration_s"].to_numpy(dtype=float)) if "duration_s" in local_ref_df.columns else np.asarray([])

    return {
        "score": safe_float(score),
        "matched_reference_count": int(len(local_ref_df)),
        "local_reference_duration_s": {
            "mean": safe_float(local_durations.mean()) if local_durations.size else np.nan,
            "min": safe_float(local_durations.min()) if local_durations.size else np.nan,
            "max": safe_float(local_durations.max()) if local_durations.size else np.nan,
        },
        "details": detail_scores,
    }


def range_acceptance_score(x: float, lo: float, hi: float, softness: float | None = None) -> float:
    """
    Implementa la logica de range acceptance score dentro del pipeline del TFG.

    Parametros principales: x, lo, hi, softness.
    """

    if not np.isfinite(x):
        return np.nan
    width = max(hi - lo, 1e-6)
    scale = softness if softness is not None else width / 4.0
    scale = max(scale, 1e-6)
    if lo <= x <= hi:
        return 1.0
    delta = (lo - x) if x < lo else (x - hi)
    return float(1.0 / (1.0 + delta / scale))


def per_piece_reference_free(gen_row: pd.Series) -> Dict:
    """
    Implementa la logica de per piece reference free dentro del pipeline del TFG.

    Parametros principales: gen_row.
    """

    checks = {
        "rms_mean": range_acceptance_score(safe_float(gen_row.get("rms_mean", np.nan)), 0.02, 0.22),
        "spectral_centroid_mean": range_acceptance_score(safe_float(gen_row.get("spectral_centroid_mean", np.nan)), 250.0, 2200.0),
        "spectral_bandwidth_mean": range_acceptance_score(safe_float(gen_row.get("spectral_bandwidth_mean", np.nan)), 350.0, 2600.0),
        "spectral_rolloff_mean": range_acceptance_score(safe_float(gen_row.get("spectral_rolloff_mean", np.nan)), 700.0, 5000.0),
        "spectral_flatness_mean": range_acceptance_score(safe_float(gen_row.get("spectral_flatness_mean", np.nan)), 0.0005, 0.08),
        "spectral_contrast_mean": range_acceptance_score(safe_float(gen_row.get("spectral_contrast_mean", np.nan)), 8.0, 28.0),
    }
    vals = [v for v in checks.values() if np.isfinite(v)]
    score = float(np.nanmean(vals) * 100.0) if vals else np.nan
    return {"score": safe_float(score), "details": checks}


def describe_piece(ref_based: Dict, ref_free: Dict) -> Tuple[List[str], List[str]]:
    """
    Implementa la logica de describe piece dentro del pipeline del TFG.

    Parametros principales: ref_based, ref_free.
    """

    strengths: List[str] = []
    issues: List[str] = []

    details_rb = ref_based.get("details", {})
    details_rf = ref_free.get("details", {})

    for feat in ["spectral_centroid_mean", "spectral_bandwidth_mean", "spectral_flatness_mean", "spectral_contrast_mean"]:
        s = details_rb.get(feat, {}).get("score", np.nan)
        if np.isfinite(s):
            if s >= 0.80:
                strengths.append(f"{feat} cercano a referencia")
            elif s <= 0.30:
                issues.append(f"{feat} alejado de referencia")

    for feat in ["rms_mean", "spectral_flatness_mean", "spectral_contrast_mean"]:
        s = safe_float(details_rf.get(feat, np.nan))
        if np.isfinite(s):
            if s >= 0.90:
                strengths.append(f"{feat} sólido")
            elif s <= 0.35:
                issues.append(f"{feat} problemático")

    strengths = list(dict.fromkeys(strengths))[:3]
    issues = list(dict.fromkeys(issues))[:3]
    return strengths, issues


# ============================================================
# UNIFICACIÓN DE CSVs
# ============================================================
def build_features_all_csv(gen_df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye una estructura auxiliar usada por el resto del flujo.

    Parametros principales: gen_df, ref_df.
    """

    gen2 = gen_df.copy()
    gen2.insert(0, "set_type", "generated")

    if ref_df is None or ref_df.empty:
        return gen2

    ref2 = ref_df.copy()
    ref2.insert(0, "set_type", "reference_window")
    return pd.concat([gen2, ref2], ignore_index=True)


def build_compact_evaluation_csv(per_piece_df: pd.DataFrame, global_report: pd.DataFrame) -> pd.DataFrame:
    """
    Construye una estructura auxiliar usada por el resto del flujo.

    Parametros principales: per_piece_df, global_report.
    """

    rows = []

    for _, row in per_piece_df.iterrows():
        d = {c: "" for c in PER_PIECE_COLUMNS + ["section"]}
        d["section"] = "per_piece"
        for c in PER_PIECE_COLUMNS:
            d[c] = row.get(c, "")
        rows.append(d)

    if global_report is not None and not global_report.empty:
        for _, row in global_report.iterrows():
            d = {c: "" for c in PER_PIECE_COLUMNS + ["section"]}
            d["section"] = "global_metric"
            d["file"] = row.get("feature", "")
            d["strengths"] = json.dumps({
                "oa": safe_float(row.get("oa", np.nan)),
                "kld_real_to_gen": safe_float(row.get("kld_real_to_gen", np.nan)),
                "real_mean": safe_float(row.get("real_mean", np.nan)),
                "gen_mean": safe_float(row.get("gen_mean", np.nan)),
                "real_std": safe_float(row.get("real_std", np.nan)),
                "gen_std": safe_float(row.get("gen_std", np.nan)),
                "n_real": int(row.get("n_real", 0)),
                "n_gen": int(row.get("n_gen", 0)),
            }, ensure_ascii=False)
            rows.append(d)

    out = pd.DataFrame(rows)
    wanted = ["section"] + PER_PIECE_COLUMNS
    return out.reindex(columns=wanted)


# ============================================================
# MAIN
# ============================================================
def main():
    """Punto de entrada del script cuando se ejecuta desde consola."""

    print("[INFO] Iniciando evaluación espectral de MIDIs generados con ventanas de referencia")
    print(f"[INFO] GENERATED_DIR = {GENERATED_DIR}")
    print(f"[INFO] OUT_DIR       = {OUT_DIR}")
    print(f"[INFO] USE_REFERENCE = {USE_REFERENCE}")

    generated_files = find_midi_files(GENERATED_DIR, RECURSIVE)
    if MAX_GENERATED_FILES is not None:
        generated_files = generated_files[:MAX_GENERATED_FILES]

    print(f"[INFO] generated files found: {len(generated_files)}")

    if len(generated_files) == 0:
        raise RuntimeError("No se encontró ningún MIDI generado. Comprobar GENERATED_DIR.")

    gen_spec_dir = OUT_DIR / "generated_spectrograms"

    # 1) Features de generados completos + PNG
    gen_df = build_feature_table(generated_files, gen_spec_dir, "generated", generate_png=True)
    if gen_df.empty:
        raise RuntimeError("No se pudo procesar ningún MIDI generado.")

    ref_df = pd.DataFrame()
    ref_full_df = pd.DataFrame()
    global_report = pd.DataFrame()
    global_report_full_piece = pd.DataFrame()
    window_lengths = []

    # 2) Referencias completas, y luego ventanas usadas para scoring
    if USE_REFERENCE:
        reference_files = choose_reference_files()
        if len(reference_files) == 0:
            raise RuntimeError("USE_REFERENCE=True pero no se encontraron referencias.")

        # Se guarda solo como diagnóstico, no se usa para scoring.
        ref_full_df = build_feature_table(reference_files, None, "reference_full_piece", generate_png=False)
        if ref_full_df.empty:
            raise RuntimeError("No se pudo procesar ningún MIDI de referencia completo.")

        if REFERENCE_WINDOW_MODE == "match_generated":
            window_lengths = generated_duration_bins(gen_df)
        elif REFERENCE_WINDOW_MODE == "fixed":
            window_lengths = [REFERENCE_WINDOW_SECONDS]
        else:
            raise ValueError(f"REFERENCE_WINDOW_MODE desconocido: {REFERENCE_WINDOW_MODE}")

        print(f"[INFO] reference_window_lengths_seconds={window_lengths}")

        ref_df = build_reference_window_table(reference_files, window_lengths, tag="reference_windows")
        if ref_df.empty:
            raise RuntimeError("No se pudo construir ninguna ventana de referencia.")

        # Principal: distribuciones generadas vs ventanas de referencia.
        global_report = global_distribution_report(ref_df, gen_df, SPECTRAL_FEATURES)
        # Diagnóstico: generadas vs piezas completas de referencia.
        global_report_full_piece = global_distribution_report(ref_full_df, gen_df, SPECTRAL_FEATURES)

    # 3) Score por pieza
    per_piece_rows = []
    per_piece_details = []

    for _, gen_row in gen_df.iterrows():
        ref_based = {"score": np.nan, "matched_reference_count": 0, "details": {}}
        if USE_REFERENCE and not ref_df.empty:
            local_pool = select_local_reference_pool(float(gen_row["duration_s"]), ref_df)
            ref_based = per_piece_reference_based(gen_row, local_pool)

        ref_free = per_piece_reference_free(gen_row)

        if USE_REFERENCE and np.isfinite(ref_based["score"]):
            overall = W_REFERENCE_BASED * float(ref_based["score"]) + W_REFERENCE_FREE * float(ref_free["score"])
        else:
            overall = float(ref_free["score"])

        label = piece_label(overall)
        strengths, issues = describe_piece(ref_based, ref_free)

        per_piece_rows.append({
            "file": gen_row["file"],
            "spectrogram_path": gen_row["spectrogram_path"],
            "duration_s": float(gen_row["duration_s"]),
            "spectral_reference_based_score": safe_float(ref_based["score"]),
            "spectral_reference_free_score": float(ref_free["score"]),
            "spectral_global_score": float(overall),
            "spectral_qualitative_label": label,
            "spectral_reference_matched_count": int(ref_based["matched_reference_count"]),
            "strengths": sanitize_text(strengths),
            "issues": sanitize_text(issues),
        })

        per_piece_details.append({
            "file": gen_row["file"],
            "spectrogram_path": gen_row["spectrogram_path"],
            "spectral_reference_based": ref_based,
            "spectral_reference_free": ref_free,
            "spectral_global_score": float(overall),
            "spectral_qualitative_label": label,
            "strengths": strengths,
            "issues": issues,
        })

    per_piece_df = pd.DataFrame(per_piece_rows).reindex(columns=PER_PIECE_COLUMNS)
    per_piece_df = per_piece_df.sort_values("spectral_global_score", ascending=False).reset_index(drop=True)

    features_all_df = build_features_all_csv(gen_df, ref_df)
    compact_eval_df = build_compact_evaluation_csv(per_piece_df, global_report)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    features_csv = OUT_DIR / "spectral_features_all.csv"
    reference_windows_csv = OUT_DIR / "spectral_reference_window_features.csv"
    reference_full_piece_csv = OUT_DIR / "spectral_reference_full_piece_features.csv"
    global_windows_csv = OUT_DIR / "spectral_global_report_windows.csv"
    global_full_piece_csv = OUT_DIR / "spectral_global_report_full_piece_diagnostic.csv"
    eval_csv = OUT_DIR / "spectral_evaluation.csv"
    details_json = OUT_DIR / "spectral_per_piece_details.json"
    summary_json = OUT_DIR / "spectral_summary.json"

    for p in [
        features_csv,
        reference_windows_csv,
        reference_full_piece_csv,
        global_windows_csv,
        global_full_piece_csv,
        eval_csv,
        details_json,
        summary_json,
    ]:
        if p.exists():
            p.unlink()

    features_all_df.to_csv(features_csv, index=False, encoding="utf-8-sig")
    if not ref_df.empty:
        ref_df.to_csv(reference_windows_csv, index=False, encoding="utf-8-sig")
    if not ref_full_df.empty:
        ref_full_df.to_csv(reference_full_piece_csv, index=False, encoding="utf-8-sig")
    if not global_report.empty:
        global_report.to_csv(global_windows_csv, index=False, encoding="utf-8-sig")
    if not global_report_full_piece.empty:
        global_report_full_piece.to_csv(global_full_piece_csv, index=False, encoding="utf-8-sig")

    compact_eval_df.to_csv(eval_csv, index=False, encoding="utf-8-sig")

    with open(details_json, "w", encoding="utf-8") as f:
        json.dump(per_piece_details, f, ensure_ascii=False, indent=2)

    summary = {
        "generated_dir": str(GENERATED_DIR),
        "use_reference": USE_REFERENCE,
        "reference_mode": REFERENCE_MODE if USE_REFERENCE else None,
        "reference_dir": str(REFERENCE_DIR) if (USE_REFERENCE and REFERENCE_MODE == "single_dir") else None,
        "finetuning_reference_dir": str(REFERENCE_DIR) if USE_REFERENCE else None,
        "n_generated_files": int(len(gen_df)),
        "n_reference_full_piece_files": int(len(ref_full_df)) if USE_REFERENCE else 0,
        "n_reference_windows": int(len(ref_df)) if USE_REFERENCE else 0,
        "reference_window_mode": REFERENCE_WINDOW_MODE,
        "reference_window_lengths_seconds": window_lengths,
        "reference_window_stride_fraction": REFERENCE_WINDOW_STRIDE_FRACTION,
        "local_ref_pool_size": LOCAL_REF_POOL_SIZE,
        "local_ref_duration_tol": LOCAL_REF_DURATION_TOL,
        "synthesis_mode": "pretty_midi_simple",
        "spectrogram": {
            "generated_pngs_only": True,
            "type": "log-mel",
            "sample_rate": SAMPLE_RATE,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "n_mels": N_MELS,
            "fmin": FMIN,
            "fmax": FMAX,
            "top_db": TOP_DB,
            "dpi": DPI,
            "colormap": COLORMAP,
        },
        "features_used": SPECTRAL_FEATURES,
        "scoring": {
            "reference_based": "1 / (1 + min(|x-mu|/sigma, 5)); references are duration-matched audio windows",
            "reference_free": "range acceptance with soft penalty outside",
            "weights": {"reference_based": W_REFERENCE_BASED, "reference_free": W_REFERENCE_FREE},
        },
        "top_global_matches_windows": global_report.head(10).to_dict(orient="records") if not global_report.empty else [],
        "top_global_matches_full_piece_diagnostic": global_report_full_piece.head(10).to_dict(orient="records") if not global_report_full_piece.empty else [],
        "per_piece_preview": per_piece_df.head(10).to_dict(orient="records"),
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[OK] Archivos generados:")
    print(f"  - {gen_spec_dir}")
    print(f"  - {features_csv}")
    print(f"  - {reference_windows_csv}")
    print(f"  - {reference_full_piece_csv}")
    print(f"  - {global_windows_csv}")
    print(f"  - {global_full_piece_csv}")
    print(f"  - {eval_csv}")
    print(f"  - {details_json}")
    print(f"  - {summary_json}")


# Ejecucion directa del script.
if __name__ == "__main__":
    main()
