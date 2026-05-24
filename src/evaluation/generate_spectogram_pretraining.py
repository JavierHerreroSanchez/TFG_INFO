"""
Evalua las piezas generadas mediante metricas simbolicas, espectrales o graficas.

Los resultados producidos aqui sirven para justificar experimentalmente la calidad del modelo en la memoria del TFG.
"""

from __future__ import annotations

import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
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
PROJECT_ROOT = Path(__file__).resolve().parents[2]

GENERATED_DIR = PROJECT_ROOT / "output" / "generation_pretraining_tfg_second"
OUT_DIR = PROJECT_ROOT / "output" / "generation_pretraining_tfg_second" / "midi_spectral_eval"

# Referencias
USE_REFERENCE = True
REFERENCE_MODE = "mixed_random"  # "single_dir" o "mixed_random"
REFERENCE_DIR = PROJECT_ROOT / "data" / "pretraining_raw" / "maestro-v3.0.0"
MAESTRO_DIR = PROJECT_ROOT / "data" / "pretraining_raw" / "maestro-v3.0.0"
ARIA_DIR = PROJECT_ROOT / "data" / "pretraining_raw" / "ariamidi"

RECURSIVE = True
MAX_GENERATED_FILES: Optional[int] = None
MAX_REFERENCE_FILES = 30000
MAESTRO_FRACTION = 0.05
REFERENCE_RANDOM_SEED = 1453

# Pool local de referencias por duración.
REFERENCE_SCORING_SCOPE = "duration_window_all"
LOCAL_REF_DURATION_TOL = 0.30  # +/-30%
MIN_DURATION_MATCHES = 8
NUM_WORKERS: Optional[int] = min(8, max(1, (os.cpu_count() or 4) - 2))
PARALLEL_MIN_FILES = 16

# Parámetros de síntesis y espectrograma.
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


def sanitize_text(parts: List[str]) -> str:

    return " | ".join(p for p in parts if p)


def finite_values(arr: np.ndarray) -> np.ndarray:

    arr = np.asarray(arr, dtype=float)
    return arr[np.isfinite(arr)]


def finite_stats(arr: np.ndarray) -> Tuple[float, float, int]:

    arr = finite_values(arr)
    if arr.size == 0:
        return np.nan, np.nan, 0
    mean = float(arr.mean())
    std = float(arr.std()) if arr.size > 1 else 0.0
    return mean, std, int(arr.size)


def piece_label(score: float) -> str:

    for th, label in QUAL_LABELS:
        if score >= th:
            return label
    return QUAL_LABELS[-1][1]


def choose_reference_files() -> List[Path]:

    if REFERENCE_MODE == "single_dir":
        files = find_midi_files(REFERENCE_DIR, RECURSIVE)
        if MAX_REFERENCE_FILES is not None:
            files = files[:MAX_REFERENCE_FILES]
        print(f"[INFO] REFERENCE_MODE=single_dir | n_reference_files={len(files)}")
        return files

    if REFERENCE_MODE != "mixed_random":
        raise ValueError(f"REFERENCE_MODE desconocido: {REFERENCE_MODE}")

    maestro_files = find_midi_files(MAESTRO_DIR, RECURSIVE)
    aria_files = find_midi_files(ARIA_DIR, RECURSIVE)

    rng = random.Random(REFERENCE_RANDOM_SEED)

    maestro_target = int(round(MAX_REFERENCE_FILES * MAESTRO_FRACTION))
    aria_target = MAX_REFERENCE_FILES - maestro_target

    maestro_take = min(maestro_target, len(maestro_files))
    aria_take = min(aria_target, len(aria_files))

    sampled_maestro = rng.sample(maestro_files, maestro_take) if maestro_take > 0 else []
    sampled_aria = rng.sample(aria_files, aria_take) if aria_take > 0 else []

    combined = sampled_maestro + sampled_aria

    remaining = MAX_REFERENCE_FILES - len(combined)
    if remaining > 0:
        used = set(combined)
        leftovers = [p for p in (maestro_files + aria_files) if p not in used]
        if leftovers:
            extra_take = min(remaining, len(leftovers))
            combined.extend(rng.sample(leftovers, extra_take))

    rng.shuffle(combined)

    print(
        f"[INFO] REFERENCE_MODE=mixed_random | "
        f"maestro_total={len(maestro_files)} aria_total={len(aria_files)} | "
        f"maestro_used={len(sampled_maestro)} aria_used={len(sampled_aria)} "
        f"final_reference_files={len(combined)}"
    )
    return combined


# ============================================================
# MIDI -> AUDIO SIMPLE
# ============================================================
def synthesize_midi_simple(midi_path: Path, sample_rate: int) -> np.ndarray:

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
    """
    Guarda resultados intermedios o finales en disco.

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

    subtitle_parts = [f"sr={sr}", f"n_fft={N_FFT}", f"hop={HOP_LENGTH}", f"mels={N_MELS}", f"dur={duration_s:.1f}s"]
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

    if y.size == 0:
        return {k: np.nan for k in SPECTRAL_FEATURES}

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)

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
# PIPELINE
# ============================================================
def process_midi_file(
    midi_path: Path,
    spec_dir: Optional[Path],
    generate_png: bool,
) -> Dict[str, float | str]:

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

    """

    total = len(files)
    rows = []
    use_parallel = NUM_WORKERS is not None and NUM_WORKERS > 1 and total >= PARALLEL_MIN_FILES

    if use_parallel:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futures = {ex.submit(process_midi_file, path, spec_dir, generate_png): path for path in files}
            for i, fut in enumerate(as_completed(futures), start=1):
                path = futures[fut]
                try:
                    rows.append(fut.result())
                except Exception as e:
                    print(f"[{tag}][WARN] fallo en {path.name}: {e}")
                if i % 10 == 0 or i == total:
                    print(f"[{tag}] procesados {i}/{total}")
    else:
        for i, path in enumerate(files, start=1):
            try:
                rows.append(process_midi_file(path, spec_dir, generate_png=generate_png))
            except Exception as e:
                print(f"[{tag}][WARN] fallo en {path.name}: {e}")
            if i % 10 == 0 or i == total:
                print(f"[{tag}] procesados {i}/{total}")
    return pd.DataFrame(rows)


# ============================================================
# COMPARACIÓN GLOBAL
# ============================================================
def normalized_hist_pair(a: np.ndarray, b: np.ndarray, bins: int | str = "auto") -> Tuple[np.ndarray, np.ndarray]:

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

    if p.size == 0 or q.size == 0:
        return np.nan
    return float(np.minimum(p, q).sum())


def kld_real_to_gen(p_real: np.ndarray, q_gen: np.ndarray) -> float:

    if p_real.size == 0 or q_gen.size == 0:
        return np.nan
    return float(entropy(p_real, q_gen))


def global_distribution_report(ref_df: pd.DataFrame, gen_df: pd.DataFrame, features: List[str]) -> pd.DataFrame:

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
def select_reference_pool(duration_s: float, ref_df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve todas las referencias de duracion compatible, sin limite de tamano."""

    if ref_df.empty or not np.isfinite(duration_s) or "duration_s" not in ref_df.columns:
        return ref_df

    ref_sub = ref_df.loc[np.isfinite(ref_df["duration_s"].to_numpy(dtype=float))].copy()
    if ref_sub.empty:
        return ref_df

    low = duration_s * (1.0 - LOCAL_REF_DURATION_TOL)
    high = duration_s * (1.0 + LOCAL_REF_DURATION_TOL)
    pool = ref_sub[
        (ref_sub["duration_s"] >= low)
        & (ref_sub["duration_s"] <= high)
    ].copy()

    return pool if len(pool) >= MIN_DURATION_MATCHES else ref_sub


def strict_feature_score(x: float, ref_vals: np.ndarray, eps: float = 1e-8) -> float:

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

    used_features = [
        "rms_mean", "rms_std",
        "spectral_centroid_mean", "spectral_centroid_std",
        "spectral_bandwidth_mean", "spectral_bandwidth_std",
        "spectral_rolloff_mean", "spectral_rolloff_std",
        "spectral_flatness_mean", "spectral_flatness_std",
        "spectral_contrast_mean", "spectral_contrast_std",
    ]

    detail_scores = {}
    vals = []
    for feat in used_features:
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
    return {
        "score": safe_float(score),
        "matched_reference_count": int(len(local_ref_df)),
        "details": detail_scores,
    }


def range_acceptance_score(x: float, lo: float, hi: float, softness: float | None = None) -> float:

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

    """

    gen2 = gen_df.copy()
    gen2.insert(0, "set_type", "generated")

    if ref_df is None or ref_df.empty:
        return gen2

    ref2 = ref_df.copy()
    ref2.insert(0, "set_type", "reference")
    return pd.concat([gen2, ref2], ignore_index=True)


def build_compact_evaluation_csv(per_piece_df: pd.DataFrame, global_report: pd.DataFrame) -> pd.DataFrame:
    """
    Construye una estructura auxiliar usada por el resto del flujo.

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

    print("[INFO] Iniciando evaluación espectral rápida de MIDIs generados")
    print(f"[INFO] GENERATED_DIR = {GENERATED_DIR}")
    print(f"[INFO] OUT_DIR       = {OUT_DIR}")
    print(f"[INFO] USE_REFERENCE = {USE_REFERENCE}")

    generated_files = find_midi_files(GENERATED_DIR, RECURSIVE)
    if MAX_GENERATED_FILES is not None:
        generated_files = generated_files[:MAX_GENERATED_FILES]
    if len(generated_files) == 0:
        raise RuntimeError("No se encontró ningún MIDI generado. Comprobar GENERATED_DIR.")

    gen_spec_dir = OUT_DIR / "generated_spectrograms"

    gen_df = build_feature_table(generated_files, gen_spec_dir, "generated", generate_png=True)
    if gen_df.empty:
        raise RuntimeError("No se pudo procesar ningún MIDI generado.")

    ref_df = pd.DataFrame()
    global_report = pd.DataFrame()

    if USE_REFERENCE:
        reference_files = choose_reference_files()
        if len(reference_files) == 0:
            raise RuntimeError("USE_REFERENCE=True pero no se encontraron referencias.")
        ref_df = build_feature_table(reference_files, None, "reference", generate_png=False)
        if ref_df.empty:
            raise RuntimeError("No se pudo procesar ningún MIDI de referencia.")
        global_report = global_distribution_report(ref_df, gen_df, SPECTRAL_FEATURES)

    per_piece_rows = []
    per_piece_details = []

    for _, gen_row in gen_df.iterrows():
        ref_based = {"score": np.nan, "matched_reference_count": 0, "details": {}}
        if USE_REFERENCE and not ref_df.empty:
            reference_pool = select_reference_pool(float(gen_row["duration_s"]), ref_df)
            ref_based = per_piece_reference_based(gen_row, reference_pool)

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
    eval_csv = OUT_DIR / "spectral_evaluation.csv"
    details_json = OUT_DIR / "spectral_per_piece_details.json"
    summary_json = OUT_DIR / "spectral_summary.json"

    for p in [features_csv, eval_csv, details_json, summary_json]:
        if p.exists():
            p.unlink()

    features_all_df.to_csv(features_csv, index=False, encoding="utf-8-sig")
    compact_eval_df.to_csv(eval_csv, index=False, encoding="utf-8-sig")

    with open(details_json, "w", encoding="utf-8") as f:
        json.dump(per_piece_details, f, ensure_ascii=False, indent=2)

    summary = {
        "generated_dir": str(GENERATED_DIR),
        "use_reference": USE_REFERENCE,
        "reference_mode": REFERENCE_MODE if USE_REFERENCE else None,
        "reference_dir": str(REFERENCE_DIR) if (USE_REFERENCE and REFERENCE_MODE == "single_dir") else None,
        "maestro_dir": str(MAESTRO_DIR) if USE_REFERENCE else None,
        "aria_dir": str(ARIA_DIR) if USE_REFERENCE else None,
        "n_generated_files": int(len(gen_df)),
        "n_reference_files": int(len(ref_df)) if USE_REFERENCE else 0,
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
            "reference_based": "1 / (1 + |x-mu|/sigma)",
            "reference_scoring_scope": REFERENCE_SCORING_SCOPE,
            "local_ref_duration_tol": LOCAL_REF_DURATION_TOL,
            "reference_free": "range acceptance with soft penalty outside",
            "weights": {"reference_based": W_REFERENCE_BASED, "reference_free": W_REFERENCE_FREE},
        },
        "top_global_matches": global_report.head(10).to_dict(orient="records") if not global_report.empty else [],
        "per_piece_preview": per_piece_df.head(10).to_dict(orient="records"),
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[OK] Archivos generados:")
    print(f"  - {gen_spec_dir}")
    print(f"  - {features_csv}")
    print(f"  - {eval_csv}")
    print(f"  - {details_json}")
    print(f"  - {summary_json}")


# Ejecucion directa del script.
if __name__ == "__main__":
    main()
