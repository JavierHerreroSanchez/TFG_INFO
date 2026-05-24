"""
Evalua las piezas generadas mediante metricas simbolicas, espectrales o graficas.

Los resultados producidos aqui sirven para justificar experimentalmente la calidad del modelo en la memoria del TFG.
"""

from __future__ import annotations

"""
Evaluación simbólica de MIDIs generados tras el fine-tuning.

Las piezas generadas tienen una duración mucho menor que muchas obras completas
del corpus de referencia. Para evitar que la duración domine la comparación, las
referencias se recortan en ventanas de tamaño comparable a las muestras generadas.
La puntuación mide cercanía estadística al corpus de referencia y se interpreta
como una aproximación operativa a la plausibilidad musical.

Salidas principales:
- `reference_features.csv`: features de ventanas de referencia usadas para scoring.
- `reference_full_piece_features.csv`: features de obras completas, solo diagnóstico.
- `per_piece_evaluation.csv`: puntuación por pieza generada.
- `summary_compact.csv`: resumen de configuración y métricas globales.
- `per_piece_details.json`: detalle de métricas y ventanas locales usadas.
"""

import copy
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import muspy
from scipy.stats import entropy

# ============================================================
# CONFIGURACIÓN
# ============================================================
GENERATED_DIR = Path(r"../../output/generation_finetuning_tfg_first")
OUT_DIR = Path(r"../../output/generation_finetuning_tfg_first/midi_eval_windows")

# Para fine-tuning se usa un único conjunto de referencia del corpus objetivo.
REFERENCE_MODE = "single_dir"
REFERENCE_DIR = Path(r"../../data/finetuning/finetuning_sonatas_aug")

RECURSIVE = True
MAX_REFERENCE_FILES = None

LOCAL_REF_POOL_SIZE = 200
LOCAL_REF_DURATION_TOL = 0.30

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

MUSPY_RESOLUTION = 24
DEFAULT_MEASURE_RESOLUTION = 96

# ============================================================
# CONFIGURACIÓN DE VENTANAS DE REFERENCIA
# ============================================================
# En vez de comparar cada generado con sonatas completas, creamos ventanas de referencia
# con duraciones parecidas a las generadas.
# "match_generated": toma las duraciones de los generados y crea ventanas redondeadas.
# "fixed": usa REFERENCE_WINDOW_BEATS.
REFERENCE_WINDOW_MODE = "match_generated"  # "match_generated" o "fixed"
REFERENCE_WINDOW_BEATS = 160.0
REFERENCE_WINDOW_BIN_BEATS = 16.0
REFERENCE_WINDOW_STRIDE_FRACTION = 0.50
MIN_REFERENCE_WINDOW_BEATS = 32.0
MAX_REFERENCE_WINDOW_BEATS = 512.0
MIN_WINDOW_NOTES = 16
MAX_WINDOWS_PER_REFERENCE_PER_SIZE = 16
REFERENCE_WINDOW_RANDOM_SEED = 1453

# Ajustes para que métricas acotadas y discretas no reciban penalizaciones
# artificiales cuando la referencia tiene varianza nula o muy baja.
USE_RANGE_FOR_SCALE_CONSISTENCY = True
ZERO_SIGMA_FALLBACK_SCALE = 1.0

GLOBAL_FEATURES = [
    "pitch_range",
    "n_pitch_classes_used",
    "n_pitches_used",
    "polyphony",
    "polyphony_rate",
    "pitch_class_entropy",
    "pitch_entropy",
    "scale_consistency",
    "empty_beat_rate",
    "empty_measure_rate",
    "groove_consistency",
    "pitch_histogram_entropy_custom",
    "consecutive_pitch_repetition_ratio_custom",
    "mean_velocity_custom",
    "std_velocity_custom",
    "mean_duration_beats_custom",
    "std_duration_beats_custom",
    "n_notes_custom",
    "duration_beats_custom",
]

PER_PIECE_COLUMNS = [
    "file",
    "duration_beats_custom",
    "n_notes_custom",
    "reference_based_score",
    "reference_based_pitch_hist_similarity",
    "reference_based_matched_reference_count",
    "reference_free_score",
    "global_score",
    "qualitative_label",
    "strengths",
    "issues",
]

# ============================================================
# UTILIDADES GENERALES
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
        if x is None:
            return float(default)
        v = float(x)
        if math.isfinite(v):
            return v
        return float(default)
    except Exception:
        return float(default)


def piece_label(score: float) -> str:
    """
    Implementa la logica de piece label dentro del pipeline del TFG.

    Parametros principales: score.
    """

    for th, label in QUAL_LABELS:
        if score >= th:
            return label
    return QUAL_LABELS[-1][1]


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


def choose_reference_files() -> List[Path]:
    """Implementa la logica de choose reference files dentro del pipeline del TFG."""

    if REFERENCE_MODE != "single_dir":
        raise ValueError(f"Para finetuning, REFERENCE_MODE debe ser 'single_dir'. Recibido: {REFERENCE_MODE}")
    files = find_midi_files(REFERENCE_DIR, RECURSIVE)
    if MAX_REFERENCE_FILES is not None:
        files = files[:MAX_REFERENCE_FILES]
    print(f"[INFO] REFERENCE_MODE=single_dir | n_reference_files={len(files)}")
    return files


# ============================================================
# CONVERSION Y FILTRADO MUSPY
# ============================================================
def load_music(path: Path) -> muspy.Music:
    """
    Carga los recursos necesarios para esta fase del pipeline.

    Parametros principales: path.
    """

    music = muspy.read_midi(path)
    if music.resolution is None:
        music.resolution = MUSPY_RESOLUTION
    return music


def keep_only_piano_tracks(music: muspy.Music) -> muspy.Music:
    """
    Implementa la logica de keep only piano tracks dentro del pipeline del TFG.

    Parametros principales: music.
    """

    piano_programs = set(range(0, 8))
    kept = []
    for track in music.tracks:
        program = getattr(track, "program", 0)
        if program in piano_programs:
            kept.append(track)
    music.tracks = kept
    return music


def iter_notes(music: muspy.Music):
    """
    Implementa la logica de iter notes dentro del pipeline del TFG.

    Parametros principales: music.
    """

    for track in music.tracks:
        for note in track.notes:
            yield note


def all_notes_sorted(music: muspy.Music) -> List:
    """
    Implementa la logica de all notes sorted dentro del pipeline del TFG.

    Parametros principales: music.
    """

    notes = list(iter_notes(music))
    notes.sort(key=lambda n: (n.time, n.pitch, getattr(n, "velocity", 64)))
    return notes


# ============================================================
# FEATURES CUSTOM MINIMAS
# ============================================================
def pitch_histogram_entropy_custom(music: muspy.Music) -> float:
    """
    Implementa la logica de pitch histogram entropy custom dentro del pipeline del TFG.

    Parametros principales: music.
    """

    notes = all_notes_sorted(music)
    if not notes:
        return np.nan
    hist = np.zeros(12, dtype=float)
    for n in notes:
        hist[n.pitch % 12] += 1.0
    hist /= hist.sum()
    return float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum())


def consecutive_pitch_repetition_ratio_custom(music: muspy.Music) -> float:
    """
    Implementa la logica de consecutive pitch repetition ratio custom dentro del pipeline del TFG.

    Parametros principales: music.
    """

    notes = all_notes_sorted(music)
    if len(notes) < 2:
        return 0.0
    reps = 0
    total = 0
    prev_pitch = notes[0].pitch
    for n in notes[1:]:
        total += 1
        if n.pitch == prev_pitch:
            reps += 1
        prev_pitch = n.pitch
    return float(reps / total) if total > 0 else 0.0


def note_durations_beats(music: muspy.Music) -> np.ndarray:
    """
    Implementa la logica de note durations beats dentro del pipeline del TFG.

    Parametros principales: music.
    """

    notes = all_notes_sorted(music)
    if not notes:
        return np.asarray([], dtype=float)
    res = music.resolution or MUSPY_RESOLUTION
    return np.asarray([n.duration / res for n in notes], dtype=float)


def note_velocities(music: muspy.Music) -> np.ndarray:
    """
    Implementa la logica de note velocities dentro del pipeline del TFG.

    Parametros principales: music.
    """

    notes = all_notes_sorted(music)
    if not notes:
        return np.asarray([], dtype=float)
    return np.asarray([getattr(n, "velocity", 64) for n in notes], dtype=float)


def duration_beats_custom(music: muspy.Music) -> float:
    """
    Implementa la logica de duration beats custom dentro del pipeline del TFG.

    Parametros principales: music.
    """

    end_time = 0
    for n in all_notes_sorted(music):
        end_time = max(end_time, int(n.time + n.duration))
    res = music.resolution or MUSPY_RESOLUTION
    return float(end_time / res) if res else np.nan


def n_notes_custom(music: muspy.Music) -> float:
    """
    Implementa la logica de n notes custom dentro del pipeline del TFG.

    Parametros principales: music.
    """

    return float(len(all_notes_sorted(music)))


def mean_velocity_custom(music: muspy.Music) -> float:
    """
    Implementa la logica de mean velocity custom dentro del pipeline del TFG.

    Parametros principales: music.
    """

    arr = note_velocities(music)
    return float(np.mean(arr)) if arr.size else np.nan


def std_velocity_custom(music: muspy.Music) -> float:
    """
    Implementa la logica de std velocity custom dentro del pipeline del TFG.

    Parametros principales: music.
    """

    arr = note_velocities(music)
    return float(np.std(arr)) if arr.size else np.nan


def mean_duration_beats_custom(music: muspy.Music) -> float:
    """
    Implementa la logica de mean duration beats custom dentro del pipeline del TFG.

    Parametros principales: music.
    """

    arr = note_durations_beats(music)
    return float(np.mean(arr)) if arr.size else np.nan


def std_duration_beats_custom(music: muspy.Music) -> float:
    """
    Implementa la logica de std duration beats custom dentro del pipeline del TFG.

    Parametros principales: music.
    """

    arr = note_durations_beats(music)
    return float(np.std(arr)) if arr.size else np.nan


def pitch_histogram_12(music: muspy.Music) -> np.ndarray:
    """
    Implementa la logica de pitch histogram 12 dentro del pipeline del TFG.

    Parametros principales: music.
    """

    notes = all_notes_sorted(music)
    hist = np.zeros(12, dtype=float)
    if not notes:
        return hist
    for n in notes:
        hist[n.pitch % 12] += 1.0
    s = hist.sum()
    return hist / s if s > 0 else hist


# ============================================================
# WRAPPERS METRICAS MUSPY
# ============================================================
def muspy_metric_safe(name: str, music: muspy.Music) -> float:
    """
    Implementa la logica de muspy metric safe dentro del pipeline del TFG.

    Parametros principales: name, music.
    """

    fn = getattr(muspy.metrics, name)
    try:
        if name in {"empty_measure_rate", "groove_consistency"}:
            return safe_float(fn(music, measure_resolution=DEFAULT_MEASURE_RESOLUTION))
        return safe_float(fn(music))
    except Exception:
        return np.nan


# ============================================================
# EXTRACCION FEATURES POR PIEZA / VENTANA
# ============================================================
def extract_features_from_music(music: muspy.Music, file_label: str) -> Dict[str, float | str]:
    """
    Implementa la logica de extract features from music dentro del pipeline del TFG.

    Parametros principales: music, file_label.
    """

    row: Dict[str, float | str] = {
        "file": str(file_label),
        "pitch_range": muspy_metric_safe("pitch_range", music),
        "n_pitch_classes_used": muspy_metric_safe("n_pitch_classes_used", music),
        "n_pitches_used": muspy_metric_safe("n_pitches_used", music),
        "polyphony": muspy_metric_safe("polyphony", music),
        "polyphony_rate": muspy_metric_safe("polyphony_rate", music),
        "pitch_class_entropy": muspy_metric_safe("pitch_class_entropy", music),
        "pitch_entropy": muspy_metric_safe("pitch_entropy", music),
        "scale_consistency": muspy_metric_safe("scale_consistency", music),
        "empty_beat_rate": muspy_metric_safe("empty_beat_rate", music),
        "empty_measure_rate": muspy_metric_safe("empty_measure_rate", music),
        "groove_consistency": muspy_metric_safe("groove_consistency", music),
        "pitch_histogram_entropy_custom": pitch_histogram_entropy_custom(music),
        "consecutive_pitch_repetition_ratio_custom": consecutive_pitch_repetition_ratio_custom(music),
        "mean_velocity_custom": mean_velocity_custom(music),
        "std_velocity_custom": std_velocity_custom(music),
        "mean_duration_beats_custom": mean_duration_beats_custom(music),
        "std_duration_beats_custom": std_duration_beats_custom(music),
        "n_notes_custom": n_notes_custom(music),
        "duration_beats_custom": duration_beats_custom(music),
        "_pitch_hist_json": json.dumps(pitch_histogram_12(music).tolist()),
    }
    return row


def extract_features(path: Path) -> Dict[str, float | str]:
    """
    Implementa la logica de extract features dentro del pipeline del TFG.

    Parametros principales: path.
    """

    music = load_music(path)
    music = keep_only_piano_tracks(music)
    return extract_features_from_music(music, str(path))


def build_feature_table(files: List[Path], tag: str) -> pd.DataFrame:
    """
    Construye una estructura auxiliar usada por el resto del flujo.

    Parametros principales: files, tag.
    """

    rows = []
    total = len(files)
    for i, path in enumerate(files, start=1):
        try:
            rows.append(extract_features(path))
        except Exception as e:
            print(f"[{tag}][WARN] fallo en {path.name}: {e}")
        if i % 50 == 0 or i == total:
            print(f"[{tag}] procesados {i}/{total}")
    return pd.DataFrame(rows)


# ============================================================
# CREACION DE VENTANAS DE REFERENCIA
# ============================================================
def music_end_tick(music: muspy.Music) -> int:
    """
    Implementa la logica de music end tick dentro del pipeline del TFG.

    Parametros principales: music.
    """

    end_time = 0
    for n in all_notes_sorted(music):
        end_time = max(end_time, int(n.time + n.duration))
    return end_time


def window_lengths_from_generated(gen_df: pd.DataFrame) -> List[float]:
    """
    Implementa la logica de window lengths from generated dentro del pipeline del TFG.

    Parametros principales: gen_df.
    """

    vals = finite_values(gen_df["duration_beats_custom"].to_numpy(dtype=float))
    vals = vals[(vals >= MIN_REFERENCE_WINDOW_BEATS) & (vals <= MAX_REFERENCE_WINDOW_BEATS)]
    if vals.size == 0:
        return [REFERENCE_WINDOW_BEATS]

    lengths = sorted({
        float(np.clip(
            round(float(v) / REFERENCE_WINDOW_BIN_BEATS) * REFERENCE_WINDOW_BIN_BEATS,
            MIN_REFERENCE_WINDOW_BEATS,
            MAX_REFERENCE_WINDOW_BEATS,
        ))
        for v in vals
    })
    return lengths if lengths else [REFERENCE_WINDOW_BEATS]


def slice_music_window(music: muspy.Music, start_tick: int, end_tick: int) -> muspy.Music:
    """
    Corta una ventana [start_tick, end_tick) y desplaza sus notas a t=0.
    No altera la tokenización ni reinterpreta la música; solo recorta notas MIDI.
    """
    out = muspy.Music(resolution=music.resolution or MUSPY_RESOLUTION)

    # Copia ligera de metadatos si existen. No son críticos para las métricas usadas,
    # pero ayuda a preservar contexto cuando MusPy lo permite.
    for attr in ["tempos", "key_signatures", "time_signatures"]:
        if hasattr(music, attr):
            try:
                setattr(out, attr, copy.deepcopy(getattr(music, attr)))
            except Exception:
                pass

    out.tracks = []
    for track in music.tracks:
        new_track = muspy.Track(
            program=getattr(track, "program", 0),
            is_drum=getattr(track, "is_drum", False),
            name=getattr(track, "name", ""),
        )

        for note in track.notes:
            note_start = int(note.time)
            note_end = int(note.time + note.duration)

            if note_end <= start_tick or note_start >= end_tick:
                continue

            clipped_start = max(note_start, start_tick)
            clipped_end = min(note_end, end_tick)
            clipped_duration = max(1, clipped_end - clipped_start)

            new_track.notes.append(muspy.Note(
                time=int(clipped_start - start_tick),
                pitch=int(note.pitch),
                duration=int(clipped_duration),
                velocity=int(getattr(note, "velocity", 64)),
            ))

        if new_track.notes:
            out.tracks.append(new_track)

    return out


def build_reference_window_table(ref_files: List[Path], window_lengths_beats: List[float]) -> pd.DataFrame:
    """
    Construye una estructura auxiliar usada por el resto del flujo.

    Parametros principales: ref_files, window_lengths_beats.
    """

    rows = []
    total = len(ref_files)
    rng = random.Random(REFERENCE_WINDOW_RANDOM_SEED)

    for i, path in enumerate(ref_files, start=1):
        try:
            music = load_music(path)
            music = keep_only_piano_tracks(music)
            res = music.resolution or MUSPY_RESOLUTION
            end_tick = music_end_tick(music)
            if end_tick <= 0:
                continue

            for window_beats in window_lengths_beats:
                window_ticks = int(round(window_beats * res))
                if window_ticks <= 0 or end_tick < window_ticks:
                    continue

                stride_ticks = max(1, int(round(window_ticks * REFERENCE_WINDOW_STRIDE_FRACTION)))
                max_start = end_tick - window_ticks
                starts = list(range(0, max_start + 1, stride_ticks))
                if not starts or starts[-1] != max_start:
                    starts.append(max_start)

                if len(starts) > MAX_WINDOWS_PER_REFERENCE_PER_SIZE:
                    starts = sorted(rng.sample(starts, MAX_WINDOWS_PER_REFERENCE_PER_SIZE))

                for start_tick in starts:
                    win = slice_music_window(music, start_tick, start_tick + window_ticks)
                    if n_notes_custom(win) < MIN_WINDOW_NOTES:
                        continue
                    row = extract_features_from_music(win, str(path))
                    row["source_file"] = str(path)
                    row["window_start_beats"] = float(start_tick / res)
                    row["window_target_beats"] = float(window_beats)
                    rows.append(row)

        except Exception as e:
            print(f"[reference_windows][WARN] fallo en {path.name}: {e}")

        if i % 10 == 0 or i == total:
            print(f"[reference_windows] procesados {i}/{total} | ventanas={len(rows)}")

    return pd.DataFrame(rows)


# ============================================================
# OA / KLD CON SCIPY + HISTOGRAMAS
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
        nb = int(np.clip(np.sqrt(a.size + b.size), 8, 48))
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
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["oa", "kld_real_to_gen"], ascending=[False, True]).reset_index(drop=True)
    return out


# ============================================================
# SCORE POR PIEZA: reference-based
# ============================================================
def select_local_reference_pool(duration_beats: float, ref_df: pd.DataFrame) -> pd.DataFrame:
    """
    Implementa la logica de select local reference pool dentro del pipeline del TFG.

    Parametros principales: duration_beats, ref_df.
    """

    if ref_df.empty or not np.isfinite(duration_beats):
        return ref_df.head(LOCAL_REF_POOL_SIZE)
    d = ref_df["duration_beats_custom"].to_numpy(dtype=float)
    mask = np.isfinite(d)
    ref_sub = ref_df.loc[mask].copy()
    if ref_sub.empty:
        return ref_df.head(LOCAL_REF_POOL_SIZE)
    low = duration_beats * (1.0 - LOCAL_REF_DURATION_TOL)
    high = duration_beats * (1.0 + LOCAL_REF_DURATION_TOL)
    pool = ref_sub[(ref_sub["duration_beats_custom"] >= low) & (ref_sub["duration_beats_custom"] <= high)].copy()
    if len(pool) < min(8, LOCAL_REF_POOL_SIZE):
        ref_sub["_dist"] = np.abs(ref_sub["duration_beats_custom"] - duration_beats)
        pool = ref_sub.sort_values("_dist").head(LOCAL_REF_POOL_SIZE).drop(columns=["_dist"], errors="ignore")
    else:
        pool["_dist"] = np.abs(pool["duration_beats_custom"] - duration_beats)
        pool = pool.sort_values("_dist").head(LOCAL_REF_POOL_SIZE).drop(columns=["_dist"], errors="ignore")
    return pool


def pitch_hist_similarity_from_json(hist_json_a: str, hist_json_b: str) -> float:
    """
    Implementa la logica de pitch hist similarity from json dentro del pipeline del TFG.

    Parametros principales: hist_json_a, hist_json_b.
    """

    a = np.asarray(json.loads(hist_json_a), dtype=float)
    b = np.asarray(json.loads(hist_json_b), dtype=float)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    a = a / a.sum()
    b = b / b.sum()
    return float(np.minimum(a, b).sum())


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

    # Si sigma es cero o casi cero, no se divide por eps: cualquier diferencia
    # produciría un z-score enorme. Una escala mínima de 1 unidad es más estable
    # para contadores discretos como n_pitch_classes_used.
    if sigma < eps:
        sigma = ZERO_SIGMA_FALLBACK_SCALE

    z = abs(float(x) - mu) / sigma
    z = min(z, 5.0)
    return float(1.0 / (1.0 + z))


def feature_score(feat: str, x: float, ref_vals: np.ndarray) -> float:
    """
    Implementa la logica de feature score dentro del pipeline del TFG.

    Parametros principales: feat, x, ref_vals.
    """

    if USE_RANGE_FOR_SCALE_CONSISTENCY and feat == "scale_consistency":
        # No penalizamos por estar por encima de la media de referencia si sigue dentro
        # de un rango musicalmente razonable. Esto corrige el caso observado en el JSON:
        # scale_consistency salía como issue aunque el valor era alto/bueno.
        return range_acceptance_score(x, 0.78, 1.0)
    return strict_feature_score(x, ref_vals)


def per_piece_reference_based(gen_row: pd.Series, local_ref_df: pd.DataFrame) -> Dict:
    # Conjunto de métricas simbólicas usadas en la comparación reference-based.
    """
    Implementa la logica de per piece reference based dentro del pipeline del TFG.

    Parametros principales: gen_row, local_ref_df.
    """

    used_features = [
        "pitch_range",
        "n_pitch_classes_used",
        "n_pitches_used",
        "polyphony",
        "polyphony_rate",
        "pitch_class_entropy",
        "pitch_entropy",
        "scale_consistency",
        "empty_beat_rate",
        "empty_measure_rate",
        "groove_consistency",
        "pitch_histogram_entropy_custom",
        "consecutive_pitch_repetition_ratio_custom",
        "mean_velocity_custom",
        "std_velocity_custom",
        "mean_duration_beats_custom",
        "std_duration_beats_custom",
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
        s = feature_score(feat, x, ref_vals)
        detail_scores[feat] = {
            "x": x,
            "mu_ref": float(ref_vals.mean()),
            "sigma_ref": float(ref_vals.std()) if ref_vals.size > 1 else 0.0,
            "score": s,
        }
        if np.isfinite(s):
            vals.append(s)

    pitch_hist_sims = []
    if "_pitch_hist_json" in local_ref_df.columns:
        for _, ref_row in local_ref_df.iterrows():
            try:
                pitch_hist_sims.append(pitch_hist_similarity_from_json(gen_row["_pitch_hist_json"], ref_row["_pitch_hist_json"]))
            except Exception:
                pass
    pitch_hist_similarity_score = float(np.nanmean(pitch_hist_sims)) if pitch_hist_sims else np.nan

    base_score = float(np.nanmean(vals) * 100.0) if vals else np.nan
    if np.isfinite(base_score) and np.isfinite(pitch_hist_similarity_score):
        score = 0.9 * base_score + 0.1 * (100.0 * pitch_hist_similarity_score)
    else:
        score = base_score

    local_durations = finite_values(local_ref_df["duration_beats_custom"].to_numpy(dtype=float)) if "duration_beats_custom" in local_ref_df.columns else np.asarray([])

    return {
        "score": safe_float(score),
        "pitch_hist_similarity_score": safe_float(pitch_hist_similarity_score),
        "matched_reference_count": int(len(local_ref_df)),
        "details": detail_scores,
        "local_reference_duration_beats": {
            "mean": safe_float(local_durations.mean()) if local_durations.size else np.nan,
            "min": safe_float(local_durations.min()) if local_durations.size else np.nan,
            "max": safe_float(local_durations.max()) if local_durations.size else np.nan,
        },
    }


# ============================================================
# SCORE POR PIEZA: reference-free
# ============================================================
def per_piece_reference_free(gen_row: pd.Series) -> Dict:
    # Comprobaciones internas que no dependen directamente del corpus de referencia.
    """
    Implementa la logica de per piece reference free dentro del pipeline del TFG.

    Parametros principales: gen_row.
    """

    checks = {
        "pitch_class_entropy": range_acceptance_score(safe_float(gen_row.get("pitch_class_entropy", np.nan)), 2.9, 3.45),
        "consecutive_pitch_repetition_ratio_custom": range_acceptance_score(1.0 - safe_float(gen_row.get("consecutive_pitch_repetition_ratio_custom", np.nan)), 0.96, 1.0),
        "empty_measure_rate": range_acceptance_score(1.0 - safe_float(gen_row.get("empty_measure_rate", np.nan)), 0.82, 1.0),
        "scale_consistency": range_acceptance_score(safe_float(gen_row.get("scale_consistency", np.nan)), 0.78, 1.0),
        "polyphony": range_acceptance_score(safe_float(gen_row.get("polyphony", np.nan)), 1.6, 4.3),
        "std_velocity_custom": range_acceptance_score(safe_float(gen_row.get("std_velocity_custom", np.nan)), 7.0, 22.0),
        "groove_consistency": range_acceptance_score(safe_float(gen_row.get("groove_consistency", np.nan)), 0.40, 1.0),
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

    for feat in ["pitch_range", "polyphony", "pitch_class_entropy", "scale_consistency"]:
        s = details_rb.get(feat, {}).get("score", np.nan)
        if np.isfinite(s):
            if s >= 0.82:
                strengths.append(f"{feat} cercano a referencia")
            elif s <= 0.30:
                issues.append(f"{feat} alejado de referencia")

    phs = safe_float(ref_based.get("pitch_hist_similarity_score", np.nan))
    if np.isfinite(phs):
        if phs >= 0.88:
            strengths.append("histograma de pitch muy alineado")
        elif phs <= 0.55:
            issues.append("histograma de pitch poco alineado")

    for feat in ["pitch_class_entropy", "scale_consistency", "groove_consistency"]:
        s = safe_float(details_rf.get(feat, np.nan))
        if np.isfinite(s):
            if s >= 0.90:
                strengths.append(f"{feat} sólido")
            elif s <= 0.35:
                issues.append(f"{feat} problemático")

    cpr = safe_float(details_rf.get("consecutive_pitch_repetition_ratio_custom", np.nan))
    if np.isfinite(cpr) and cpr <= 0.35:
        issues.append("exceso de repetición consecutiva")

    strengths = list(dict.fromkeys(strengths))[:3]
    issues = list(dict.fromkeys(issues))[:3]
    return strengths, issues


# ============================================================
# MAIN
# ============================================================
def main():
    """Punto de entrada del script cuando se ejecuta desde consola."""

    print("[INFO] Iniciando evaluación symbolic MIDI finetuning por ventanas (MusPy + SciPy)")
    print(f"[INFO] GENERATED_DIR = {GENERATED_DIR}")
    print(f"[INFO] REFERENCE_MODE = {REFERENCE_MODE}")
    print(f"[INFO] REFERENCE_DIR  = {REFERENCE_DIR}")
    print(f"[INFO] OUT_DIR       = {OUT_DIR}")

    gen_files = find_midi_files(GENERATED_DIR, RECURSIVE)
    ref_files = choose_reference_files()

    gen_df = build_feature_table(gen_files, "generated")
    ref_full_df = build_feature_table(ref_files, "reference_full_piece")

    if gen_df.empty:
        raise RuntimeError("No se pudo evaluar ninguna pieza generada.")
    if ref_full_df.empty:
        raise RuntimeError("No se pudo evaluar ninguna pieza de referencia.")

    if REFERENCE_WINDOW_MODE == "match_generated":
        window_lengths = window_lengths_from_generated(gen_df)
    elif REFERENCE_WINDOW_MODE == "fixed":
        window_lengths = [REFERENCE_WINDOW_BEATS]
    else:
        raise ValueError(f"REFERENCE_WINDOW_MODE desconocido: {REFERENCE_WINDOW_MODE}")

    print(f"[INFO] window_lengths_beats={window_lengths}")
    ref_df = build_reference_window_table(ref_files, window_lengths)
    if ref_df.empty:
        raise RuntimeError("No se pudo construir ninguna ventana de referencia.")

    per_piece_rows = []
    per_piece_details = []

    for _, gen_row in gen_df.iterrows():
        local_pool = select_local_reference_pool(float(gen_row["duration_beats_custom"]), ref_df)
        ref_based = per_piece_reference_based(gen_row, local_pool)
        ref_free = per_piece_reference_free(gen_row)
        overall = W_REFERENCE_BASED * float(ref_based["score"]) + W_REFERENCE_FREE * float(ref_free["score"])
        label = piece_label(overall)
        strengths, issues = describe_piece(ref_based, ref_free)

        per_piece_rows.append({
            "file": gen_row["file"],
            "duration_beats_custom": float(gen_row["duration_beats_custom"]),
            "n_notes_custom": float(gen_row["n_notes_custom"]),
            "reference_based_score": float(ref_based["score"]),
            "reference_based_pitch_hist_similarity": float(ref_based["pitch_hist_similarity_score"]),
            "reference_based_matched_reference_count": int(ref_based["matched_reference_count"]),
            "reference_free_score": float(ref_free["score"]),
            "global_score": float(overall),
            "qualitative_label": label,
            "strengths": sanitize_text(strengths),
            "issues": sanitize_text(issues),
        })

        per_piece_details.append({
            "file": gen_row["file"],
            "reference_based": ref_based,
            "reference_free": ref_free,
            "global_score": float(overall),
            "qualitative_label": label,
            "strengths": strengths,
            "issues": issues,
        })

    per_piece_df = pd.DataFrame(per_piece_rows).reindex(columns=PER_PIECE_COLUMNS)
    per_piece_df = per_piece_df.sort_values("global_score", ascending=False).reset_index(drop=True)

    # Reporte principal: generado vs ventanas de referencia.
    global_report = global_distribution_report(ref_df, gen_df, GLOBAL_FEATURES)
    # Reporte diagnóstico: generado vs obras completas, útil para mostrar el sesgo por duración.
    global_report_full_piece_diagnostic = global_distribution_report(ref_full_df, gen_df, GLOBAL_FEATURES)

    summary_rows = []
    summary_rows.append({"section": "meta", "key": "generated_dir", "value": str(GENERATED_DIR)})
    summary_rows.append({"section": "meta", "key": "reference_mode", "value": REFERENCE_MODE})
    summary_rows.append({"section": "meta", "key": "reference_dir", "value": str(REFERENCE_DIR)})
    summary_rows.append({"section": "meta", "key": "n_generated_files", "value": int(len(gen_df))})
    summary_rows.append({"section": "meta", "key": "n_reference_files_full_piece", "value": int(len(ref_full_df))})
    summary_rows.append({"section": "meta", "key": "n_reference_windows", "value": int(len(ref_df))})
    summary_rows.append({"section": "meta", "key": "window_lengths_beats", "value": json.dumps(window_lengths)})
    summary_rows.append({"section": "meta", "key": "scoring_note", "value": "Mismas métricas/pesos que finetuning; referencias recortadas en ventanas; scale_consistency corregida como rango; sigma cero con fallback."})

    for _, row in global_report.head(10).iterrows():
        summary_rows.append({
            "section": "global_report_top10_windows",
            "key": row["feature"],
            "value": json.dumps({
                "oa": safe_float(row["oa"]),
                "kld_real_to_gen": safe_float(row["kld_real_to_gen"]),
                "real_mean": safe_float(row["real_mean"]),
                "gen_mean": safe_float(row["gen_mean"]),
            }, ensure_ascii=False),
        })

    for _, row in global_report_full_piece_diagnostic.head(10).iterrows():
        summary_rows.append({
            "section": "global_report_top10_full_piece_diagnostic",
            "key": row["feature"],
            "value": json.dumps({
                "oa": safe_float(row["oa"]),
                "kld_real_to_gen": safe_float(row["kld_real_to_gen"]),
                "real_mean": safe_float(row["real_mean"]),
                "gen_mean": safe_float(row["gen_mean"]),
            }, ensure_ascii=False),
        })

    summary_df = pd.DataFrame(summary_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reference_csv = OUT_DIR / "reference_features.csv"  # ventanas, usado para scoring
    reference_full_piece_csv = OUT_DIR / "reference_full_piece_features.csv"  # diagnóstico
    generated_csv = OUT_DIR / "generated_features.csv"
    per_piece_csv = OUT_DIR / "per_piece_evaluation.csv"
    summary_csv = OUT_DIR / "summary_compact.csv"
    per_piece_details_json = OUT_DIR / "per_piece_details.json"
    global_report_csv = OUT_DIR / "global_report_windows.csv"
    global_report_full_piece_csv = OUT_DIR / "global_report_full_piece_diagnostic.csv"

    for p in [
        reference_csv,
        reference_full_piece_csv,
        generated_csv,
        per_piece_csv,
        summary_csv,
        per_piece_details_json,
        global_report_csv,
        global_report_full_piece_csv,
    ]:
        if p.exists():
            p.unlink()

    ref_df.drop(columns=["_pitch_hist_json"], errors="ignore").to_csv(reference_csv, index=False, encoding="utf-8-sig")
    ref_full_df.drop(columns=["_pitch_hist_json"], errors="ignore").to_csv(reference_full_piece_csv, index=False, encoding="utf-8-sig")
    gen_df.drop(columns=["_pitch_hist_json"], errors="ignore").to_csv(generated_csv, index=False, encoding="utf-8-sig")
    per_piece_df.to_csv(per_piece_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    global_report.to_csv(global_report_csv, index=False, encoding="utf-8-sig")
    global_report_full_piece_diagnostic.to_csv(global_report_full_piece_csv, index=False, encoding="utf-8-sig")

    with open(per_piece_details_json, "w", encoding="utf-8") as f:
        json.dump(per_piece_details, f, ensure_ascii=False, indent=2)

    print("[OK] Archivos generados:")
    print(f"  - {reference_csv}")
    print(f"  - {reference_full_piece_csv}")
    print(f"  - {generated_csv}")
    print(f"  - {global_report_csv}")
    print(f"  - {global_report_full_piece_csv}")
    print(f"  - {per_piece_csv}")
    print(f"  - {summary_csv}")
    print(f"  - {per_piece_details_json}")


# Ejecucion directa del script.
if __name__ == "__main__":
    main()
