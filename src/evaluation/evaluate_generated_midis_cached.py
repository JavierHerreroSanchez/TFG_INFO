from __future__ import annotations

"""
Evaluación symbolic MIDI para pretraining con caché completa de referencias.

Cambio principal respecto al script original:
- Antes se muestreaban como máximo 4000 MIDIs de referencia y, desde ahí, se elegía
  un pool local de 48 referencias por pieza generada.
- Ahora se construye primero una caché CSV con las features simbólicas de TODO el
  corpus de referencia disponible (MAESTRO + ARIA, por defecto).
- Si la caché ya existe, se carga directamente.
- El score por pieza sigue usando una subselección local de duración parecida
  (LOCAL_REF_POOL_SIZE = 48), pero esa subselección se toma desde la caché completa.

Esto permite que:
1) el score reference-based use vecinos locales más representativos;
2) OA/KLD se calculen contra la distribución completa cacheada, no contra una muestra pequeña;
3) no sea necesario recalcular las features del corpus en cada evaluación.
"""

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import muspy
import numpy as np
import pandas as pd
from scipy.stats import entropy

# ============================================================
# CONFIGURACION PYCHARM
# ============================================================
GENERATED_DIR = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\output\generation_pretraining_tfg_first\batch_1")
OUT_DIR = Path(r"../../output/generation_pretraining_tfg_first/batch_1/midi_eval_final")

# Modo de referencia:
# - "single_dir": usa REFERENCE_DIR completo
# - "mixed_all": usa MAESTRO_DIR + ARIA_DIR completos
# - "mixed_random": compatibilidad con el modo antiguo; solo se recomienda para pruebas rápidas
REFERENCE_MODE = "mixed_all"

REFERENCE_DIR = Path(r"../../data/pretraining_raw/maestro-v3.0.0")
MAESTRO_DIR = Path(r"../../data/pretraining_raw/maestro-v3.0.0")
ARIA_DIR = Path(r"../../data/pretraining_raw/ariamidi")

RECURSIVE = True

# Caché completa de features simbólicas de referencia.
# La primera ejecución la crea; las siguientes la reutilizan.
REFERENCE_CACHE_DIR = Path(r"../../data/interim/indexes").resolve()
REFERENCE_FEATURES_CSV = REFERENCE_CACHE_DIR / "pretraining_symbolic_reference_features.csv"
REFERENCE_FAILED_CSV = REFERENCE_CACHE_DIR / "pretraining_symbolic_reference_failed.csv"
REBUILD_REFERENCE_CACHE = False
RESUME_PARTIAL_CACHE = True
CACHE_WRITE_EVERY = 1000

# Solo para pruebas rápidas. Para la evaluación final, dejar en None.
# Si REFERENCE_MODE="mixed_random" y MAX_REFERENCE_FILES no es None, se muestrea el corpus.
MAX_REFERENCE_FILES: Optional[int] = None
MAESTRO_FRACTION = 0.5
REFERENCE_RANDOM_SEED = 1453

# Pool local de referencias para score por pieza.
# IMPORTANTE: ahora se elige desde la caché completa, no desde una muestra de 4000.
LOCAL_REF_POOL_SIZE = 600
LOCAL_REF_DURATION_TOL = 0.30  # +/-30%

# Pesos del score global
W_REFERENCE_BASED = 0.75
W_REFERENCE_FREE = 0.25

QUAL_LABELS = [
    (85.0, "muy plausible"),
    (72.0, "plausible"),
    (58.0, "aceptable con anomalías"),
    (45.0, "débil"),
    (-1.0, "fuera de distribución"),
]

# Resolución para MusPy
MUSPY_RESOLUTION = 24
DEFAULT_MEASURE_RESOLUTION = 96  # 4/4 con 24 ticks por negra

# Features de reporte global referencia vs generado
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

REFERENCE_CACHE_REQUIRED_COLUMNS = ["file", "_pitch_hist_json", *GLOBAL_FEATURES]

# ============================================================
# UTILIDADES GENERALES
# ============================================================
def find_midi_files(root: Path, recursive: bool = True) -> List[Path]:
    pats = ["*.mid", "*.midi"]
    files: List[Path] = []
    for pat in pats:
        files.extend(root.rglob(pat) if recursive else root.glob(pat))
    return sorted({p.resolve() for p in files if p.is_file()})


def safe_float(x, default=np.nan) -> float:
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
    for th, label in QUAL_LABELS:
        if score >= th:
            return label
    return QUAL_LABELS[-1][1]


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


def infer_reference_source(path: Path) -> str:
    s = str(path).replace("\\", "/").lower()
    if "maestro" in s:
        return "maestro"
    if "aria" in s:
        return "aria"
    return "unknown"


def choose_reference_files() -> List[Path]:
    """
    Devuelve la lista de MIDIs de referencia.

    Para evaluación final de pretraining se recomienda REFERENCE_MODE="mixed_all"
    y MAX_REFERENCE_FILES=None, para usar todo MAESTRO + ARIA.
    """
    if REFERENCE_MODE == "single_dir":
        files = find_midi_files(REFERENCE_DIR, RECURSIVE)
        if MAX_REFERENCE_FILES is not None:
            files = files[:MAX_REFERENCE_FILES]
        print(f"[INFO] REFERENCE_MODE=single_dir | n_reference_files={len(files)}")
        return files

    if REFERENCE_MODE not in {"mixed_all", "mixed_random"}:
        raise ValueError(f"REFERENCE_MODE desconocido: {REFERENCE_MODE}")

    maestro_files = find_midi_files(MAESTRO_DIR, RECURSIVE)
    aria_files = find_midi_files(ARIA_DIR, RECURSIVE)

    if REFERENCE_MODE == "mixed_all" or MAX_REFERENCE_FILES is None:
        combined = maestro_files + aria_files
        print(
            f"[INFO] REFERENCE_MODE=mixed_all | "
            f"maestro_total={len(maestro_files)} aria_total={len(aria_files)} | "
            f"final_reference_files={len(combined)}"
        )
        return sorted({p.resolve() for p in combined if p.is_file()})

    # Compatibilidad con el modo antiguo: muestreo aleatorio estratificado.
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
            combined.extend(rng.sample(leftovers, min(remaining, len(leftovers))))

    rng.shuffle(combined)
    print(
        f"[INFO] REFERENCE_MODE=mixed_random | "
        f"maestro_total={len(maestro_files)} aria_total={len(aria_files)} | "
        f"final_reference_files={len(combined)}"
    )
    return combined


# ============================================================
# CONVERSION Y FILTRADO MUSPY
# ============================================================
def load_music(path: Path) -> muspy.Music:
    music = muspy.read_midi(path)
    if music.resolution is None:
        music.resolution = MUSPY_RESOLUTION
    return music


def keep_only_piano_tracks(music: muspy.Music) -> muspy.Music:
    piano_programs = set(range(0, 8))
    kept = []
    for track in music.tracks:
        program = getattr(track, "program", 0)
        if program in piano_programs:
            kept.append(track)
    music.tracks = kept
    return music


# ============================================================
# FEATURES CUSTOM MINIMAS
# ============================================================
def iter_notes(music: muspy.Music):
    for track in music.tracks:
        for note in track.notes:
            yield note


def all_notes_sorted(music: muspy.Music) -> List:
    notes = list(iter_notes(music))
    notes.sort(key=lambda n: (n.time, n.pitch, getattr(n, "velocity", 64)))
    return notes


def pitch_histogram_entropy_custom(music: muspy.Music) -> float:
    notes = all_notes_sorted(music)
    if not notes:
        return np.nan
    hist = np.zeros(12, dtype=float)
    for n in notes:
        hist[n.pitch % 12] += 1.0
    hist /= hist.sum()
    return float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum())


def consecutive_pitch_repetition_ratio_custom(music: muspy.Music) -> float:
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
    notes = all_notes_sorted(music)
    if not notes:
        return np.asarray([], dtype=float)
    res = music.resolution or MUSPY_RESOLUTION
    return np.asarray([n.duration / res for n in notes], dtype=float)


def note_velocities(music: muspy.Music) -> np.ndarray:
    notes = all_notes_sorted(music)
    if not notes:
        return np.asarray([], dtype=float)
    return np.asarray([getattr(n, "velocity", 64) for n in notes], dtype=float)


def duration_beats_custom(music: muspy.Music) -> float:
    end_time = 0
    for n in all_notes_sorted(music):
        end_time = max(end_time, int(n.time + n.duration))
    res = music.resolution or MUSPY_RESOLUTION
    return float(end_time / res) if res else np.nan


def n_notes_custom(music: muspy.Music) -> float:
    return float(len(all_notes_sorted(music)))


def mean_velocity_custom(music: muspy.Music) -> float:
    arr = note_velocities(music)
    return float(np.mean(arr)) if arr.size else np.nan


def std_velocity_custom(music: muspy.Music) -> float:
    arr = note_velocities(music)
    return float(np.std(arr)) if arr.size else np.nan


def mean_duration_beats_custom(music: muspy.Music) -> float:
    arr = note_durations_beats(music)
    return float(np.mean(arr)) if arr.size else np.nan


def std_duration_beats_custom(music: muspy.Music) -> float:
    arr = note_durations_beats(music)
    return float(np.std(arr)) if arr.size else np.nan


def pitch_histogram_12(music: muspy.Music) -> np.ndarray:
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
    fn = getattr(muspy.metrics, name)
    try:
        if name in {"empty_measure_rate", "groove_consistency"}:
            return safe_float(fn(music, measure_resolution=DEFAULT_MEASURE_RESOLUTION))
        return safe_float(fn(music))
    except Exception:
        return np.nan


# ============================================================
# EXTRACCION FEATURES POR PIEZA
# ============================================================
def extract_features(path: Path) -> Dict[str, float | str]:
    music = load_music(path)
    music = keep_only_piano_tracks(music)

    row: Dict[str, float | str] = {
        "file": str(path),
        "source": infer_reference_source(path),
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


def build_feature_table(files: List[Path], tag: str) -> pd.DataFrame:
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
# CACHE DE FEATURES DE REFERENCIA
# ============================================================
def validate_reference_cache(df: pd.DataFrame) -> None:
    missing = [c for c in REFERENCE_CACHE_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"La caché de referencia no contiene columnas requeridas: {missing}")


def coerce_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in GLOBAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "file" in df.columns:
        df["file"] = df["file"].astype(str)
    if "_pitch_hist_json" in df.columns:
        df["_pitch_hist_json"] = df["_pitch_hist_json"].astype(str)
    return df


def load_reference_cache(cache_csv: Path) -> pd.DataFrame:
    print(f"[cache] Cargando caché de referencia: {cache_csv}")
    df = pd.read_csv(cache_csv)
    validate_reference_cache(df)
    df = coerce_feature_columns(df)
    print(f"[cache] OK | reference_rows={len(df):,}")
    return df


def append_rows_to_csv(rows: List[Dict], csv_path: Path, write_header: bool) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(
        csv_path,
        mode="w" if write_header else "a",
        header=write_header,
        index=False,
        encoding="utf-8-sig",
    )


def build_reference_feature_cache(cache_csv: Path) -> pd.DataFrame:
    """
    Construye la caché CSV de features simbólicas para todo el corpus de referencia.

    Se escribe incrementalmente para no guardar las 372k filas en memoria durante
    la extracción. Si existe un .partial.csv y RESUME_PARTIAL_CACHE=True, continúa
    desde ahí.
    """
    REFERENCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_csv = cache_csv.with_suffix(cache_csv.suffix + ".partial")

    ref_files = choose_reference_files()
    if not ref_files:
        raise RuntimeError("No se encontró ningún MIDI de referencia para construir la caché.")

    processed: set[str] = set()
    write_header = True

    if tmp_csv.exists() and RESUME_PARTIAL_CACHE:
        try:
            processed_df = pd.read_csv(tmp_csv, usecols=["file"])
            processed = set(processed_df["file"].astype(str).tolist())
            write_header = False
            print(f"[cache] Reanudando caché parcial: {tmp_csv} | ya procesados={len(processed):,}")
        except Exception as e:
            print(f"[cache][WARN] No se pudo leer la caché parcial ({e}). Se reconstruirá desde cero.")
            tmp_csv.unlink(missing_ok=True)
            write_header = True
    else:
        tmp_csv.unlink(missing_ok=True)

    failed_rows = []
    batch = []
    total = len(ref_files)

    print(f"[cache] Construyendo caché de features simbólicas | total_files={total:,}")
    for i, path in enumerate(ref_files, start=1):
        path_str = str(path)
        if path_str in processed:
            if i % 5000 == 0 or i == total:
                print(f"[cache] skip {i:,}/{total:,} | ya en parcial")
            continue

        try:
            batch.append(extract_features(path))
        except Exception as e:
            failed_rows.append({"file": path_str, "error": repr(e)})

        if len(batch) >= CACHE_WRITE_EVERY:
            append_rows_to_csv(batch, tmp_csv, write_header=write_header)
            write_header = False
            batch.clear()

        if i % 1000 == 0 or i == total:
            written = "?"
            if tmp_csv.exists():
                try:
                    # Número aproximado: contar líneas es caro pero útil cada 1000; no lo hacemos exacto.
                    written = "partial"
                except Exception:
                    pass
            print(f"[cache] procesados {i:,}/{total:,} | batch={len(batch)} | written={written} | failed={len(failed_rows):,}")

    if batch:
        append_rows_to_csv(batch, tmp_csv, write_header=write_header)
        batch.clear()

    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(REFERENCE_FAILED_CSV, index=False, encoding="utf-8-sig")
        print(f"[cache][WARN] Fallos guardados en: {REFERENCE_FAILED_CSV} | n_failed={len(failed_rows):,}")

    tmp_csv.replace(cache_csv)
    print(f"[cache] Caché completada: {cache_csv}")
    return load_reference_cache(cache_csv)


def load_or_build_reference_features() -> pd.DataFrame:
    if REFERENCE_FEATURES_CSV.exists() and not REBUILD_REFERENCE_CACHE:
        return load_reference_cache(REFERENCE_FEATURES_CSV)

    if REBUILD_REFERENCE_CACHE and REFERENCE_FEATURES_CSV.exists():
        print(f"[cache] REBUILD_REFERENCE_CACHE=True -> eliminando caché previa: {REFERENCE_FEATURES_CSV}")
        REFERENCE_FEATURES_CSV.unlink()

    return build_reference_feature_cache(REFERENCE_FEATURES_CSV)


# ============================================================
# OA / KLD CON SCIPY + HISTOGRAMAS
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
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["oa", "kld_real_to_gen"], ascending=[False, True]).reset_index(drop=True)
    return out


# ============================================================
# SCORE POR PIEZA: reference-based
# ============================================================
def select_local_reference_pool(duration_beats: float, ref_df: pd.DataFrame) -> pd.DataFrame:
    """
    Selecciona referencias locales desde la caché completa.

    Primero intenta usar piezas dentro de +/- LOCAL_REF_DURATION_TOL.
    Si hay suficientes, toma las LOCAL_REF_POOL_SIZE más cercanas en duración.
    Si no hay suficientes, toma directamente las LOCAL_REF_POOL_SIZE más cercanas
    de toda la caché.
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
    a = np.asarray(json.loads(hist_json_a), dtype=float)
    b = np.asarray(json.loads(hist_json_b), dtype=float)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    a = a / a.sum()
    b = b / b.sum()
    return float(np.minimum(a, b).sum())


def strict_feature_score(x: float, ref_vals: np.ndarray, eps: float = 1e-8) -> float:
    """
    Score sencillo:
    score = 1 / (1 + z), con z = |x - mu| / sigma
    sigma con suelo mínimo para evitar explosiones.
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
        s = strict_feature_score(x, ref_vals)
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
                pitch_hist_sims.append(
                    pitch_hist_similarity_from_json(gen_row["_pitch_hist_json"], ref_row["_pitch_hist_json"])
                )
            except Exception:
                pass
    pitch_hist_similarity_score = float(np.nanmean(pitch_hist_sims)) if pitch_hist_sims else np.nan

    base_score = float(np.nanmean(vals) * 100.0) if vals else np.nan
    if np.isfinite(base_score) and np.isfinite(pitch_hist_similarity_score):
        # 90% features, 10% histograma
        score = 0.9 * base_score + 0.1 * (100.0 * pitch_hist_similarity_score)
    else:
        score = base_score

    return {
        "score": safe_float(score),
        "pitch_hist_similarity_score": safe_float(pitch_hist_similarity_score),
        "matched_reference_count": int(len(local_ref_df)),
        "details": detail_scores,
    }


# ============================================================
# SCORE POR PIEZA: reference-free
# ============================================================
def range_acceptance_score(x: float, lo: float, hi: float, softness: float | None = None) -> float:
    """
    Score = 1 si cae dentro del rango aceptable.
    Fuera del rango, caída suave según distancia.
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
    checks = {
        "pitch_class_entropy": range_acceptance_score(safe_float(gen_row.get("pitch_class_entropy", np.nan)), 2.9, 3.45),
        "consecutive_pitch_repetition_ratio_custom": range_acceptance_score(
            1.0 - safe_float(gen_row.get("consecutive_pitch_repetition_ratio_custom", np.nan)), 0.96, 1.0
        ),
        "empty_measure_rate": range_acceptance_score(
            1.0 - safe_float(gen_row.get("empty_measure_rate", np.nan)), 0.82, 1.0
        ),
        "scale_consistency": range_acceptance_score(safe_float(gen_row.get("scale_consistency", np.nan)), 0.78, 1.0),
        "polyphony": range_acceptance_score(safe_float(gen_row.get("polyphony", np.nan)), 1.6, 4.3),
        "std_velocity_custom": range_acceptance_score(safe_float(gen_row.get("std_velocity_custom", np.nan)), 7.0, 22.0),
        "groove_consistency": range_acceptance_score(safe_float(gen_row.get("groove_consistency", np.nan)), 0.40, 1.0),
    }
    vals = [v for v in checks.values() if np.isfinite(v)]
    score = float(np.nanmean(vals) * 100.0) if vals else np.nan
    return {"score": safe_float(score), "details": checks}


# ============================================================
# DESCRIPCION CUALITATIVA
# ============================================================
def describe_piece(ref_based: Dict, ref_free: Dict) -> Tuple[List[str], List[str]]:
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
    print("[INFO] Iniciando evaluación symbolic MIDI con caché completa (MusPy + SciPy)")
    print(f"[INFO] GENERATED_DIR = {GENERATED_DIR}")
    print(f"[INFO] REFERENCE_MODE = {REFERENCE_MODE}")
    print(f"[INFO] REFERENCE_FEATURES_CSV = {REFERENCE_FEATURES_CSV}")
    print(f"[INFO] OUT_DIR       = {OUT_DIR}")

    gen_files = find_midi_files(GENERATED_DIR, RECURSIVE)
    gen_df = build_feature_table(gen_files, "generated")
    ref_df = load_or_build_reference_features()

    if gen_df.empty:
        raise RuntimeError("No se pudo evaluar ninguna pieza generada.")
    if ref_df.empty:
        raise RuntimeError("No se pudo evaluar ninguna pieza de referencia.")

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

        local_duration_stats = {}
        if not local_pool.empty and "duration_beats_custom" in local_pool.columns:
            d = finite_values(local_pool["duration_beats_custom"].to_numpy(dtype=float))
            if d.size:
                local_duration_stats = {
                    "min": float(d.min()),
                    "mean": float(d.mean()),
                    "max": float(d.max()),
                }

        per_piece_details.append({
            "file": gen_row["file"],
            "reference_based": ref_based,
            "reference_free": ref_free,
            "global_score": float(overall),
            "qualitative_label": label,
            "strengths": strengths,
            "issues": issues,
            "local_reference_duration_beats": local_duration_stats,
        })

    per_piece_df = pd.DataFrame(per_piece_rows).reindex(columns=PER_PIECE_COLUMNS)
    per_piece_df = per_piece_df.sort_values("global_score", ascending=False).reset_index(drop=True)

    # OA/KLD contra la caché completa de referencias.
    global_report = global_distribution_report(ref_df, gen_df, GLOBAL_FEATURES)

    summary_rows = []
    summary_rows.append({"section": "meta", "key": "generated_dir", "value": str(GENERATED_DIR)})
    summary_rows.append({"section": "meta", "key": "reference_mode", "value": REFERENCE_MODE})
    summary_rows.append({"section": "meta", "key": "reference_features_csv", "value": str(REFERENCE_FEATURES_CSV)})
    summary_rows.append({"section": "meta", "key": "n_generated_files", "value": int(len(gen_df))})
    summary_rows.append({"section": "meta", "key": "n_reference_feature_rows", "value": int(len(ref_df))})
    summary_rows.append({"section": "meta", "key": "local_ref_pool_size", "value": int(LOCAL_REF_POOL_SIZE)})
    summary_rows.append({"section": "meta", "key": "local_ref_duration_tol", "value": float(LOCAL_REF_DURATION_TOL)})

    for _, row in global_report.head(10).iterrows():
        summary_rows.append({
            "section": "global_report_top10",
            "key": row["feature"],
            "value": json.dumps({
                "oa": safe_float(row["oa"]),
                "kld_real_to_gen": safe_float(row["kld_real_to_gen"]),
                "real_mean": safe_float(row["real_mean"]),
                "gen_mean": safe_float(row["gen_mean"]),
                "n_real": int(row["n_real"]),
                "n_gen": int(row["n_gen"]),
            }, ensure_ascii=False),
        })

    summary_df = pd.DataFrame(summary_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_piece_csv = OUT_DIR / "per_piece_evaluation.csv"
    summary_csv = OUT_DIR / "summary_compact.csv"
    per_piece_details_json = OUT_DIR / "per_piece_details.json"
    global_report_csv = OUT_DIR / "global_distribution_report.csv"
    generated_features_csv = OUT_DIR / "generated_features.csv"

    for p in [per_piece_csv, summary_csv, per_piece_details_json, global_report_csv, generated_features_csv]:
        if p.exists():
            p.unlink()

    gen_df.drop(columns=["_pitch_hist_json"], errors="ignore").to_csv(
        generated_features_csv, index=False, encoding="utf-8-sig"
    )
    per_piece_df.to_csv(per_piece_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    global_report.to_csv(global_report_csv, index=False, encoding="utf-8-sig")

    with open(per_piece_details_json, "w", encoding="utf-8") as f:
        json.dump(per_piece_details, f, ensure_ascii=False, indent=2)

    print("[OK] Archivos generados:")
    print(f"  - {generated_features_csv}")
    print(f"  - {global_report_csv}")
    print(f"  - {per_piece_csv}")
    print(f"  - {summary_csv}")
    print(f"  - {per_piece_details_json}")
    print(f"[OK] Caché de referencia usada:")
    print(f"  - {REFERENCE_FEATURES_CSV}")


if __name__ == "__main__":
    main()
