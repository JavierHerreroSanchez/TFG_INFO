from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import pretty_midi
except Exception as e:  # pragma: no cover
    raise SystemExit("No se pudo importar pretty_midi. Instálalo con: pip install pretty_midi") from e

# =============================================================================
# CONFIGURACIÓN DE RUTAS
# =============================================================================
GENERATED_DIR = Path(r"/output/generation_v2")
REFERENCE_DIR = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\data\raw\ariamidi\aa")
OUT_DIR = Path(r"/output/midi_eval_refsplit")

RECURSIVE = True
PIANO_ONLY = True              # Filtra solo programas MIDI 0..7 e ignora percusión.
INCLUDE_DRUMS = False

# Pool de referencias locales por pieza generada (emparejadas por duración)
LOCAL_REF_K = 64               # nº máximo de referencias para el pool local.
LOCAL_REF_MIN = 12             # nº mínimo deseable.
LOCAL_DURATION_TOL = 0.50      # tolerancia relativa inicial (+/-50%).

# Métricas basadas en referencias
REFERENCE_BASED_FEATURES = [
    "pitch_range",
    "used_pitch_classes",
    "avg_pitch_interval",
    "polyphony",
    "qualified_note_ratio",
    "onset_density",
    "mean_duration",
    "std_duration",
    "mean_velocity",
    "std_velocity",
    "pitch_std",
    "tone_span",
    "pitch_class_entropy",
    "consecutive_pitch_repetition_ratio",
    "empty_bar_ratio",
    "scale_consistency",
    "repetition_rate",
]

REFERENCE_BASED_WEIGHTS = {
    "pitch_range": 1.0,
    "used_pitch_classes": 0.9,
    "avg_pitch_interval": 1.0,
    "polyphony": 1.1,
    "qualified_note_ratio": 0.7,
    "onset_density": 1.0,
    "mean_duration": 0.8,
    "std_duration": 0.8,
    "mean_velocity": 0.6,
    "std_velocity": 0.8,
    "pitch_std": 0.8,
    "tone_span": 0.9,
    "pitch_class_entropy": 1.0,
    "consecutive_pitch_repetition_ratio": 1.0,
    "empty_bar_ratio": 0.7,
    "scale_consistency": 0.8,
    "repetition_rate": 0.9,
}

# Métricas sin referencia
REFERENCE_FREE_WEIGHTS = {
    "pitch_class_entropy": 1.0,
    "consecutive_pitch_repetition_ratio": 1.0,
    "empty_bar_ratio": 0.8,
    "scale_consistency": 0.8,
    "repetition_rate": 0.8,
    "polyphony": 0.7,
    "tone_span": 0.7,
    "onset_density": 0.7,
    "qualified_note_ratio": 0.6,
    "std_velocity": 0.5,
}

# Mezcla final de score.
W_REFERENCE_BASED = 0.65
W_REFERENCE_FREE = 0.35

# OA / KLD global corpus-vs-corpus.
HIST_BINS = 30
EPS = 1e-12
MIDI_SUFFIXES = {".mid", ".midi"}
PIANO_PROGRAMS = set(range(8))

PER_PIECE_COLUMNS = [
    "file",
    "duration_s",
    "n_notes",
    "reference_based_score",
    "reference_based_pitch_hist_similarity",
    "reference_based_matched_reference_count",
    "reference_free_score",
    "global_score",
    "qualitative_label",
    "strengths",
    "issues",
    "reference_based_details_json",
    "reference_free_details_json",
]


# =============================================================================
# ESTRUCTURAS
# =============================================================================
@dataclass
class NoteEvent:
    pitch: int
    start: float
    end: float
    velocity: int
    program: int
    is_drum: bool

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


# =============================================================================
# UTILIDADES GENERALES
# =============================================================================
def safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def safe_std(values: Sequence[float]) -> float:
    return float(np.std(values)) if values else 0.0


def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_exp_from_z(z: float, softness: float = 1.25) -> float:
    """Convierte distancia en z-score a score [0,100]."""
    z = max(0.0, float(z))
    return 100.0 * math.exp(-z / max(softness, EPS))


def piece_label(score: float) -> str:
    if score >= 85:
        return "muy plausible"
    if score >= 70:
        return "plausible"
    if score >= 55:
        return "aceptable con anomalías"
    if score >= 40:
        return "débil"
    return "fuera de distribución"


def weighted_average(items: Iterable[Tuple[float, float]]) -> float:
    num = 0.0
    den = 0.0
    for value, weight in items:
        num += value * weight
        den += weight
    return num / den if den > 0 else 0.0


# =============================================================================
# CARGA DE MIDI
# =============================================================================
def find_midi_files(root: Path, recursive: bool = True) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"No existe el directorio: {root}")
    globber = root.rglob if recursive else root.glob
    files = sorted(p for p in globber("*") if p.suffix.lower() in MIDI_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No se encontraron MIDIs en: {root}")
    return files


def load_pm(path: Path) -> pretty_midi.PrettyMIDI:
    try:
        return pretty_midi.PrettyMIDI(str(path))
    except Exception as e:
        raise RuntimeError(f"No se pudo abrir el MIDI: {path}") from e


def collect_notes(pm: pretty_midi.PrettyMIDI) -> List[NoteEvent]:
    notes: List[NoteEvent] = []
    for inst in pm.instruments:
        if inst.is_drum and not INCLUDE_DRUMS:
            continue
        if PIANO_ONLY and (inst.is_drum or inst.program not in PIANO_PROGRAMS):
            continue
        for n in inst.notes:
            if n.end <= n.start:
                continue
            notes.append(
                NoteEvent(
                    pitch=int(n.pitch),
                    start=float(n.start),
                    end=float(n.end),
                    velocity=int(n.velocity),
                    program=int(inst.program),
                    is_drum=bool(inst.is_drum),
                )
            )
    notes.sort(key=lambda x: (x.start, x.pitch, x.end))
    return notes


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================
def average_polyphony(notes: Sequence[NoteEvent]) -> float:
    if not notes:
        return 0.0
    events: List[Tuple[float, int]] = []
    for n in notes:
        events.append((n.start, +1))
        events.append((n.end, -1))
    events.sort(key=lambda x: (x[0], -x[1]))

    active = 0
    prev_t = events[0][0]
    weighted = 0.0
    active_time = 0.0
    for t, delta in events:
        dt = t - prev_t
        if dt > 0 and active > 0:
            weighted += active * dt
            active_time += dt
        active += delta
        prev_t = t
    return weighted / active_time if active_time > 0 else 0.0


def duration_buckets(durations: Sequence[float], bucket_ms: int = 50) -> List[float]:
    step = bucket_ms / 1000.0
    return [round(d / step) * step for d in durations]


def pitch_class_histogram(notes: Sequence[NoteEvent]) -> np.ndarray:
    hist = np.zeros(12, dtype=np.float64)
    for n in notes:
        hist[n.pitch % 12] += 1.0
    s = hist.sum()
    return hist / s if s > 0 else hist


def pitch_class_entropy_from_hist(hist: np.ndarray) -> float:
    mask = hist > 0
    return float(-(hist[mask] * np.log2(hist[mask])).sum()) if mask.any() else 0.0


def consecutive_pitch_repetition_ratio(pitches: Sequence[int], n: int = 3) -> float:
    if len(pitches) < n:
        return 0.0
    marked = np.zeros(len(pitches), dtype=np.float64)
    i = 0
    while i < len(pitches):
        j = i + 1
        while j < len(pitches) and pitches[j] == pitches[i]:
            j += 1
        if (j - i) >= n:
            marked[i:j] = 1.0
        i = j
    return float(marked.mean())


def repetition_rate_ngram(pitches: Sequence[int], n: int = 3) -> float:
    if len(pitches) < n:
        return 0.0
    grams = [tuple(pitches[i:i+n]) for i in range(len(pitches) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / max(1, len(grams))


MAJOR_SCALES = [set((root + interval) % 12 for interval in [0, 2, 4, 5, 7, 9, 11]) for root in range(12)]
MINOR_SCALES = [set((root + interval) % 12 for interval in [0, 2, 3, 5, 7, 8, 10]) for root in range(12)]
ALL_DIATONIC = MAJOR_SCALES + MINOR_SCALES


def scale_consistency(notes: Sequence[NoteEvent]) -> float:
    if not notes:
        return 0.0
    pcs = [n.pitch % 12 for n in notes]
    total = len(pcs)
    best = 0
    for scale in ALL_DIATONIC:
        inside = sum(1 for pc in pcs if pc in scale)
        best = max(best, inside)
    return best / total if total > 0 else 0.0


def empty_bar_ratio(pm: pretty_midi.PrettyMIDI, notes: Sequence[NoteEvent]) -> float:
    if not notes:
        return 1.0
    try:
        downbeats = list(pm.get_downbeats())
    except Exception:
        downbeats = []
    end_time = max(pm.get_end_time(), max(n.end for n in notes))

    if len(downbeats) < 2:
        # Fallback si no hay downbeats detectables: aproximamos con compases de 2s.
        if end_time <= 0:
            return 1.0
        n_bars = max(1, int(math.ceil(end_time / 2.0)))
        edges = np.linspace(0.0, end_time, n_bars + 1)
    else:
        edges = np.array(downbeats, dtype=np.float64)
        if edges[-1] < end_time:
            last_bar = edges[-1] - edges[-2]
            edges = np.append(edges, max(end_time, edges[-1] + max(last_bar, 1e-3)))

    onsets = np.array([n.start for n in notes], dtype=np.float64)
    empty = 0
    total = 0
    for i in range(len(edges) - 1):
        a, b = float(edges[i]), float(edges[i + 1])
        total += 1
        if not np.any((onsets >= a) & (onsets < b)):
            empty += 1
    return empty / total if total > 0 else 0.0


def extract_features(path: Path) -> Dict[str, float]:
    pm = load_pm(path)
    notes = collect_notes(pm)
    total_duration = float(max(pm.get_end_time(), max((n.end for n in notes), default=0.0)))

    if not notes:
        empty_hist = np.zeros(12, dtype=np.float64)
        return {
            "file": str(path),
            "duration_s": total_duration,
            "n_notes": 0.0,
            "pitch_range": 0.0,
            "used_pitch_classes": 0.0,
            "avg_pitch_interval": 0.0,
            "unique_pitches": 0.0,
            "unique_durations": 0.0,
            "polyphony": 0.0,
            "qualified_note_ratio": 0.0,
            "onset_density": 0.0,
            "mean_duration": 0.0,
            "std_duration": 0.0,
            "mean_velocity": 0.0,
            "std_velocity": 0.0,
            "pitch_std": 0.0,
            "tone_span": 0.0,
            "pitch_class_entropy": 0.0,
            "consecutive_pitch_repetition_ratio": 1.0,
            "empty_bar_ratio": 1.0,
            "scale_consistency": 0.0,
            "repetition_rate": 0.0,
            **{f"pch_{i}": float(empty_hist[i]) for i in range(12)},
        }

    pitches = [n.pitch for n in notes]
    durations = [n.duration for n in notes]
    velocities = [n.velocity for n in notes]
    onset_sorted = sorted(notes, key=lambda n: (n.start, n.pitch, n.end))
    onset_pitches = [n.pitch for n in onset_sorted]
    intervals = [abs(onset_pitches[i] - onset_pitches[i - 1]) for i in range(1, len(onset_pitches))]
    hist = pitch_class_histogram(notes)

    result = {
        "file": str(path),
        "duration_s": total_duration,
        "n_notes": float(len(notes)),
        "pitch_range": float(max(pitches) - min(pitches)),
        "used_pitch_classes": float(len({p % 12 for p in pitches})),
        "avg_pitch_interval": safe_mean(intervals),
        "unique_pitches": float(len(set(pitches))),
        "unique_durations": float(len(set(duration_buckets(durations, 50)))),
        "polyphony": average_polyphony(notes),
        "qualified_note_ratio": float(sum(1 for d in durations if d >= 0.125) / len(durations)),
        "onset_density": len(notes) / max(total_duration, EPS),
        "mean_duration": safe_mean(durations),
        "std_duration": safe_std(durations),
        "mean_velocity": safe_mean(velocities),
        "std_velocity": safe_std(velocities),
        "pitch_std": safe_std(pitches),
        "tone_span": float(max(pitches) - min(pitches)),
        "pitch_class_entropy": pitch_class_entropy_from_hist(hist),
        "consecutive_pitch_repetition_ratio": consecutive_pitch_repetition_ratio(onset_pitches, 3),
        "empty_bar_ratio": empty_bar_ratio(pm, notes),
        "scale_consistency": scale_consistency(notes),
        "repetition_rate": repetition_rate_ngram(onset_pitches, 3),
    }
    for i in range(12):
        result[f"pch_{i}"] = float(hist[i])
    return result


# =============================================================================
# REFERENCE-BASED
# =============================================================================
def select_local_reference_pool(gen_duration: float, ref_df: pd.DataFrame) -> pd.DataFrame:
    if ref_df.empty:
        return ref_df
    low = gen_duration * (1.0 - LOCAL_DURATION_TOL)
    high = gen_duration * (1.0 + LOCAL_DURATION_TOL)
    pool = ref_df[(ref_df["duration_s"] >= low) & (ref_df["duration_s"] <= high)]
    if len(pool) >= LOCAL_REF_MIN:
        pool = pool.copy()
        pool["dur_dist"] = (pool["duration_s"] - gen_duration).abs()
        return pool.sort_values("dur_dist").head(LOCAL_REF_K).drop(columns=["dur_dist"])
    pool = ref_df.copy()
    pool["dur_dist"] = (pool["duration_s"] - gen_duration).abs()
    return pool.sort_values("dur_dist").head(max(LOCAL_REF_MIN, LOCAL_REF_K)).drop(columns=["dur_dist"])


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(p.sum(), EPS)
    q = q / max(q.sum(), EPS)
    m = 0.5 * (p + q)
    def kld(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2((a[mask] + EPS) / (b[mask] + EPS))))
    return 0.5 * kld(p, m) + 0.5 * kld(q, m)


def pitch_hist_similarity_score(gen_row: pd.Series, ref_pool: pd.DataFrame) -> float:
    gen_hist = np.array([gen_row[f"pch_{i}"] for i in range(12)], dtype=np.float64)
    ref_hist = ref_pool[[f"pch_{i}" for i in range(12)]].mean(axis=0).to_numpy(dtype=np.float64)
    jsd = js_divergence(gen_hist, ref_hist)
    return 100.0 * (1.0 - clip01(jsd / 1.0))


def per_piece_reference_based(gen_row: pd.Series, ref_pool: pd.DataFrame) -> Dict[str, object]:
    detail: Dict[str, Dict[str, float]] = {}
    feature_scores: List[Tuple[float, float]] = []

    for feat in REFERENCE_BASED_FEATURES:
        mu = float(ref_pool[feat].mean())
        sigma = float(ref_pool[feat].std(ddof=0))
        x = float(gen_row[feat])
        z = abs(x - mu) / max(sigma, 1e-6)
        score = score_exp_from_z(z)
        weight = REFERENCE_BASED_WEIGHTS.get(feat, 1.0)
        feature_scores.append((score, weight))
        detail[feat] = {
            "value": x,
            "ref_mean": mu,
            "ref_std": sigma,
            "z_abs": float(z),
            "score": float(score),
        }

    hist_score = pitch_hist_similarity_score(gen_row, ref_pool)
    feature_scores.append((hist_score, 1.2))

    overall = weighted_average(feature_scores)
    return {
        "matched_reference_count": int(len(ref_pool)),
        "pitch_hist_similarity_score": float(hist_score),
        "score": float(overall),
        "details": detail,
    }


# =============================================================================
# REFERENCE-FREE
# =============================================================================
def band_score(x: float, low_bad: float, low_good: float, high_good: float, high_bad: float) -> float:
    """Puntuación trapezoidal 0..100."""
    if x <= low_bad or x >= high_bad:
        return 0.0
    if low_good <= x <= high_good:
        return 100.0
    if x < low_good:
        return 100.0 * (x - low_bad) / max(low_good - low_bad, EPS)
    return 100.0 * (high_bad - x) / max(high_bad - high_good, EPS)


def descending_score(x: float, good: float, bad: float) -> float:
    if x <= good:
        return 100.0
    if x >= bad:
        return 0.0
    return 100.0 * (bad - x) / max(bad - good, EPS)


def ascending_score(x: float, bad: float, good: float) -> float:
    if x <= bad:
        return 0.0
    if x >= good:
        return 100.0
    return 100.0 * (x - bad) / max(good - bad, EPS)


def per_piece_reference_free(gen_row: pd.Series) -> Dict[str, object]:
    sub = {
        "pitch_class_entropy": band_score(float(gen_row["pitch_class_entropy"]), 1.5, 2.5, 3.6, 4.2),
        "consecutive_pitch_repetition_ratio": descending_score(float(gen_row["consecutive_pitch_repetition_ratio"]), 0.01, 0.15),
        "empty_bar_ratio": descending_score(float(gen_row["empty_bar_ratio"]), 0.05, 0.60),
        "scale_consistency": ascending_score(float(gen_row["scale_consistency"]), 0.45, 0.85),
        "repetition_rate": band_score(float(gen_row["repetition_rate"]), 0.0, 0.03, 0.35, 0.80),
        "polyphony": band_score(float(gen_row["polyphony"]), 0.7, 1.2, 5.5, 8.5),
        "tone_span": band_score(float(gen_row["tone_span"]), 18.0, 30.0, 84.0, 96.0),
        "onset_density": band_score(float(gen_row["onset_density"]), 0.5, 2.0, 15.0, 25.0),
        "qualified_note_ratio": band_score(float(gen_row["qualified_note_ratio"]), 0.02, 0.10, 0.95, 1.01),
        "std_velocity": band_score(float(gen_row["std_velocity"]), 0.0, 4.0, 22.0, 35.0),
    }
    overall = weighted_average((score, REFERENCE_FREE_WEIGHTS[name]) for name, score in sub.items())
    return {"score": float(overall), "details": {k: float(v) for k, v in sub.items()}}


# =============================================================================
# OA / KLD GLOBALES
# =============================================================================
def overlapping_area(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.minimum(a, b).sum())


def kld_real_to_gen(a: np.ndarray, b: np.ndarray) -> float:
    a = a + EPS
    b = b + EPS
    a = a / a.sum()
    b = b / b.sum()
    return float(np.sum(a * np.log(a / b)))


def hist_pair(real_values: np.ndarray, gen_values: np.ndarray, n_bins: int = HIST_BINS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_values = np.concatenate([real_values, gen_values]) if len(gen_values) else real_values
    lo = float(np.min(all_values))
    hi = float(np.max(all_values))
    if math.isclose(lo, hi):
        hi = lo + 1.0
    bins = np.linspace(lo, hi, n_bins + 1)
    real_hist, edges = np.histogram(real_values, bins=bins, density=True)
    gen_hist, _ = np.histogram(gen_values, bins=edges, density=True)
    widths = np.diff(edges)
    real_prob = real_hist * widths
    gen_prob = gen_hist * widths
    real_prob = real_prob / max(real_prob.sum(), EPS)
    gen_prob = gen_prob / max(gen_prob.sum(), EPS)
    return real_prob, gen_prob, edges


def global_distribution_report(ref_df: pd.DataFrame, gen_df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    rows = []
    for feat in features:
        real = ref_df[feat].to_numpy(dtype=np.float64)
        gen = gen_df[feat].to_numpy(dtype=np.float64)
        real_prob, gen_prob, edges = hist_pair(real, gen, HIST_BINS)
        rows.append({
            "feature": feat,
            "oa": overlapping_area(real_prob, gen_prob),
            "kld_real_to_gen": kld_real_to_gen(real_prob, gen_prob),
            "real_mean": float(np.mean(real)),
            "gen_mean": float(np.mean(gen)) if len(gen) else 0.0,
            "real_std": float(np.std(real)),
            "gen_std": float(np.std(gen)) if len(gen) else 0.0,
            "n_real": int(len(real)),
            "n_gen": int(len(gen)),
            "n_bins": int(len(edges) - 1),
        })
    return pd.DataFrame(rows)


# =============================================================================
# INTERPRETACIÓN CUALITATIVA
# =============================================================================
def describe_piece(ref_based: Dict[str, object], ref_free: Dict[str, object]) -> Tuple[List[str], List[str]]:
    strengths: List[str] = []
    issues: List[str] = []

    rb_details = ref_based["details"]
    rf_details = ref_free["details"]

    sorted_rb = sorted(rb_details.items(), key=lambda kv: kv[1]["score"], reverse=True)
    sorted_rf = sorted(rf_details.items(), key=lambda kv: kv[1], reverse=True)
    for feat, item in sorted_rb[:2]:
        if item["score"] >= 70:
            strengths.append(f"Cercanía razonable a referencia en {feat}.")
    for feat, score in sorted_rf[:2]:
        if score >= 70:
            strengths.append(f"Buen comportamiento interno en {feat}.")

    sorted_rb_bad = sorted(rb_details.items(), key=lambda kv: kv[1]["score"])
    sorted_rf_bad = sorted(rf_details.items(), key=lambda kv: kv[1])
    for feat, item in sorted_rb_bad[:2]:
        if item["score"] < 50:
            issues.append(f"Se aleja de la referencia en {feat}.")
    for feat, score in sorted_rf_bad[:2]:
        if score < 50:
            issues.append(f"Señal interna débil en {feat}.")

    strengths = list(dict.fromkeys(strengths))[:3]
    issues = list(dict.fromkeys(issues))[:3]
    return strengths, issues


# =============================================================================
# MAIN
# =============================================================================
def build_feature_table(files: Sequence[Path], label: str) -> pd.DataFrame:
    rows = []
    for i, path in enumerate(files, start=1):
        try:
            row = extract_features(path)
            row["dataset"] = label
            rows.append(row)
        except Exception as e:
            print(f"[WARN] Saltando {path.name}: {e}")
        if i % 50 == 0 or i == len(files):
            print(f"[{label}] procesados {i}/{len(files)}")
    return pd.DataFrame(rows)


def main() -> None:
    print("[INFO] Iniciando evaluación symbolic MIDI con división reference-based / reference-free")
    print(f"[INFO] GENERATED_DIR = {GENERATED_DIR}")
    print(f"[INFO] REFERENCE_DIR = {REFERENCE_DIR}")
    print(f"[INFO] OUT_DIR       = {OUT_DIR}")
    print(f"[INFO] PIANO_ONLY    = {PIANO_ONLY}")

    gen_files = find_midi_files(GENERATED_DIR, RECURSIVE)
    ref_files = find_midi_files(REFERENCE_DIR, RECURSIVE)

    gen_df = build_feature_table(gen_files, "generated")
    ref_df = build_feature_table(ref_files, "reference")

    if gen_df.empty:
        raise RuntimeError("No se pudo evaluar ninguna pieza generada.")
    if ref_df.empty:
        raise RuntimeError("No se pudo evaluar ninguna pieza de referencia.")

    per_piece_rows = []
    per_piece_details = []

    for _, gen_row in gen_df.iterrows():
        local_pool = select_local_reference_pool(float(gen_row["duration_s"]), ref_df)
        ref_based = per_piece_reference_based(gen_row, local_pool)
        ref_free = per_piece_reference_free(gen_row)
        overall = W_REFERENCE_BASED * float(ref_based["score"]) + W_REFERENCE_FREE * float(ref_free["score"])
        label = piece_label(overall)
        strengths, issues = describe_piece(ref_based, ref_free)

        piece_file = str(gen_row["file"])

        # CSV limpio: solo columnas escalares / texto simple
        per_piece_rows.append({
            "file": piece_file,
            "duration_s": float(gen_row["duration_s"]),
            "n_notes": float(gen_row["n_notes"]),
            "reference_based_score": float(ref_based["score"]),
            "reference_based_pitch_hist_similarity": float(ref_based["pitch_hist_similarity_score"]),
            "reference_based_matched_reference_count": int(ref_based["matched_reference_count"]),
            "reference_free_score": float(ref_free["score"]),
            "global_score": float(overall),
            "qualitative_label": label,
            "strengths": " | ".join(strengths),
            "issues": " | ".join(issues),
        })

        # Detalles completos en JSON aparte
        per_piece_details.append({
            "file": piece_file,
            "reference_based": ref_based,
            "reference_free": ref_free,
            "global_score": float(overall),
            "qualitative_label": label,
            "strengths": strengths,
            "issues": issues,
        })

    PER_PIECE_COLUMNS = [
        "file",
        "duration_s",
        "n_notes",
        "reference_based_score",
        "reference_based_pitch_hist_similarity",
        "reference_based_matched_reference_count",
        "reference_free_score",
        "global_score",
        "qualitative_label",
        "strengths",
        "issues",
    ]

    per_piece_df = pd.DataFrame(per_piece_rows)
    per_piece_df = per_piece_df.reindex(columns=PER_PIECE_COLUMNS)

    if not per_piece_df.empty:
        per_piece_df = per_piece_df.sort_values("global_score", ascending=False).reset_index(drop=True)

    global_features = [
        "pitch_range", "used_pitch_classes", "avg_pitch_interval", "polyphony",
        "qualified_note_ratio", "onset_density", "mean_duration", "std_duration",
        "mean_velocity", "std_velocity", "pitch_std", "tone_span",
        "pitch_class_entropy", "consecutive_pitch_repetition_ratio", "empty_bar_ratio",
        "scale_consistency", "repetition_rate"
    ]
    global_report = global_distribution_report(ref_df, gen_df, global_features)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    generated_csv = OUT_DIR / "generated_features.csv"
    reference_csv = OUT_DIR / "reference_features.csv"
    per_piece_csv = OUT_DIR / "per_piece_evaluation.csv"
    global_csv = OUT_DIR / "global_reference_based_report.csv"
    per_piece_xlsx = OUT_DIR / "per_piece_evaluation.xlsx"
    per_piece_details_json = OUT_DIR / "per_piece_details.json"

    for path in [generated_csv, reference_csv, per_piece_csv, global_csv, per_piece_xlsx, per_piece_details_json]:
        if path.exists():
            path.unlink()

    gen_df.to_csv(generated_csv, index=False, header=True, mode="w", encoding="utf-8-sig")
    ref_df.to_csv(reference_csv, index=False, header=True, mode="w", encoding="utf-8-sig")
    per_piece_df.to_csv(per_piece_csv, index=False, header=True, mode="w", encoding="utf-8-sig")
    global_report.to_csv(global_csv, index=False, header=True, mode="w", encoding="utf-8-sig")

    try:
        per_piece_df.to_excel(per_piece_xlsx, index=False)
    except Exception:
        pass

    with open(per_piece_details_json, "w", encoding="utf-8") as f:
        json.dump(per_piece_details, f, ensure_ascii=False, indent=2)

    summary = {
        "generated_dir": str(GENERATED_DIR),
        "reference_dir": str(REFERENCE_DIR),
        "n_generated_files": int(len(gen_df)),
        "n_reference_files": int(len(ref_df)),
        "piano_only": bool(PIANO_ONLY),
        "reference_based_features": REFERENCE_BASED_FEATURES,
        "reference_free_features": list(REFERENCE_FREE_WEIGHTS.keys()),
        "global_score_mean": float(per_piece_df["global_score"].mean()),
        "global_score_std": float(per_piece_df["global_score"].std(ddof=0)),
        "label_counts": per_piece_df["qualitative_label"].value_counts().to_dict(),
        "best_pieces": per_piece_df[["file", "global_score", "qualitative_label"]].head(5).to_dict(orient="records"),
        "worst_pieces": per_piece_df[["file", "global_score", "qualitative_label"]].tail(5).to_dict(orient="records"),
        "best_global_features_by_oa": global_report.sort_values("oa", ascending=False).head(5).to_dict(orient="records"),
        "worst_global_features_by_oa": global_report.sort_values("oa", ascending=True).head(5).to_dict(orient="records"),
        "lowest_global_kld": global_report.sort_values("kld_real_to_gen", ascending=True).head(5).to_dict(orient="records"),
        "highest_global_kld": global_report.sort_values("kld_real_to_gen", ascending=False).head(5).to_dict(orient="records"),
    }
    with open(OUT_DIR / "summary_refsplit.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[OK] Evaluación terminada.")
    print(f"[OK] CSV por pieza: {OUT_DIR / 'per_piece_evaluation.csv'}")
    print(f"[OK] Informe global: {OUT_DIR / 'global_reference_based_report.csv'}")
    print(f"[OK] Resumen JSON:  {OUT_DIR / 'summary_refsplit.json'}")


if __name__ == "__main__":
    main()
