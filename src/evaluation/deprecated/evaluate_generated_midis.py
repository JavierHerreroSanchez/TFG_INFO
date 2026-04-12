from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import pretty_midi
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "No se pudo importar pretty_midi. Instálalo en tu entorno de PyCharm con: pip install pretty_midi"
    ) from e

# =============================================================================
# CONFIGURACIÓN PARA PYCHARM
# -----------------------------------------------------------------------------
# Edita solo estas rutas y opciones, y ejecuta el script con el botón Run.
# No necesita argumentos por terminal.
# =============================================================================
GENERATED_DIR = Path(r"/output/generation_v2")
REFERENCE_DIR = Path(r"/data/pretraining_raw/maestro-v3.0.0")
OUT_DIR = Path(r"/output/midi_eval")

INCLUDE_DRUMS = False
RECURSIVE = True
QUALIFIED_NOTE_THRESHOLD_S = 0.125

MIDI_SUFFIXES = {".mid", ".midi"}
EPS = 1e-12


@dataclass
class NoteEvent:
    pitch: int
    start: float
    end: float
    velocity: int
    instrument_program: int
    is_drum: bool

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def find_midi_files(root: Path, recursive: bool = True) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"No existe el directorio: {root}")
    globber = root.rglob if recursive else root.glob
    files = sorted(p for p in globber("*") if p.suffix.lower() in MIDI_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No se encontraron MIDIs en: {root}")
    return files


def load_pretty_midi(path: Path) -> pretty_midi.PrettyMIDI:
    try:
        return pretty_midi.PrettyMIDI(str(path))
    except Exception as e:
        raise RuntimeError(f"No se pudo abrir el MIDI: {path}") from e


def collect_notes(pm: pretty_midi.PrettyMIDI, include_drums: bool = False) -> List[NoteEvent]:
    notes: List[NoteEvent] = []
    for inst in pm.instruments:
        if inst.is_drum and not include_drums:
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
                    instrument_program=int(inst.program),
                    is_drum=bool(inst.is_drum),
                )
            )
    notes.sort(key=lambda x: (x.start, x.pitch, x.end))
    return notes


def safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def safe_std(values: Sequence[float]) -> float:
    return float(np.std(values)) if values else 0.0


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


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
    weighted_sum = 0.0
    active_time = 0.0

    for t, delta in events:
        dt = t - prev_t
        if dt > 0 and active > 0:
            weighted_sum += active * dt
            active_time += dt
        active += delta
        prev_t = t

    return weighted_sum / active_time if active_time > 0 else 0.0


def repetition_ratio(sorted_pitches: Sequence[int], n: int = 3) -> float:
    if len(sorted_pitches) < n:
        return 0.0

    marked = np.zeros(len(sorted_pitches), dtype=np.float64)
    start = 0
    while start < len(sorted_pitches):
        end = start + 1
        while end < len(sorted_pitches) and sorted_pitches[end] == sorted_pitches[start]:
            end += 1
        run_len = end - start
        if run_len >= n:
            marked[start:end] = 1.0
        start = end
    return float(marked.mean())


def onset_density(notes: Sequence[NoteEvent], total_duration: float) -> float:
    if total_duration <= 0:
        return 0.0
    return len(notes) / total_duration


def pitch_class_histogram(notes: Sequence[NoteEvent]) -> np.ndarray:
    hist = np.zeros(12, dtype=np.float64)
    for n in notes:
        hist[n.pitch % 12] += 1.0
    s = hist.sum()
    return hist / s if s > 0 else hist


def pitch_class_entropy(notes: Sequence[NoteEvent]) -> float:
    hist = pitch_class_histogram(notes)
    mask = hist > 0
    return float(-(hist[mask] * np.log2(hist[mask])).sum()) if mask.any() else 0.0


def duration_buckets(durations: Sequence[float], bucket_ms: int = 50) -> List[float]:
    step = bucket_ms / 1000.0
    return [round(d / step) * step for d in durations]


def extract_basic_metrics(path: Path, include_drums: bool = False, qn_threshold_s: float = 0.125) -> Dict[str, float]:
    pm = load_pretty_midi(path)
    notes = collect_notes(pm, include_drums=include_drums)

    if not notes:
        return {
            "file": str(path),
            "n_notes": 0,
            "duration_s": float(pm.get_end_time()),
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
            "consecutive_pitch_repetition_ratio": 0.0,
        }

    total_duration = float(max(pm.get_end_time(), max(n.end for n in notes)))
    pitches = [n.pitch for n in notes]
    durations = [n.duration for n in notes]
    velocities = [n.velocity for n in notes]
    start_sorted_notes = sorted(notes, key=lambda n: (n.start, n.pitch, n.end))
    onset_sorted_pitches = [n.pitch for n in start_sorted_notes]

    intervals = [
        abs(onset_sorted_pitches[i] - onset_sorted_pitches[i - 1])
        for i in range(1, len(onset_sorted_pitches))
    ]

    unique_duration_count = len(set(duration_buckets(durations, bucket_ms=50)))
    qn_ratio = sum(1 for d in durations if d >= qn_threshold_s) / len(durations)

    return {
        "file": str(path),
        "n_notes": float(len(notes)),
        "duration_s": total_duration,
        "pitch_range": float(max(pitches) - min(pitches)) if pitches else 0.0,
        "used_pitch_classes": float(len({p % 12 for p in pitches})),
        "avg_pitch_interval": safe_mean(intervals),
        "unique_pitches": float(len(set(pitches))),
        "unique_durations": float(unique_duration_count),
        "polyphony": average_polyphony(notes),
        "qualified_note_ratio": clamp01(qn_ratio),
        "onset_density": onset_density(notes, total_duration),
        "mean_duration": safe_mean(durations),
        "std_duration": safe_std(durations),
        "mean_velocity": safe_mean(velocities),
        "std_velocity": safe_std(velocities),
        "pitch_std": safe_std(pitches),
        "tone_span": float(max(pitches) - min(pitches)) if pitches else 0.0,
        "pitch_class_entropy": pitch_class_entropy(notes),
        "consecutive_pitch_repetition_ratio": repetition_ratio(onset_sorted_pitches, n=3),
    }


def freedman_diaconis_bin_count(values: np.ndarray, minimum_bins: int = 10, maximum_bins: int = 60) -> int:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return minimum_bins
    q75, q25 = np.percentile(values, [75, 25])
    iqr = q75 - q25
    if iqr <= 0:
        return min(maximum_bins, max(minimum_bins, int(round(np.sqrt(values.size)))))
    bin_width = 2 * iqr / np.cbrt(values.size)
    if bin_width <= 0:
        return minimum_bins
    n_bins = int(math.ceil((values.max() - values.min()) / bin_width)) if values.max() > values.min() else minimum_bins
    return int(min(maximum_bins, max(minimum_bins, n_bins)))


def build_common_bins(real_values: np.ndarray, gen_values: np.ndarray, feature_name: str) -> np.ndarray:
    both = np.concatenate([real_values, gen_values]).astype(np.float64)
    both = both[np.isfinite(both)]
    if both.size == 0:
        return np.array([0.0, 1.0], dtype=np.float64)

    vmin, vmax = float(both.min()), float(both.max())
    if math.isclose(vmin, vmax):
        return np.array([vmin - 0.5, vmax + 0.5], dtype=np.float64)

    discrete_like = {
        "n_notes", "pitch_range", "used_pitch_classes", "unique_pitches",
        "unique_durations", "tone_span"
    }
    if feature_name in discrete_like:
        start = math.floor(vmin) - 0.5
        end = math.ceil(vmax) + 0.5
        return np.arange(start, end + 1.0, 1.0, dtype=np.float64)

    n_bins = freedman_diaconis_bin_count(both)
    return np.linspace(vmin, vmax, num=n_bins + 1, dtype=np.float64)


def histogram_density(values: np.ndarray, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(values, bins=bins, density=False)
    hist = hist.astype(np.float64)
    widths = np.diff(edges)
    total_mass = hist.sum()
    if total_mass <= 0:
        density = np.zeros_like(hist, dtype=np.float64)
    else:
        density = hist / (total_mass * widths)
    return density, widths


def overlapping_area(real_values: np.ndarray, gen_values: np.ndarray, bins: np.ndarray) -> float:
    p, widths = histogram_density(real_values, bins)
    q, _ = histogram_density(gen_values, bins)
    return float(np.sum(np.minimum(p, q) * widths))


def kld(real_values: np.ndarray, gen_values: np.ndarray, bins: np.ndarray, eps: float = EPS) -> float:
    p_counts, _ = np.histogram(real_values, bins=bins)
    q_counts, _ = np.histogram(gen_values, bins=bins)
    p = p_counts.astype(np.float64) + eps
    q = q_counts.astype(np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


FEATURE_COLUMNS = [
    "n_notes",
    "duration_s",
    "pitch_range",
    "used_pitch_classes",
    "avg_pitch_interval",
    "unique_pitches",
    "unique_durations",
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
]


def compare_feature_distributions(real_df: pd.DataFrame, gen_df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        real_values = real_df[feature].to_numpy(dtype=np.float64)
        gen_values = gen_df[feature].to_numpy(dtype=np.float64)
        bins = build_common_bins(real_values, gen_values, feature)
        rows.append(
            {
                "feature": feature,
                "oa": overlapping_area(real_values, gen_values, bins),
                "kld_real_to_gen": kld(real_values, gen_values, bins),
                "real_mean": float(np.mean(real_values)) if len(real_values) else 0.0,
                "gen_mean": float(np.mean(gen_values)) if len(gen_values) else 0.0,
                "real_std": float(np.std(real_values)) if len(real_values) else 0.0,
                "gen_std": float(np.std(gen_values)) if len(gen_values) else 0.0,
                "n_real": int(len(real_values)),
                "n_gen": int(len(gen_values)),
                "n_bins": int(len(bins) - 1),
            }
        )
    out = pd.DataFrame(rows).sort_values(["oa", "kld_real_to_gen"], ascending=[False, True])
    return out.reset_index(drop=True)


def evaluate_directory(midi_dir: Path, include_drums: bool = False, recursive: bool = True) -> pd.DataFrame:
    rows = []
    for path in find_midi_files(midi_dir, recursive=recursive):
        try:
            rows.append(extract_basic_metrics(path, include_drums=include_drums, qn_threshold_s=QUALIFIED_NOTE_THRESHOLD_S))
        except Exception as e:
            rows.append({"file": str(path), "error": str(e)})
    return pd.DataFrame(rows)


def dataset_summary(df: pd.DataFrame, features: Sequence[str]) -> Dict[str, Dict[str, float]]:
    summary = {}
    valid = df.dropna(subset=[f for f in features if f in df.columns])
    for feature in features:
        vals = valid[feature].to_numpy(dtype=np.float64)
        summary[feature] = {
            "mean": float(np.mean(vals)) if len(vals) else 0.0,
            "std": float(np.std(vals)) if len(vals) else 0.0,
            "median": float(np.median(vals)) if len(vals) else 0.0,
            "min": float(np.min(vals)) if len(vals) else 0.0,
            "max": float(np.max(vals)) if len(vals) else 0.0,
        }
    return summary


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_outputs(
    out_dir: Path,
    gen_df: pd.DataFrame,
    real_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    generated_dir: Path,
    reference_dir: Path,
) -> None:
    ensure_output_dir(out_dir)
    gen_csv = out_dir / "generated_metrics.csv"
    real_csv = out_dir / "reference_metrics.csv"
    cmp_csv = out_dir / "distribution_comparison.csv"
    summary_json = out_dir / "summary.json"

    gen_df.to_csv(gen_csv, index=False)
    real_df.to_csv(real_csv, index=False)
    comparison_df.to_csv(cmp_csv, index=False)

    payload = {
        "generated_dir": str(generated_dir),
        "reference_dir": str(reference_dir),
        "n_generated_files": int(len(gen_df)),
        "n_reference_files": int(len(real_df)),
        "feature_columns": FEATURE_COLUMNS,
        "generated_summary": dataset_summary(gen_df.drop(columns=["error"], errors="ignore"), FEATURE_COLUMNS),
        "reference_summary": dataset_summary(real_df.drop(columns=["error"], errors="ignore"), FEATURE_COLUMNS),
        "best_aligned_features_by_oa": comparison_df.sort_values("oa", ascending=False).head(5).to_dict(orient="records"),
        "worst_aligned_features_by_oa": comparison_df.sort_values("oa", ascending=True).head(5).to_dict(orient="records"),
        "lowest_kld_features": comparison_df.sort_values("kld_real_to_gen", ascending=True).head(5).to_dict(orient="records"),
        "highest_kld_features": comparison_df.sort_values("kld_real_to_gen", ascending=False).head(5).to_dict(orient="records"),
    }
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] Guardado: {gen_csv}")
    print(f"[OK] Guardado: {real_csv}")
    print(f"[OK] Guardado: {cmp_csv}")
    print(f"[OK] Guardado: {summary_json}")


def main() -> None:
    print("[INFO] Configuración PyCharm")
    print(f"[INFO] GENERATED_DIR = {GENERATED_DIR}")
    print(f"[INFO] REFERENCE_DIR = {REFERENCE_DIR}")
    print(f"[INFO] OUT_DIR = {OUT_DIR}")
    print(f"[INFO] INCLUDE_DRUMS = {INCLUDE_DRUMS}")
    print(f"[INFO] RECURSIVE = {RECURSIVE}")
    print(f"[INFO] QUALIFIED_NOTE_THRESHOLD_S = {QUALIFIED_NOTE_THRESHOLD_S}")

    print("\n[INFO] Evaluando MIDIs generados...")
    gen_df = evaluate_directory(GENERATED_DIR, include_drums=INCLUDE_DRUMS, recursive=RECURSIVE)
    print("[INFO] Evaluando corpus de referencia...")
    real_df = evaluate_directory(REFERENCE_DIR, include_drums=INCLUDE_DRUMS, recursive=RECURSIVE)

    if "error" in gen_df.columns:
        n_err = int(gen_df["error"].notna().sum())
        if n_err > 0:
            print(f"[WARN] Generados con error: {n_err}")
    if "error" in real_df.columns:
        n_err = int(real_df["error"].notna().sum())
        if n_err > 0:
            print(f"[WARN] Referencia con error: {n_err}")

    gen_ok = gen_df[~gen_df.get("error", pd.Series([False] * len(gen_df))).notna()].copy() if "error" in gen_df.columns else gen_df.copy()
    real_ok = real_df[~real_df.get("error", pd.Series([False] * len(real_df))).notna()].copy() if "error" in real_df.columns else real_df.copy()

    if gen_ok.empty:
        raise RuntimeError("No hay MIDIs generados válidos para evaluar.")
    if real_ok.empty:
        raise RuntimeError("No hay MIDIs de referencia válidos para evaluar.")

    comparison_df = compare_feature_distributions(real_ok, gen_ok, FEATURE_COLUMNS)
    save_outputs(OUT_DIR, gen_df, real_df, comparison_df, GENERATED_DIR, REFERENCE_DIR)

    print("\n=== Resumen rápido ===")
    print(comparison_df[["feature", "oa", "kld_real_to_gen", "real_mean", "gen_mean"]].to_string(index=False))
    print("\nInterpretación: OA más alto y KLD más bajo indican mayor parecido distribucional.")


if __name__ == "__main__":
    main()
