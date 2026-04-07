from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy

# ============================================================
# CONFIGURACION
# ============================================================
GENERATED_DIR = Path(r"/output/generation_v2")
REFERENCE_DIR = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\data\raw\ariamidi\aa")
OUT_DIR = Path(r"/output/midi_eval")

RECURSIVE = True
PIANO_ONLY = True
IGNORE_DRUMS = True

# Pool local de referencias para score por pieza
LOCAL_REF_POOL_SIZE = 64
LOCAL_REF_DURATION_TOL = 0.35  # +/-35%

# Pesos score final
W_REFERENCE_BASED = 0.65
W_REFERENCE_FREE = 0.35

# Cualitativos
QUAL_LABELS = [
    (85.0, "muy plausible"),
    (70.0, "plausible"),
    (55.0, "aceptable con anomalías"),
    (40.0, "débil"),
    (-1.0, "fuera de distribución"),
]

# Resolución para MusPy cuando haga falta cuantizar MIDI
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
    "drum_pattern_consistency",
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
# IMPORTS DE LIBRERIAS DE DOMINIO
# ============================================================
try:
    import muspy
except Exception as e:
    raise RuntimeError(
        "No se pudo importar muspy. Instala con: pip install muspy"
    ) from e


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


# ============================================================
# CONVERSION Y FILTRADO MUSPY
# ============================================================
def load_music(path: Path) -> muspy.Music:
    music = muspy.read_midi(path)
    if music.resolution is None:
        music.resolution = MUSPY_RESOLUTION
    return music


def keep_only_piano_tracks(music: muspy.Music) -> muspy.Music:
    if not PIANO_ONLY:
        return music
    piano_programs = set(range(0, 8))
    kept = []
    for track in music.tracks:
        is_drum = bool(getattr(track, "is_drum", False))
        program = getattr(track, "program", 0)
        if IGNORE_DRUMS and is_drum:
            continue
        if program in piano_programs:
            kept.append(track)
    music.tracks = kept
    return music


# ============================================================
# FEATURES CUSTOM MINIMAS
# ============================================================
def iter_notes(music: muspy.Music):
    for track in music.tracks:
        if IGNORE_DRUMS and getattr(track, "is_drum", False):
            continue
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
        "drum_pattern_consistency": muspy_metric_safe("drum_pattern_consistency", music),
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
# OA / KLD CON SCIPY + HISTOGRAMAS
# ============================================================
def normalized_hist_pair(a: np.ndarray, b: np.ndarray, bins: int | str = "auto") -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.asarray([]), np.asarray([])

    lo = min(np.min(a), np.min(b))
    hi = max(np.max(a), np.max(b))
    if lo == hi:
        lo -= 0.5
        hi += 0.5

    # máximo razonable para no romper con datasets pequeños
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
        p, q = normalized_hist_pair(ref_df[feat].to_numpy(), gen_df[feat].to_numpy())
        rows.append({
            "feature": feat,
            "oa": overlap_area(p, q),
            "kld_real_to_gen": kld_real_to_gen(p, q),
            "real_mean": safe_float(np.nanmean(ref_df[feat].to_numpy(dtype=float))),
            "gen_mean": safe_float(np.nanmean(gen_df[feat].to_numpy(dtype=float))),
            "real_std": safe_float(np.nanstd(ref_df[feat].to_numpy(dtype=float))),
            "gen_std": safe_float(np.nanstd(gen_df[feat].to_numpy(dtype=float))),
            "n_real": int(np.isfinite(ref_df[feat].to_numpy(dtype=float)).sum()),
            "n_gen": int(np.isfinite(gen_df[feat].to_numpy(dtype=float)).sum()),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["oa", "kld_real_to_gen"], ascending=[False, True]).reset_index(drop=True)
    return out


# ============================================================
# SCORE POR PIEZA: reference-based
# ============================================================
def select_local_reference_pool(duration_beats: float, ref_df: pd.DataFrame) -> pd.DataFrame:
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
        pool = pool.head(LOCAL_REF_POOL_SIZE)
    return pool


def pitch_hist_similarity_from_json(hist_json_a: str, hist_json_b: str) -> float:
    a = np.asarray(json.loads(hist_json_a), dtype=float)
    b = np.asarray(json.loads(hist_json_b), dtype=float)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    a = a / a.sum()
    b = b / b.sum()
    return float(np.minimum(a, b).sum())


def robust_feature_score(x: float, mu: float, sigma: float, eps: float = 1e-8) -> float:
    if not (np.isfinite(x) and np.isfinite(mu) and np.isfinite(sigma)):
        return np.nan
    sigma = max(float(sigma), eps)
    z = abs(float(x) - float(mu)) / sigma
    return float(math.exp(-z))


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
        mu = safe_float(np.nanmean(local_ref_df[feat].to_numpy(dtype=float)))
        sigma = safe_float(np.nanstd(local_ref_df[feat].to_numpy(dtype=float)))
        x = safe_float(gen_row.get(feat, np.nan))
        s = robust_feature_score(x, mu, sigma)
        detail_scores[feat] = {
            "x": x,
            "mu_ref": mu,
            "sigma_ref": sigma,
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
    if np.isfinite(pitch_hist_similarity_score):
        score = 0.8 * base_score + 20.0 * 0.2 * pitch_hist_similarity_score
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
def bounded_score_high_good(x: float, lo: float, hi: float) -> float:
    if not np.isfinite(x):
        return np.nan
    if x < lo:
        return max(0.0, 1.0 - (lo - x) / max(abs(lo), 1.0))
    if x > hi:
        return max(0.0, 1.0 - (x - hi) / max(abs(hi), 1.0))
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    if half <= 0:
        return 1.0
    return float(max(0.0, 1.0 - abs(x - center) / half))


def per_piece_reference_free(gen_row: pd.Series) -> Dict:
    checks = {
        "pitch_class_entropy": bounded_score_high_good(safe_float(gen_row.get("pitch_class_entropy", np.nan)), 2.6, 3.6),
        "consecutive_pitch_repetition_ratio_custom": bounded_score_high_good(1.0 - safe_float(gen_row.get("consecutive_pitch_repetition_ratio_custom", np.nan)), 0.90, 1.0),
        "empty_measure_rate": bounded_score_high_good(1.0 - safe_float(gen_row.get("empty_measure_rate", np.nan)), 0.65, 1.0),
        "scale_consistency": bounded_score_high_good(safe_float(gen_row.get("scale_consistency", np.nan)), 0.6, 1.0),
        "polyphony": bounded_score_high_good(safe_float(gen_row.get("polyphony", np.nan)), 1.3, 5.5),
        "std_velocity_custom": bounded_score_high_good(safe_float(gen_row.get("std_velocity_custom", np.nan)), 3.0, 28.0),
        "groove_consistency": bounded_score_high_good(safe_float(gen_row.get("groove_consistency", np.nan)), 0.2, 1.0),
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
            if s >= 0.75:
                strengths.append(f"{feat} cercano a referencia")
            elif s <= 0.35:
                issues.append(f"{feat} alejado de referencia")

    phs = safe_float(ref_based.get("pitch_hist_similarity_score", np.nan))
    if np.isfinite(phs):
        if phs >= 0.8:
            strengths.append("histograma de pitch muy alineado")
        elif phs <= 0.45:
            issues.append("histograma de pitch poco alineado")

    for feat in ["pitch_class_entropy", "scale_consistency", "groove_consistency"]:
        s = safe_float(details_rf.get(feat, np.nan))
        if np.isfinite(s):
            if s >= 0.75:
                strengths.append(f"{feat} sólido")
            elif s <= 0.35:
                issues.append(f"{feat} problemático")

    cpr = safe_float(details_rf.get("consecutive_pitch_repetition_ratio_custom", np.nan))
    if np.isfinite(cpr) and cpr <= 0.35:
        issues.append("exceso de repetición consecutiva")

    # recorta duplicados y longitud
    strengths = list(dict.fromkeys(strengths))[:3]
    issues = list(dict.fromkeys(issues))[:3]
    return strengths, issues


# ============================================================
# MAIN
# ============================================================
def main():
    print("[INFO] Iniciando evaluación symbolic MIDI con librerías (MusPy + SciPy)")
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

    global_report = global_distribution_report(ref_df, gen_df, GLOBAL_FEATURES)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    generated_csv = OUT_DIR / "generated_features.csv"
    reference_csv = OUT_DIR / "reference_features.csv"
    per_piece_csv = OUT_DIR / "per_piece_evaluation.csv"
    per_piece_xlsx = OUT_DIR / "per_piece_evaluation.xlsx"
    per_piece_details_json = OUT_DIR / "per_piece_details.json"
    global_csv = OUT_DIR / "global_reference_based_report.csv"
    summary_json = OUT_DIR / "summary_libs.json"

    for p in [generated_csv, reference_csv, per_piece_csv, per_piece_xlsx, per_piece_details_json, global_csv, summary_json]:
        if p.exists():
            p.unlink()

    # CSVs limpios, sin JSON embebido
    gen_df.drop(columns=["_pitch_hist_json"], errors="ignore").to_csv(generated_csv, index=False, encoding="utf-8-sig")
    ref_df.drop(columns=["_pitch_hist_json"], errors="ignore").to_csv(reference_csv, index=False, encoding="utf-8-sig")
    per_piece_df.to_csv(per_piece_csv, index=False, encoding="utf-8-sig")
    global_report.to_csv(global_csv, index=False, encoding="utf-8-sig")

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
        "global_best_oa": global_report.head(5).to_dict(orient="records") if not global_report.empty else [],
        "per_piece_preview": per_piece_df.head(10).to_dict(orient="records"),
        "libraries_used": ["muspy", "scipy", "pandas", "numpy"],
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[OK] Archivos generados:")
    print(f"  - {generated_csv}")
    print(f"  - {reference_csv}")
    print(f"  - {per_piece_csv}")
    print(f"  - {per_piece_xlsx}")
    print(f"  - {per_piece_details_json}")
    print(f"  - {global_csv}")
    print(f"  - {summary_json}")


if __name__ == "__main__":
    main()
