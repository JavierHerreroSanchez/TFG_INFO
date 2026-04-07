import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pretty_midi


# =========================
# CONFIG (AJUSTA ESTO)
# =========================
# Carpeta donde están los generados
GENERATED_DIR = Path(r"/output/generation_v2")
GENERATED_GLOB = "generated_from_json*.mid"   # o "*.mid"

# Referencias
REF_DIRS = [
    Path(r"/data/raw/maestro-v3.0.0"),
    Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\data\raw\ariamidi"),
]

# Para que no tarde demasiado: limita el nº de archivos de referencia
MAX_REF_FILES = 400   # sube si quieres (p.ej. 2000), pero empezaría con 400

# Salidas
OUT_DIR = GENERATED_DIR / "metrics_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_GEN_CSV = OUT_DIR / "generated_metrics.csv"
OUT_REF_CSV = OUT_DIR / "reference_metrics.csv"
OUT_SUMMARY_JSON = OUT_DIR / "summary.json"

# Asunciones para análisis por compás si no hay TS clara
DEFAULT_BEATS_PER_BAR = 4

# Rango piano típico (para flags)
PIANO_MIN = 21
PIANO_MAX = 108

# Umbrales “red flags” (ajústalos con el tiempo)
MAX_POLYPHONY_RED = 15
NOTES_PER_S_RED = 20.0
DUR_MEAN_RED_LOW = 0.03
# =========================


def entropy_from_hist(hist: np.ndarray) -> float:
    h = hist.astype(np.float64)
    s = h.sum()
    if s <= 0:
        return 0.0
    p = h / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def get_all_notes(pm: pretty_midi.PrettyMIDI, include_drums: bool = False) -> List[pretty_midi.Note]:
    notes = []
    for inst in pm.instruments:
        if (not include_drums) and inst.is_drum:
            continue
        notes.extend(inst.notes)
    return notes


def time_weighted_polyphony(notes: List[pretty_midi.Note]) -> Tuple[float, int]:
    if not notes:
        return 0.0, 0
    events = []
    for n in notes:
        events.append((n.start, +1))
        events.append((n.end, -1))
    events.sort(key=lambda x: (x[0], -x[1]))

    cur = 0
    max_cur = 0
    weighted = 0.0
    last_t = events[0][0]

    for t, d in events:
        dt = t - last_t
        if dt > 0:
            weighted += cur * dt
            last_t = t
        cur += d
        max_cur = max(max_cur, cur)

    total_time = (max(n.end for n in notes) - min(n.start for n in notes))
    if total_time <= 0:
        return 0.0, int(max_cur)
    return float(weighted / total_time), int(max_cur)


def estimate_key_ks(pc_hist: np.ndarray) -> Dict[str, float]:
    maj = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minr = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    x = pc_hist.astype(np.float64)
    if x.sum() == 0:
        return {"key": "Unknown", "mode": "Unknown", "score": 0.0}
    x = x / x.sum()

    def corr(a, b):
        a = (a - a.mean()) / (a.std() + 1e-12)
        b = (b - b.mean()) / (b.std() + 1e-12)
        return float(np.dot(a, b) / (len(a) + 1e-12))

    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    best = (-1e9, "Unknown", "Unknown")

    for shift in range(12):
        smaj = corr(x, np.roll(maj, shift))
        smin = corr(x, np.roll(minr, shift))
        if smaj > best[0]:
            best = (smaj, names[shift], "major")
        if smin > best[0]:
            best = (smin, names[shift], "minor")

    return {"key": best[1], "mode": best[2], "score": float(best[0])}


def infer_beats_per_bar(pm: pretty_midi.PrettyMIDI) -> int:
    # Si hay time signature, usamos la primera
    if pm.time_signature_changes:
        ts = pm.time_signature_changes[0]
        if ts.numerator > 0:
            return int(ts.numerator)
    return DEFAULT_BEATS_PER_BAR


def bar_pitchclass_vectors(pm: pretty_midi.PrettyMIDI, notes: List[pretty_midi.Note]) -> List[np.ndarray]:
    if not notes:
        return []
    beats = pm.get_beats()
    if beats is None or len(beats) < 8:
        tempo = pm.estimate_tempo()
        if tempo <= 0:
            return []
        beat_len = 60.0 / tempo
        t0 = min(n.start for n in notes)
        t1 = max(n.end for n in notes)
        beats = np.arange(t0, t1 + beat_len, beat_len)
    beats = np.asarray(beats)

    bpb = infer_beats_per_bar(pm)
    n_bars = (len(beats) - 1) // bpb
    if n_bars <= 1:
        return []

    bar_starts = [beats[i * bpb] for i in range(n_bars)]
    bar_ends = [beats[(i + 1) * bpb] for i in range(n_bars)]

    onsets = [(n.start, n.pitch % 12) for n in notes]
    onsets.sort()
    idx = 0

    vecs = []
    for bs, be in zip(bar_starts, bar_ends):
        v = np.zeros(12, dtype=np.float64)
        while idx < len(onsets) and onsets[idx][0] < bs:
            idx += 1
        j = idx
        while j < len(onsets) and onsets[j][0] < be:
            v[onsets[j][1]] += 1.0
            j += 1
        vecs.append(v)
    return vecs


@dataclass
class MidiMetrics:
    path: str
    duration_s: float
    n_instruments: int
    n_notes: int
    notes_per_s: float

    pitch_min: int
    pitch_max: int
    pitch_mean: float
    pitch_std: float
    pitch_outside_piano_ratio: float
    pitch_class_entropy: float

    vel_mean: float
    vel_std: float

    dur_mean: float
    dur_median: float
    dur_std: float

    avg_polyphony: float
    max_polyphony: int

    ioi_mean: float
    ioi_median: float
    ioi_std: float

    est_tempo_bpm: float
    est_key: str
    est_mode: str
    key_score: float

    bar_sim_mean: float
    bar_sim_high_ratio: float

    red_flag: str


def compute_metrics(midi_path: Path) -> MidiMetrics:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = get_all_notes(pm, include_drums=False)

    duration_s = float(pm.get_end_time())
    n_instruments = len(pm.instruments)
    n_notes = len(notes)
    notes_per_s = float(n_notes / max(duration_s, 1e-9))

    if n_notes == 0:
        return MidiMetrics(
            path=str(midi_path), duration_s=duration_s, n_instruments=n_instruments, n_notes=0,
            notes_per_s=0.0, pitch_min=-1, pitch_max=-1, pitch_mean=0.0, pitch_std=0.0,
            pitch_outside_piano_ratio=0.0, pitch_class_entropy=0.0, vel_mean=0.0, vel_std=0.0,
            dur_mean=0.0, dur_median=0.0, dur_std=0.0, avg_polyphony=0.0, max_polyphony=0,
            ioi_mean=0.0, ioi_median=0.0, ioi_std=0.0, est_tempo_bpm=float(pm.estimate_tempo() or 0.0),
            est_key="Unknown", est_mode="Unknown", key_score=0.0,
            bar_sim_mean=0.0, bar_sim_high_ratio=0.0, red_flag="NO_NOTES"
        )

    pitches = np.array([n.pitch for n in notes], dtype=np.int32)
    vels = np.array([n.velocity for n in notes], dtype=np.float64)
    durs = np.array([max(0.0, n.end - n.start) for n in notes], dtype=np.float64)
    onsets = np.sort(np.array([n.start for n in notes], dtype=np.float64))

    pitch_min = int(pitches.min())
    pitch_max = int(pitches.max())
    pitch_mean = float(pitches.mean())
    pitch_std = float(pitches.std())

    outside = np.mean((pitches < PIANO_MIN) | (pitches > PIANO_MAX))
    pitch_outside_ratio = float(outside)

    pc_hist = np.zeros(12, dtype=np.float64)
    for n in notes:
        pc_hist[n.pitch % 12] += max(1e-9, n.end - n.start)
    pc_entropy = entropy_from_hist(pc_hist)

    vel_mean = float(vels.mean())
    vel_std = float(vels.std())

    dur_mean = float(durs.mean())
    dur_median = float(np.median(durs))
    dur_std = float(durs.std())

    avg_poly, max_poly = time_weighted_polyphony(notes)

    if len(onsets) >= 2:
        ioi = np.diff(onsets)
        ioi_mean = float(ioi.mean())
        ioi_median = float(np.median(ioi))
        ioi_std = float(ioi.std())
    else:
        ioi_mean = ioi_median = ioi_std = 0.0

    tempo = float(pm.estimate_tempo() or 0.0)
    key_info = estimate_key_ks(pc_hist)

    bar_vecs = bar_pitchclass_vectors(pm, notes)
    sims = []
    for i in range(1, len(bar_vecs)):
        sims.append(cosine_sim(bar_vecs[i - 1], bar_vecs[i]))
    if sims:
        bar_sim_mean = float(np.mean(sims))
        bar_sim_high_ratio = float(np.mean([1.0 if s >= 0.85 else 0.0 for s in sims]))
    else:
        bar_sim_mean = 0.0
        bar_sim_high_ratio = 0.0

    # Red flags
    flags = []
    if max_poly >= MAX_POLYPHONY_RED:
        flags.append(f"MAX_POLY>={MAX_POLYPHONY_RED}")
    if notes_per_s >= NOTES_PER_S_RED:
        flags.append(f"NOTES_PER_S>={NOTES_PER_S_RED}")
    if dur_mean > 0 and dur_mean <= DUR_MEAN_RED_LOW:
        flags.append("DUR_MEAN_TOO_SHORT")
    if pitch_outside_ratio >= 0.01:
        flags.append("PITCH_OUTSIDE_PIANO")

    red = "|".join(flags) if flags else "OK"

    return MidiMetrics(
        path=str(midi_path),
        duration_s=duration_s,
        n_instruments=n_instruments,
        n_notes=n_notes,
        notes_per_s=notes_per_s,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        pitch_mean=pitch_mean,
        pitch_std=pitch_std,
        pitch_outside_piano_ratio=pitch_outside_ratio,
        pitch_class_entropy=pc_entropy,
        vel_mean=vel_mean,
        vel_std=vel_std,
        dur_mean=dur_mean,
        dur_median=dur_median,
        dur_std=dur_std,
        avg_polyphony=avg_poly,
        max_polyphony=max_poly,
        ioi_mean=ioi_mean,
        ioi_median=ioi_median,
        ioi_std=ioi_std,
        est_tempo_bpm=tempo,
        est_key=key_info["key"],
        est_mode=key_info["mode"],
        key_score=float(key_info["score"]),
        bar_sim_mean=bar_sim_mean,
        bar_sim_high_ratio=bar_sim_high_ratio,
        red_flag=red,
    )


def list_midis_in_dirs(dirs: List[Path], max_files: int) -> List[Path]:
    all_files = []
    for d in dirs:
        if not d.exists():
            continue
        all_files.extend(list(d.rglob("*.mid")))
        all_files.extend(list(d.rglob("*.midi")))
    all_files = sorted(set(all_files))
    if max_files and len(all_files) > max_files:
        # muestreo determinista: coge espaciados
        idxs = np.linspace(0, len(all_files) - 1, max_files).astype(int)
        all_files = [all_files[i] for i in idxs]
    return all_files


def df_from_metrics(paths: List[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        try:
            m = compute_metrics(p)
            rows.append(asdict(m))
        except Exception as e:
            rows.append({"path": str(p), "red_flag": f"ERROR:{e}"})
    return pd.DataFrame(rows)


def summarize_compare(df_gen: pd.DataFrame, df_ref: pd.DataFrame) -> Dict:
    # columnas numéricas para comparar
    numeric_cols = [
        "duration_s", "n_notes", "notes_per_s",
        "pitch_mean", "pitch_std", "pitch_outside_piano_ratio",
        "vel_mean", "vel_std",
        "dur_mean", "dur_median", "dur_std",
        "avg_polyphony", "max_polyphony",
        "ioi_mean", "ioi_median", "ioi_std",
        "est_tempo_bpm",
        "pitch_class_entropy",
        "bar_sim_mean", "bar_sim_high_ratio",
    ]
    out = {"n_generated": int(len(df_gen)), "n_reference": int(len(df_ref)), "z_scores": {}}

    ref = df_ref[numeric_cols].apply(pd.to_numeric, errors="coerce")
    gen = df_gen[numeric_cols].apply(pd.to_numeric, errors="coerce")

    ref_mean = ref.mean(numeric_only=True)
    ref_std = ref.std(numeric_only=True).replace(0, np.nan)

    gen_mean = gen.mean(numeric_only=True)

    z = (gen_mean - ref_mean) / ref_std
    out["z_scores"] = {k: float(v) for k, v in z.to_dict().items()}

    # Red-flag ratios
    out["generated_redflag_ratio"] = float(np.mean(df_gen["red_flag"].fillna("OK").ne("OK")))
    out["generated_top_flags"] = df_gen["red_flag"].value_counts().head(10).to_dict()

    return out


def main():
    gen_files = sorted(GENERATED_DIR.glob(GENERATED_GLOB))
    assert gen_files, f"No encontré generados con {GENERATED_DIR}\\{GENERATED_GLOB}"
    print(f"[GEN] found={len(gen_files)}")

    ref_files = list_midis_in_dirs(REF_DIRS, MAX_REF_FILES)
    assert ref_files, f"No encontré midis en REF_DIRS: {REF_DIRS}"
    print(f"[REF] found={len(ref_files)} (sampled)")

    df_gen = df_from_metrics(gen_files)
    df_ref = df_from_metrics(ref_files)

    df_gen.to_csv(OUT_GEN_CSV, index=False, encoding="utf-8")
    df_ref.to_csv(OUT_REF_CSV, index=False, encoding="utf-8")

    summary = summarize_compare(df_gen, df_ref)
    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] saved {OUT_GEN_CSV}")
    print(f"[OK] saved {OUT_REF_CSV}")
    print(f"[OK] saved {OUT_SUMMARY_JSON}")

    print("\n=== SUMMARY (z-scores gen vs ref) ===")
    for k, v in summary["z_scores"].items():
        print(f"{k:28s}  z={v:+.2f}")
    print("\nredflag_ratio:", summary["generated_redflag_ratio"])
    print("top_flags:", summary["generated_top_flags"])


if __name__ == "__main__":
    main()