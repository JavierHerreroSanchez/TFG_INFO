"""
Limpieza y filtrado del corpus MIDI de piano para fine-tuning.

Objetivo
--------
Los datasets MIDI masivos (p. ej., GiantMIDI) suelen contener:
  - silencios iniciales largos (antes del primer NOTE_ON),
  - silencios internos anómalos (gaps) por errores de extracción o concatenación,
  - ficheros duplicados o transcripciones incompletas.

Este script recorre recursivamente un directorio de MIDIs y genera un corpus "clean":
  1) Filtrado (hard gate) por calidad mínima (duración, nº de notas, densidad, ratio de silencio).
  2) Recorte del silencio inicial y compresión de gaps largos.
  3) Split opcional si se detecta un gap muy grande (p. ej., dos piezas pegadas).
  4) Escritura robusta (evita errores de mido por tiempos negativos).

Salidas
-------
  - OUT_CLEAN_DIR: MIDIs limpios.
  - OUT_CLEAN_INDEX_CSV: índice de MIDIs limpios (columna 'path' = path_clean).
  - OUT_REPORT_CSV: reporte completo (kept + dropped, con motivos).

Requisitos: pandas, miditoolkit
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from miditoolkit import MidiFile, Instrument, Note, TempoChange, ControlChange, PitchBend

# =============================================================================
# RUTAS
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_ROOT_DIR = PROJECT_ROOT / "data" / "finetuning" / "finetuning_sonatas_raw"
OUT_CLEAN_DIR = PROJECT_ROOT / "data" / "finetuning" / "finetuning_sonatas_clean"
OUT_CLEAN_INDEX_CSV = PROJECT_ROOT / "output" / "generation_finetuning_tfg_first" / "finetuning_clean_index.csv"
OUT_REPORT_CSV = PROJECT_ROOT / "output" / "generation_finetuning_tfg_first" / "finetuning_clean_report.csv"

# Mantiene la estructura de carpetas de INPUT_ROOT_DIR en OUT_CLEAN_DIR.
PRESERVE_TREE = True
COMMON_ROOT = INPUT_ROOT_DIR

# Control de ejecución
DRY_RUN = False
CONTINUE_ON_FAILURE = True
DROP_DUPLICATES = True

# =============================================================================
# PARÁMETROS DE LIMPIEZA TEMPORAL
# =============================================================================
# Recorte de silencio inicial: si el primer onset está más allá de este umbral, se desplaza a t=0.
LEADING_SILENCE_TRIM_S = 1.0

# Compresión de silencios internos: si un gap >= GAP_COMPRESS_FROM_S, se reduce a GAP_COMPRESS_TO_S.
GAP_COMPRESS_FROM_S = 4.0
GAP_COMPRESS_TO_S = 1.5

# Split opcional: si hay un gap >= GAP_SPLIT_S y ambos lados tienen suficiente música.
DO_SPLIT = True
GAP_SPLIT_S = 12.0
MIN_SEG_S_FOR_SPLIT = 60.0

# =============================================================================
# FILTRO DE CALIDAD (HARD GATE) + DETECCIÓN DE "SOSPECHOSOS"
# =============================================================================
# Hard gate (si falla -> DROP)
MIN_DURATION_S = 60.0          # segundos
MIN_NOTES = 300                # nº de notas
MIN_NOTES_PER_SEC = 3.5        # densidad mínima
MAX_SILENCE_RATIO = 0.55       # ratio máximo de silencio interno (gaps / duración)

# Soft scoring (si DROP_SUSPECTS=True y score>=threshold -> DROP)
DROP_SUSPECTS = True
SUSPECT_SCORE_THRESHOLD = 5

# Señales típicas de transcripción rara en piano solo
SILENCE_RATIO_BAD = 0.35
PITCH_MIN_OK = 21              # A0
PITCH_MAX_OK = 108             # C8
OUT_OF_PIANO_RATIO_BAD = 0.02
SHORT_DUR_BEATS = 1 / 64
SHORT_DUR_RATIO_BAD = 0.35
POLYPHONY_MAX_BAD = 14
TEMPO_CHANGES_BAD = 50
NON_PIANO_PROGRAM_RATIO_BAD = 0.25
TINY_FILE_BYTES = 10_000

# =============================================================================
# Helpers
# =============================================================================

def _ls(obj, attr: str):

    x = getattr(obj, attr, None)
    return x if x is not None else []


def list_midis_recursively(root: Path) -> List[Path]:

    exts = {".mid", ".midi"}
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def canonical_hash(items: List[Tuple[int, int, int, int]]) -> str:

    h = hashlib.md5()
    for s, e, p, v in items:
        h.update(s.to_bytes(4, "little", signed=False))
        h.update(e.to_bytes(4, "little", signed=False))
        h.update(p.to_bytes(2, "little", signed=False))
        h.update(v.to_bytes(2, "little", signed=False))
    return h.hexdigest()


def content_hash(m: MidiFile) -> str:

    items = []

    # Notas
    for inst_i, inst in enumerate(_ls(m, "instruments")):
        for n in _ls(inst, "notes"):
            items.append(("N", inst_i, int(n.start), int(n.end), int(n.pitch), int(n.velocity)))

        # CC (incluye pedal 64)
        for cc in _ls(inst, "control_changes"):
            items.append(("C", inst_i, int(cc.time), int(cc.number), int(cc.value)))

        # Pitch bend
        for pb in _ls(inst, "pitch_bends"):
            items.append(("P", inst_i, int(pb.time), int(pb.pitch)))

    # Tempo
    for tc in _ls(m, "tempo_changes"):
        # cuantiza un poco para estabilidad
        items.append(("T", int(tc.time), int(round(tc.tempo * 10))))

    items.sort()
    if not items:
        return ""

    h = hashlib.md5()
    for it in items:
        h.update(repr(it).encode("utf-8"))
    return h.hexdigest()


# =============================================================================
# tick <-> sec con cambios de tempo (para medir silencios en segundos)
# =============================================================================

def build_tick_to_sec_map(m: MidiFile) -> Tuple[List[int], List[float], List[float], int]:
    """
    Construye una estructura auxiliar usada por el resto del flujo.

    """

    tpq = m.ticks_per_beat or 480
    tcs = sorted(_ls(m, "tempo_changes"), key=lambda x: x.time)

    if not tcs:
        tempo_ticks, tempo_bpms = [0], [120.0]
    else:
        if tcs[0].time != 0:
            tempo_ticks = [0] + [tc.time for tc in tcs]
            tempo_bpms = [tcs[0].tempo] + [tc.tempo for tc in tcs]
        else:
            tempo_ticks = [tc.time for tc in tcs]
            tempo_bpms = [tc.tempo for tc in tcs]

    sec_at_tick = [0.0]
    for i in range(1, len(tempo_ticks)):
        prev_tick, cur_tick = tempo_ticks[i - 1], tempo_ticks[i]
        bpm = tempo_bpms[i - 1]
        sec_per_tick = (60.0 / bpm) / tpq
        sec_at_tick.append(sec_at_tick[-1] + (cur_tick - prev_tick) * sec_per_tick)

    return tempo_ticks, tempo_bpms, sec_at_tick, tpq


def tick_to_sec(tick: int, tempo_ticks: List[int], tempo_bpms: List[float], sec_at_tick: List[float], tpq: int) -> float:

    lo, hi = 0, len(tempo_ticks) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if tempo_ticks[mid] <= tick:
            lo = mid
        else:
            hi = mid - 1
    i = lo
    sec_per_tick = (60.0 / tempo_bpms[i]) / tpq
    return sec_at_tick[i] + (tick - tempo_ticks[i]) * sec_per_tick


def sec_to_tick(sec: float, tempo_ticks: List[int], tempo_bpms: List[float], sec_at_tick: List[float], tpq: int) -> int:

    lo, hi = 0, len(sec_at_tick) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sec_at_tick[mid] <= sec:
            lo = mid
        else:
            hi = mid - 1
    i = lo
    sec_per_tick = (60.0 / tempo_bpms[i]) / tpq
    return int(tempo_ticks[i] + (sec - sec_at_tick[i]) / sec_per_tick)


# =============================================================================
# Extracción de notas y métricas
# =============================================================================

def notes_simple(m: MidiFile) -> List[Tuple[int, int]]:
    """(start_tick, end_tick)"""
    out: List[Tuple[int, int]] = []
    for inst in _ls(m, "instruments"):
        for n in _ls(inst, "notes"):
            out.append((int(n.start), int(n.end)))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def notes_full(m: MidiFile) -> List[Tuple[int, int, int, int, int, bool]]:
    """(start, end, pitch, velocity, program, is_drum)"""
    out: List[Tuple[int, int, int, int, int, bool]] = []
    for inst in _ls(m, "instruments"):
        prog = int(getattr(inst, "program", -1))
        is_drum = bool(getattr(inst, "is_drum", False))
        for n in _ls(inst, "notes"):
            out.append((int(n.start), int(n.end), int(n.pitch), int(n.velocity), prog, is_drum))
    out.sort(key=lambda x: (x[0], x[1], x[2]))
    return out


def polyphony_max(ns: List[Tuple[int, int, int, int, int, bool]]) -> int:
    """Máximo nº de notas simultáneas (sweep line)."""
    if not ns:
        return 0
    events = []
    for s, e, *_ in ns:
        if e <= s:
            continue
        events.append((s, +1))
        events.append((e, -1))
    events.sort(key=lambda x: (x[0], -x[1]))  # starts before ends at same tick
    cur = 0
    mx = 0
    for _, d in events:
        cur += d
        mx = max(mx, cur)
    return mx


def compute_gaps_seconds(m: MidiFile) -> Tuple[float, float, List[Tuple[float, float, float]]]:
    """
    Gap = intervalo donde no suena ninguna nota (en ningún track).
    Devuelve: (first_onset_s, duration_s, [(gap_start_s, gap_end_s, gap_len_s), ...])
    """
    tempo_ticks, tempo_bpms, sec_at_tick, tpq = build_tick_to_sec_map(m)
    ns = notes_simple(m)
    if not ns:
        return 0.0, 0.0, []

    first_tick = ns[0][0]
    last_tick = max(e for _, e in ns)
    first_onset_s = tick_to_sec(first_tick, tempo_ticks, tempo_bpms, sec_at_tick, tpq)
    duration_s = tick_to_sec(last_tick, tempo_ticks, tempo_bpms, sec_at_tick, tpq)

    gaps: List[Tuple[float, float, float]] = []
    cur_end = ns[0][1]
    for s, e in ns[1:]:
        if s > cur_end:
            ga = tick_to_sec(cur_end, tempo_ticks, tempo_bpms, sec_at_tick, tpq)
            gb = tick_to_sec(s, tempo_ticks, tempo_bpms, sec_at_tick, tpq)
            gaps.append((ga, gb, gb - ga))
        cur_end = max(cur_end, e)

    return first_onset_s, duration_s, gaps


def compute_quality_features(m: MidiFile, src: Path) -> Dict:

    ns_full = notes_full(m)
    first_onset_s, duration_s, gaps = compute_gaps_seconds(m)

    max_gap_s = max((g[2] for g in gaps), default=0.0)
    silence_ratio = (sum(g[2] for g in gaps) / max(duration_s, 1e-9)) if gaps else 0.0

    n_notes = len(ns_full)
    notes_per_sec = n_notes / max(duration_s, 1e-9)

    tpq = m.ticks_per_beat or 480

    out_of_piano = 0
    short_dur = 0
    for s, e, pitch, *_ in ns_full:
        if pitch < PITCH_MIN_OK or pitch > PITCH_MAX_OK:
            out_of_piano += 1
        if (e - s) / tpq < SHORT_DUR_BEATS:
            short_dur += 1

    out_of_piano_ratio = (out_of_piano / n_notes) if n_notes else 1.0
    short_dur_ratio = (short_dur / n_notes) if n_notes else 1.0

    programs = [(int(getattr(inst, "program", -1)), bool(getattr(inst, "is_drum", False))) for inst in _ls(m, "instruments")]
    if not programs:
        non_piano_ratio = 1.0
    else:
        non_piano = 0
        for prog, is_drum in programs:
            if is_drum or prog < 0 or prog > 7:  # GM: piano 0..7
                non_piano += 1
        non_piano_ratio = non_piano / len(programs)

    return {
        "file_size": src.stat().st_size,
        "tpq": tpq,
        "n_tracks": len(_ls(m, "instruments")),
        "n_tempo_changes": len(_ls(m, "tempo_changes")),
        "n_notes": n_notes,
        "notes_per_sec": notes_per_sec,
        "first_onset_s": first_onset_s,
        "duration_s": duration_s,
        "max_gap_s": max_gap_s,
        "silence_ratio": silence_ratio,
        "out_of_piano_ratio": out_of_piano_ratio,
        "short_dur_ratio": short_dur_ratio,
        "polyphony_max": polyphony_max(ns_full),
        "non_piano_program_ratio": non_piano_ratio,
    }


def hard_drop_reason(feat: Dict) -> str:
    """Criterios mínimos: si se incumple alguno, se descarta el fichero."""
    if feat["duration_s"] < MIN_DURATION_S:
        return f"too_short({feat['duration_s']:.1f}s)"
    if feat["n_notes"] < MIN_NOTES:
        return f"too_few_notes({feat['n_notes']})"
    if feat["notes_per_sec"] < MIN_NOTES_PER_SEC:
        return f"too_sparse({feat['notes_per_sec']:.2f}n/s)"
    if feat["silence_ratio"] > MAX_SILENCE_RATIO:
        return f"too_much_silence({feat['silence_ratio']:.2f})"
    return ""


def suspect_score(feat: Dict, dup_seen: bool) -> Tuple[int, List[str]]:
    """
    Score acumulativo (más flexible que hard gate).
    Si DROP_SUSPECTS=True y score>=SUSPECT_SCORE_THRESHOLD -> DROP.
    """
    score = 0
    reasons: List[str] = []

    if feat["notes_per_sec"] < MIN_NOTES_PER_SEC:
        score += 3
        reasons.append(f"low_density={feat['notes_per_sec']:.2f}n/s")
    if feat["silence_ratio"] > SILENCE_RATIO_BAD:
        score += 2
        reasons.append(f"high_silence_ratio={feat['silence_ratio']:.2f}")
    if feat["out_of_piano_ratio"] > OUT_OF_PIANO_RATIO_BAD:
        score += 2
        reasons.append(f"out_of_piano_ratio={feat['out_of_piano_ratio']:.2f}")
    if feat["short_dur_ratio"] > SHORT_DUR_RATIO_BAD:
        score += 2
        reasons.append(f"short_dur_ratio={feat['short_dur_ratio']:.2f}")
    if feat["polyphony_max"] > POLYPHONY_MAX_BAD:
        score += 1
        reasons.append(f"polyphony_max={feat['polyphony_max']}")
    if feat["n_tempo_changes"] > TEMPO_CHANGES_BAD:
        score += 1
        reasons.append(f"many_tempo_changes={feat['n_tempo_changes']}")
    if feat["non_piano_program_ratio"] > NON_PIANO_PROGRAM_RATIO_BAD:
        score += 2
        reasons.append(f"non_piano_tracks_ratio={feat['non_piano_program_ratio']:.2f}")
    if feat["file_size"] < TINY_FILE_BYTES:
        score += 2
        reasons.append(f"tiny_file={feat['file_size']}B")
    if dup_seen:
        score += 2
        reasons.append("duplicate_content_seen_before")

    return score, reasons


# =============================================================================
# Limpieza de estructura temporal.
# =============================================================================

def shift_all_events(m: MidiFile, shift_ticks: int) -> None:
    """Desplaza todos los eventos hacia atrás (recorte del silencio inicial)."""
    if shift_ticks <= 0:
        return

    for inst in _ls(m, "instruments"):
        for n in _ls(inst, "notes"):
            n.start -= shift_ticks
            n.end -= shift_ticks
        for cc in _ls(inst, "control_changes"):
            cc.time -= shift_ticks
        for pb in _ls(inst, "pitch_bends"):
            pb.time -= shift_ticks

    for tc in _ls(m, "tempo_changes"):
        tc.time -= shift_ticks

    for attr in ["time_signature_changes", "key_signature_changes", "markers", "lyrics"]:
        lst = getattr(m, attr, None)
        if lst is None:
            continue
        for x in lst:
            x.time -= shift_ticks


def compress_gaps_inplace(m: MidiFile) -> int:
    """
    Compresión de gaps:
      - detecta gaps >= GAP_COMPRESS_FROM_S,
      - reduce cada gap a GAP_COMPRESS_TO_S desplazando eventos posteriores.
    """
    tempo_ticks, tempo_bpms, sec_at_tick, tpq = build_tick_to_sec_map(m)
    ns = notes_simple(m)
    if not ns:
        return 0

    cuts: List[Tuple[int, int]] = []
    cur_end = ns[0][1]
    for s, e in ns[1:]:
        if s > cur_end:
            ga_s = tick_to_sec(cur_end, tempo_ticks, tempo_bpms, sec_at_tick, tpq)
            gb_s = tick_to_sec(s, tempo_ticks, tempo_bpms, sec_at_tick, tpq)
            gap_s = gb_s - ga_s
            if gap_s >= GAP_COMPRESS_FROM_S:
                reduce_s = gap_s - GAP_COMPRESS_TO_S
                cut_tick = int(cur_end)
                cut_sec = tick_to_sec(cut_tick, tempo_ticks, tempo_bpms, sec_at_tick, tpq)
                reduce_tick = sec_to_tick(cut_sec + reduce_s, tempo_ticks, tempo_bpms, sec_at_tick, tpq) - sec_to_tick(
                    cut_sec, tempo_ticks, tempo_bpms, sec_at_tick, tpq
                )
                cuts.append((cut_tick, max(0, int(reduce_tick))))
        cur_end = max(cur_end, e)

    if not cuts:
        return 0
    cuts.sort(key=lambda x: x[0])

    def total_shift(t: int) -> int:

        s = 0
        for ct, rt in cuts:
            if t > ct:
                s += rt
            else:
                break
        return s

    for inst in _ls(m, "instruments"):
        for n in _ls(inst, "notes"):
            n.start -= total_shift(int(n.start))
            n.end -= total_shift(int(n.end))
        for cc in _ls(inst, "control_changes"):
            cc.time -= total_shift(int(cc.time))
        for pb in _ls(inst, "pitch_bends"):
            pb.time -= total_shift(int(pb.time))

    for tc in _ls(m, "tempo_changes"):
        tc.time -= total_shift(int(tc.time))

    for attr in ["time_signature_changes", "key_signature_changes", "markers", "lyrics"]:
        lst = getattr(m, attr, None)
        if lst is None:
            continue
        for x in lst:
            x.time -= total_shift(int(x.time))

    return len(cuts)


def split_on_clear_gap(m: MidiFile) -> List[MidiFile]:
    """
    Split binario por el gap más grande (si es suficientemente grande).
    Útil para detectar concatenaciones (dos piezas en un mismo MIDI).
    """
    if not DO_SPLIT:
        return [m]

    first_onset_s, duration_s, gaps = compute_gaps_seconds(m)
    candidates = [g for g in gaps if g[2] >= GAP_SPLIT_S]
    if not candidates:
        return [m]

    ga, gb, _ = max(candidates, key=lambda x: x[2])
    cut_sec = (ga + gb) / 2.0
    if cut_sec < MIN_SEG_S_FOR_SPLIT or (duration_s - cut_sec) < MIN_SEG_S_FOR_SPLIT:
        return [m]

    tempo_ticks, tempo_bpms, sec_at_tick, tpq = build_tick_to_sec_map(m)
    cut_tick = sec_to_tick(cut_sec, tempo_ticks, tempo_bpms, sec_at_tick, tpq)

    ns = notes_simple(m)
    last_tick = max(e for _, e in ns)

    def clone_segment(start_tick: int, end_tick: int) -> MidiFile:

        m2 = MidiFile()
        m2.ticks_per_beat = m.ticks_per_beat

        tcs = [TempoChange(tc.tempo, int(tc.time - start_tick)) for tc in _ls(m, "tempo_changes") if start_tick <= tc.time <= end_tick]
        m2.tempo_changes = sorted(tcs, key=lambda tc: tc.time) if tcs else [TempoChange(120.0, 0)]

        for attr in ["time_signature_changes", "key_signature_changes", "markers", "lyrics"]:
            lst = getattr(m, attr, None)
            if lst is None:
                continue
            keep = [x for x in lst if start_tick <= x.time <= end_tick]
            for x in keep:
                x.time = int(x.time - start_tick)
            keep.sort(key=lambda x: x.time)
            setattr(m2, attr, keep)

        m2.instruments = []
        for inst in _ls(m, "instruments"):
            inst2 = Instrument(program=inst.program, is_drum=inst.is_drum, name=getattr(inst, "name", ""))

            for n in _ls(inst, "notes"):
                if n.start >= start_tick and n.end <= end_tick:
                    inst2.notes.append(Note(int(n.velocity), int(n.pitch), int(n.start - start_tick), int(n.end - start_tick)))

            for cc in _ls(inst, "control_changes"):
                if start_tick <= cc.time <= end_tick:
                    inst2.control_changes.append(ControlChange(int(cc.number), int(cc.value), int(cc.time - start_tick)))

            for pb in _ls(inst, "pitch_bends"):
                if start_tick <= pb.time <= end_tick:
                    inst2.pitch_bends.append(PitchBend(int(pb.pitch), int(pb.time - start_tick)))

            if inst2.notes or inst2.control_changes or inst2.pitch_bends:
                m2.instruments.append(inst2)

        return m2

    seg1 = clone_segment(0, cut_tick)
    seg2 = clone_segment(cut_tick, last_tick)
    return [seg1, seg2] if seg1.instruments and seg2.instruments else [m]


# =============================================================================
# Escritura robusta (evita crash por tiempos negativos / orden)
# =============================================================================

def sanitize_miditoolkit(m: MidiFile) -> None:
    """Normaliza tiempos y orden para un dump() seguro."""
    m.tempo_changes = [tc for tc in _ls(m, "tempo_changes") if tc.time >= 0]
    m.tempo_changes.sort(key=lambda tc: tc.time)
    if not m.tempo_changes:
        m.tempo_changes = [TempoChange(120.0, 0)]

    for attr in ["time_signature_changes", "key_signature_changes", "markers", "lyrics"]:
        lst = getattr(m, attr, None)
        if lst is None:
            continue
        lst = [x for x in lst if getattr(x, "time", 0) >= 0]
        lst.sort(key=lambda x: x.time)
        setattr(m, attr, lst)

    new_insts = []
    for inst in _ls(m, "instruments"):
        inst.notes = [n for n in _ls(inst, "notes") if n.start >= 0 and n.end > n.start]
        inst.notes.sort(key=lambda n: (n.start, n.end, n.pitch, n.velocity))

        inst.control_changes = [cc for cc in _ls(inst, "control_changes") if cc.time >= 0]
        inst.control_changes.sort(key=lambda cc: cc.time)

        inst.pitch_bends = [pb for pb in _ls(inst, "pitch_bends") if pb.time >= 0]
        inst.pitch_bends.sort(key=lambda pb: pb.time)

        if inst.notes or inst.control_changes or inst.pitch_bends:
            new_insts.append(inst)

    m.instruments = new_insts


def rebuild_miditoolkit(m: MidiFile) -> MidiFile:
    """Reconstruye el objeto para evitar deltas negativos internos al guardar."""
    m2 = MidiFile()
    m2.ticks_per_beat = m.ticks_per_beat

    tcs = [TempoChange(tc.tempo, max(0, int(tc.time))) for tc in _ls(m, "tempo_changes")]
    m2.tempo_changes = sorted(tcs, key=lambda tc: tc.time) if tcs else [TempoChange(120.0, 0)]

    for attr in ["time_signature_changes", "key_signature_changes", "markers", "lyrics"]:
        lst = getattr(m, attr, None)
        if lst is None:
            continue
        clean = [x for x in lst if getattr(x, "time", 0) >= 0]
        clean.sort(key=lambda x: x.time)
        setattr(m2, attr, clean)

    m2.instruments = []
    for inst in _ls(m, "instruments"):
        inst2 = Instrument(program=inst.program, is_drum=inst.is_drum, name=getattr(inst, "name", ""))

        inst2.notes = [Note(int(n.velocity), int(n.pitch), int(n.start), int(n.end)) for n in _ls(inst, "notes") if n.start >= 0 and n.end > n.start]
        inst2.notes.sort(key=lambda n: (n.start, n.end, n.pitch, n.velocity))

        inst2.control_changes = [ControlChange(int(cc.number), int(cc.value), int(cc.time)) for cc in _ls(inst, "control_changes") if cc.time >= 0]
        inst2.control_changes.sort(key=lambda cc: cc.time)

        inst2.pitch_bends = [PitchBend(int(pb.pitch), int(pb.time)) for pb in _ls(inst, "pitch_bends") if pb.time >= 0]
        inst2.pitch_bends.sort(key=lambda pb: pb.time)

        if inst2.notes or inst2.control_changes or inst2.pitch_bends:
            m2.instruments.append(inst2)

    return m2


def safe_dump_miditoolkit(m: MidiFile, out_path: Path, debug_tag: str = "") -> bool:
    """Dump con reintento (sanitize -> dump; si falla -> rebuild -> dump)."""
    try:
        sanitize_miditoolkit(m)
        m.dump(str(out_path))
        return True
    except ValueError as e:
        try:
            m2 = rebuild_miditoolkit(m)
            sanitize_miditoolkit(m2)
            m2.dump(str(out_path))
            return True
        except Exception as e2:
            print(f"[dump fail] {debug_tag} | {type(e).__name__}: {e} | retry: {type(e2).__name__}: {e2}")
            return False
    except Exception as e:
        print(f"[dump fail] {debug_tag} | {type(e).__name__}: {e}")
        return False


# =============================================================================
# Pipeline principal
# =============================================================================

def make_out_path(src: Path, part_idx: int) -> Path:

    if PRESERVE_TREE:
        try:
            rel = src.relative_to(COMMON_ROOT)
        except Exception:
            rel = Path(src.name)
        base = OUT_CLEAN_DIR / rel
    else:
        base = OUT_CLEAN_DIR / src.name

    stem = base.with_suffix("").name
    suffix = f".part{part_idx}" if part_idx > 1 else ""
    return base.parent / f"{stem}{suffix}.mid"


def main():
    """Punto de entrada del script cuando se ejecuta desde consola."""

    if not INPUT_ROOT_DIR.exists():
        raise FileNotFoundError(f"No existe INPUT_ROOT_DIR: {INPUT_ROOT_DIR}")

    OUT_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CLEAN_INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)

    midis = list_midis_recursively(INPUT_ROOT_DIR)

    seen_hash: Dict[str, str] = {}
    report_rows: List[Dict] = []
    kept_rows: List[Dict] = []

    for i, src in enumerate(midis, start=1):
        row: Dict = {
            "path_original": str(src),
            "rel_path": str(src.relative_to(INPUT_ROOT_DIR)),
            "keep_final": 0,
            "action_done": "",
            "drop_reason": "",
            "path_clean": "",
            "part_idx": 0,
        }

        try:
            midi = MidiFile(str(src))
        except Exception as e:
            row.update({"action_done": "DROP", "drop_reason": f"read_error:{type(e).__name__}"})
            report_rows.append(row)
            if not CONTINUE_ON_FAILURE:
                raise
            continue

        # 1) Duplicados (streaming)
        dup_seen = False
        if DROP_DUPLICATES:
            h = content_hash(midi)
            row["dup_hash"] = h
            if h:
                if h in seen_hash:
                    dup_seen = True
                    row.update({"action_done": "DROP", "drop_reason": f"duplicate_of={seen_hash[h]}"})
                    report_rows.append(row)
                    continue
                seen_hash[h] = str(src)

        # 2) Métricas + filtrado de calidad
        feat = compute_quality_features(midi, src)
        row.update(feat)

        reason = hard_drop_reason(feat)
        if reason:
            row.update({"action_done": "DROP", "drop_reason": reason})
            report_rows.append(row)
            continue

        sc, why = suspect_score(feat, dup_seen=dup_seen)
        row.update({"suspect_score": int(sc), "suspect_reason": "; ".join(why) if why else "ok"})
        if DROP_SUSPECTS and sc >= SUSPECT_SCORE_THRESHOLD:
            row.update({"action_done": "DROP", "drop_reason": f"suspect(score={sc})"})
            report_rows.append(row)
            continue

        # 3) Limpieza de estructura temporal.
        actions: List[str] = []

        if feat["first_onset_s"] > LEADING_SILENCE_TRIM_S:
            tempo_ticks, tempo_bpms, sec_at_tick, tpq = build_tick_to_sec_map(midi)
            shift_tick = sec_to_tick(float(feat["first_onset_s"]), tempo_ticks, tempo_bpms, sec_at_tick, tpq)
            shift_all_events(midi, shift_tick)
            actions.append("TRIM_START")

        parts = split_on_clear_gap(midi)
        if len(parts) > 1:
            actions.append("SPLIT")

        for part_idx, part in enumerate(parts, start=1):
            n_comp = compress_gaps_inplace(part)
            actions_part = actions + (["COMPRESS_GAPS"] if n_comp > 0 else [])

            # filtro por parte (evita colar fragmentos cortos tras split)
            feat_part = compute_quality_features(part, src)
            part_reason = hard_drop_reason(feat_part)
            if part_reason:
                rr = dict(row)
                rr.update({"action_done": "DROP", "drop_reason": f"part{part_idx}:{part_reason}"})
                report_rows.append(rr)
                continue

            dst = make_out_path(src, part_idx)
            dst.parent.mkdir(parents=True, exist_ok=True)

            ok = True
            if not DRY_RUN:
                ok = safe_dump_miditoolkit(part, dst, debug_tag=str(src))

            rr = dict(row)
            if ok:
                rr.update({
                    "keep_final": 1,
                    "action_done": "+".join(actions_part) if actions_part else "KEEP",
                    "path_clean": str(dst),
                    "part_idx": part_idx,
                })
                kept_rows.append(rr)
            else:
                rr.update({"action_done": "DROP", "drop_reason": "dump_failed"})
                if not CONTINUE_ON_FAILURE:
                    raise RuntimeError(f"dump_failed: {src}")

            report_rows.append(rr)

        if i % 200 == 0:
            print(f"[PROC] {i}/{len(midis)}")

    # 4) CSVs
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(OUT_REPORT_CSV, index=False)

    kept_df = pd.DataFrame(kept_rows)
    if len(kept_df):
        kept_df = kept_df.copy()
        kept_df["path"] = kept_df["path_clean"]
        kept_df.to_csv(OUT_CLEAN_INDEX_CSV, index=False)
    else:
        pd.DataFrame(columns=["path"]).to_csv(OUT_CLEAN_INDEX_CSV, index=False)

    print(f"[OK] report -> {OUT_REPORT_CSV} rows={len(report_df)}")
    print(f"[OK] clean index -> {OUT_CLEAN_INDEX_CSV} rows={len(kept_df)}")
    print(f"[OK] clean dir -> {OUT_CLEAN_DIR}")


# Ejecución directa del script.
if __name__ == "__main__":
    main()
