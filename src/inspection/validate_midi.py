"""
validate_midi.py

Validador práctico de archivos MIDI para validar de corpus.
Pensado para ejecutarse cómodamente desde PyCharm o terminal.

Qué comprueba:
- Si el MIDI se puede parsear correctamente
- Número de notas, duración, instrumentos y densidad
- Tempos, compases y key signatures declaradas
- Estimación tonal a partir del contenido de notas
- Posible key signature sospechosa (por ejemplo, C major por defecto)
- Calidad de cuantización aproximada
- Notas anómalas: duración no positiva, muy cortas, fuera de rango pianístico
- Solapamientos dentro de un mismo instrumento/pitch
- Puntuación global y veredicto final

Dependencias:
    pip install pretty_midi numpy

Uso:
    python validate_midi.py /ruta/a/midis --recursive --pretty
    python validate_midi.py /ruta/a/midis --csv reporte.csv --json resumen.json
    python validate_midi.py /ruta/a/archivo.mid --pretty
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pretty_midi


PITCH_CLASS_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F",
                           "F#", "G", "G#", "A", "A#", "B"]
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                          2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=float)
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                          2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=float)

DEFAULT_MIN_PIANO_PITCH = 21
DEFAULT_MAX_PIANO_PITCH = 108


@dataclass
class MidiValidationResult:
    """Representa MidiValidationResult dentro del flujo experimental del TFG."""

    path: str
    status: str
    score: float
    parse_ok: bool
    error: str

    duration_sec: float
    note_count: int
    instrument_count: int
    non_drum_instrument_count: int
    drum_instrument_count: int

    tempo_mean_bpm: float
    tempo_change_count: int
    time_signature_count: int
    key_signature_count: int

    declared_key: str
    estimated_key: str
    estimated_key_confidence: float
    suspicious_declared_key: bool

    notes_per_second: float
    mean_note_duration: float
    short_note_ratio: float
    invalid_duration_count: int

    pitch_min: int
    pitch_max: int
    out_of_piano_range_ratio: float

    overlap_ratio: float
    quantization_error_beats: float

    warnings: str


def midi_key_number_to_name(key_number: int) -> str:
    """
    pretty_midi: 0-11 major, 12-23 minor
    """
    if 0 <= key_number <= 11:
        return f"{PITCH_CLASS_NAMES_SHARP[key_number]} major"
    if 12 <= key_number <= 23:
        return f"{PITCH_CLASS_NAMES_SHARP[key_number - 12]} minor"
    return f"unknown({key_number})"


def safe_mean(values: Sequence[float]) -> float:
    """
    Implementa la logica de safe mean dentro del pipeline del TFG.

    Parametros principales: values.
    """

    return float(np.mean(values)) if values else 0.0


def gather_midi_files(path: str, recursive: bool) -> List[str]:
    """
    Implementa la logica de gather midi files dentro del pipeline del TFG.

    Parametros principales: path, recursive.
    """

    if os.path.isfile(path):
        return [path]

    midi_files = []
    valid_ext = {".mid", ".midi"}

    if recursive:
        for root, _, files in os.walk(path):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext in valid_ext:
                    midi_files.append(os.path.join(root, name))
    else:
        for name in os.listdir(path):
            full = os.path.join(path, name)
            if os.path.isfile(full):
                ext = os.path.splitext(name)[1].lower()
                if ext in valid_ext:
                    midi_files.append(full)

    midi_files.sort()
    return midi_files


def estimate_key_from_notes(pm: pretty_midi.PrettyMIDI) -> Tuple[str, float, int, str]:
    """
    Estima tonalidad a partir de la distribución de clases de pitch
    usando perfiles tipo Krumhansl-Schmuckler.
    Devuelve:
      key_name, confidence, tonic_pc, mode
    """
    pitch_class_hist = np.zeros(12, dtype=float)

    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            dur = max(0.0, note.end - note.start)
            pitch_class_hist[note.pitch % 12] += dur if dur > 0 else 0.0

    total = pitch_class_hist.sum()
    if total <= 0:
        return "unknown", 0.0, -1, "unknown"

    pitch_class_hist /= total

    major_scores = []
    minor_scores = []

    for shift in range(12):
        maj = np.corrcoef(pitch_class_hist, np.roll(MAJOR_PROFILE, shift))[0, 1]
        mino = np.corrcoef(pitch_class_hist, np.roll(MINOR_PROFILE, shift))[0, 1]
        major_scores.append(maj if not np.isnan(maj) else -1.0)
        minor_scores.append(mino if not np.isnan(mino) else -1.0)

    best_major_pc = int(np.argmax(major_scores))
    best_minor_pc = int(np.argmax(minor_scores))
    best_major_score = float(major_scores[best_major_pc])
    best_minor_score = float(minor_scores[best_minor_pc])

    all_scores = major_scores + minor_scores
    sorted_scores = sorted(all_scores, reverse=True)

    if best_major_score >= best_minor_score:
        key_name = f"{PITCH_CLASS_NAMES_SHARP[best_major_pc]} major"
        best_score = best_major_score
        tonic_pc = best_major_pc
        mode = "major"
    else:
        key_name = f"{PITCH_CLASS_NAMES_SHARP[best_minor_pc]} minor"
        best_score = best_minor_score
        tonic_pc = best_minor_pc
        mode = "minor"

    if len(sorted_scores) >= 2:
        confidence = float(best_score - sorted_scores[1])
    else:
        confidence = float(best_score)

    return key_name, confidence, tonic_pc, mode


def get_declared_key(pm: pretty_midi.PrettyMIDI) -> str:
    """
    Implementa la logica de get declared key dentro del pipeline del TFG.

    Parametros principales: pm.
    """

    if not pm.key_signature_changes:
        return ""
    return midi_key_number_to_name(pm.key_signature_changes[0].key_number)


def compute_overlap_ratio(pm: pretty_midi.PrettyMIDI) -> float:
    """
    Mide el porcentaje aproximado de notas que se solapan con otra nota
    del mismo pitch dentro del mismo instrumento.
    """
    total_notes = 0
    overlap_notes = 0

    for inst in pm.instruments:
        if inst.is_drum:
            continue

        by_pitch: Dict[int, List[pretty_midi.Note]] = {}
        for note in inst.notes:
            by_pitch.setdefault(note.pitch, []).append(note)

        for pitch_notes in by_pitch.values():
            pitch_notes.sort(key=lambda n: (n.start, n.end))
            prev_end = -math.inf
            for note in pitch_notes:
                total_notes += 1
                if note.start < prev_end - 1e-6:
                    overlap_notes += 1
                prev_end = max(prev_end, note.end)

    if total_notes == 0:
        return 0.0
    return overlap_notes / total_notes


def estimate_quantization_error_beats(pm: pretty_midi.PrettyMIDI) -> float:
    """
    Estima error medio de cuantización en unidades de beat.
    Usa los downbeats y/o el tempo inicial como referencia.
    Cuanto menor, mejor.
    """
    all_onsets = []

    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            all_onsets.append(note.start)

    if not all_onsets:
        return 0.0

    all_onsets = np.array(sorted(all_onsets), dtype=float)

    try:
        tempo_times, tempi = pm.get_tempo_changes()
        if len(tempi) == 0:
            bpm = 120.0
        else:
            bpm = float(np.median(tempi))
    except Exception:
        bpm = 120.0

    beat_sec = 60.0 / max(1e-6, bpm)

    candidate_steps = np.array([
        1.0,      # negra
        0.5,      # corchea
        0.25,     # semicorchea
        1.0 / 3.0,
        1.0 / 6.0,
        0.125
    ], dtype=float) * beat_sec

    t0 = float(all_onsets[0])

    errors = []
    for onset in all_onsets:
        local_errors = []
        delta = onset - t0
        for step in candidate_steps:
            grid_idx = round(delta / step)
            snapped = t0 + grid_idx * step
            local_errors.append(abs(onset - snapped) / beat_sec)
        errors.append(min(local_errors))

    return float(np.mean(errors))


def evaluate_midi(
    path: str,
    min_piano_pitch: int = DEFAULT_MIN_PIANO_PITCH,
    max_piano_pitch: int = DEFAULT_MAX_PIANO_PITCH,
) -> MidiValidationResult:
    """
    Evalua las salidas generadas mediante metricas del proyecto.

    Parametros principales: path, min_piano_pitch, max_piano_pitch.
    """

    warnings = []

    try:
        pm = pretty_midi.PrettyMIDI(path)
    except Exception as e:
        return MidiValidationResult(
            path=path,
            status="broken",
            score=0.0,
            parse_ok=False,
            error=str(e),
            duration_sec=0.0,
            note_count=0,
            instrument_count=0,
            non_drum_instrument_count=0,
            drum_instrument_count=0,
            tempo_mean_bpm=0.0,
            tempo_change_count=0,
            time_signature_count=0,
            key_signature_count=0,
            declared_key="",
            estimated_key="unknown",
            estimated_key_confidence=0.0,
            suspicious_declared_key=False,
            notes_per_second=0.0,
            mean_note_duration=0.0,
            short_note_ratio=0.0,
            invalid_duration_count=0,
            pitch_min=-1,
            pitch_max=-1,
            out_of_piano_range_ratio=0.0,
            overlap_ratio=0.0,
            quantization_error_beats=1.0,
            warnings="parse_error",
        )

    duration_sec = float(pm.get_end_time())
    instrument_count = len(pm.instruments)
    drum_instrument_count = sum(1 for i in pm.instruments if i.is_drum)
    non_drum_instrument_count = instrument_count - drum_instrument_count

    all_notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        all_notes.extend(inst.notes)

    note_count = len(all_notes)

    if note_count == 0:
        warnings.append("no_notes")

    pitches = [n.pitch for n in all_notes] if all_notes else []
    durations = [(n.end - n.start) for n in all_notes] if all_notes else []

    invalid_duration_count = sum(1 for d in durations if d <= 0.0)
    positive_durations = [d for d in durations if d > 0.0]
    mean_note_duration = safe_mean(positive_durations)

    short_note_threshold = 0.03
    short_note_ratio = (
        sum(1 for d in positive_durations if d < short_note_threshold) / len(positive_durations)
        if positive_durations else 0.0
    )

    pitch_min = min(pitches) if pitches else -1
    pitch_max = max(pitches) if pitches else -1

    out_of_piano_range_count = sum(
        1 for p in pitches if p < min_piano_pitch or p > max_piano_pitch
    )
    out_of_piano_range_ratio = (
        out_of_piano_range_count / len(pitches) if pitches else 0.0
    )

    notes_per_second = note_count / duration_sec if duration_sec > 0 else 0.0

    try:
        tempo_times, tempi = pm.get_tempo_changes()
        tempo_mean_bpm = float(np.mean(tempi)) if len(tempi) > 0 else 120.0
        tempo_change_count = max(0, len(tempi) - 1)
    except Exception:
        tempo_mean_bpm = 120.0
        tempo_change_count = 0

    time_signature_count = len(pm.time_signature_changes)
    key_signature_count = len(pm.key_signature_changes)

    declared_key = get_declared_key(pm)
    estimated_key, estimated_key_confidence, _, _ = estimate_key_from_notes(pm)

    suspicious_declared_key = False
    if declared_key:
        if declared_key != estimated_key and estimated_key != "unknown":
            suspicious_declared_key = True
            warnings.append("declared_key_mismatch")
        if declared_key == "C major" and estimated_key not in ("C major", "A minor", "unknown"):
            suspicious_declared_key = True
            if "declared_key_mismatch" not in warnings:
                warnings.append("suspicious_c_major")

    overlap_ratio = compute_overlap_ratio(pm)
    quantization_error_beats = estimate_quantization_error_beats(pm)

    if duration_sec <= 0:
        warnings.append("zero_duration")
    if invalid_duration_count > 0:
        warnings.append("invalid_note_durations")
    if short_note_ratio > 0.15:
        warnings.append("many_very_short_notes")
    if out_of_piano_range_ratio > 0.10:
        warnings.append("many_notes_out_of_piano_range")
    if overlap_ratio > 0.10:
        warnings.append("many_overlaps")
    if quantization_error_beats > 0.08:
        warnings.append("poor_quantization")
    if notes_per_second > 25:
        warnings.append("very_high_density")
    if non_drum_instrument_count == 0 and note_count > 0:
        warnings.append("only_drums_or_unusual_setup")
    if instrument_count > 8:
        warnings.append("many_instruments")

    score = 100.0

    if note_count == 0:
        score -= 80.0
    if duration_sec <= 0:
        score -= 60.0
    score -= min(30.0, invalid_duration_count * 2.0)
    score -= short_note_ratio * 20.0
    score -= out_of_piano_range_ratio * 25.0
    score -= overlap_ratio * 25.0
    score -= min(25.0, quantization_error_beats * 200.0)
    score -= 10.0 if suspicious_declared_key else 0.0
    score -= 8.0 if notes_per_second > 25 else 0.0
    score -= 6.0 if instrument_count > 8 else 0.0

    score = max(0.0, min(100.0, score))

    if not note_count or duration_sec <= 0:
        status = "bad"
    elif score >= 85:
        status = "good"
    elif score >= 70:
        status = "usable"
    elif score >= 50:
        status = "warning"
    else:
        status = "bad"

    return MidiValidationResult(
        path=path,
        status=status,
        score=round(score, 3),
        parse_ok=True,
        error="",
        duration_sec=round(duration_sec, 6),
        note_count=note_count,
        instrument_count=instrument_count,
        non_drum_instrument_count=non_drum_instrument_count,
        drum_instrument_count=drum_instrument_count,
        tempo_mean_bpm=round(tempo_mean_bpm, 6),
        tempo_change_count=tempo_change_count,
        time_signature_count=time_signature_count,
        key_signature_count=key_signature_count,
        declared_key=declared_key,
        estimated_key=estimated_key,
        estimated_key_confidence=round(float(estimated_key_confidence), 6),
        suspicious_declared_key=suspicious_declared_key,
        notes_per_second=round(notes_per_second, 6),
        mean_note_duration=round(mean_note_duration, 6),
        short_note_ratio=round(short_note_ratio, 6),
        invalid_duration_count=invalid_duration_count,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        out_of_piano_range_ratio=round(out_of_piano_range_ratio, 6),
        overlap_ratio=round(overlap_ratio, 6),
        quantization_error_beats=round(quantization_error_beats, 6),
        warnings=";".join(warnings),
    )


def print_pretty_summary(results: List[MidiValidationResult]) -> None:
    """
    Implementa la logica de print pretty summary dentro del pipeline del TFG.

    Parametros principales: results.
    """

    if not results:
        print("No se encontraron archivos MIDI.")
        return

    total = len(results)
    parse_ok = sum(r.parse_ok for r in results)
    broken = sum(r.status == "broken" for r in results)
    good = sum(r.status == "good" for r in results)
    usable = sum(r.status == "usable" for r in results)
    warning = sum(r.status == "warning" for r in results)
    bad = sum(r.status == "bad" for r in results)

    mean_score = np.mean([r.score for r in results]) if results else 0.0
    mean_notes = np.mean([r.note_count for r in results]) if results else 0.0
    mean_dur = np.mean([r.duration_sec for r in results]) if results else 0.0

    print("=" * 80)
    print("RESUMEN VALIDACIÓN MIDI")
    print("=" * 80)
    print(f"Total archivos:           {total}")
    print(f"Parseados correctamente:  {parse_ok}")
    print(f"Broken:                   {broken}")
    print(f"Good:                     {good}")
    print(f"Usable:                   {usable}")
    print(f"Warning:                  {warning}")
    print(f"Bad:                      {bad}")
    print(f"Score medio:              {mean_score:.2f}")
    print(f"Notas medias:             {mean_notes:.2f}")
    print(f"Duración media (s):       {mean_dur:.2f}")
    print()

    worst = sorted(results, key=lambda r: r.score)[:10]
    print("PEORES 10 ARCHIVOS")
    print("-" * 80)
    for r in worst:
        print(f"[{r.status:7}] score={r.score:6.2f}  notes={r.note_count:6d}  {r.path}")
        if r.warnings:
            print(f"    warnings: {r.warnings}")
        if r.error:
            print(f"    error: {r.error}")
    print()


def save_csv(results: List[MidiValidationResult], path: str) -> None:
    """
    Guarda resultados intermedios o finales en disco.

    Parametros principales: results, path.
    """

    if not results:
        return
    fieldnames = list(asdict(results[0]).keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def save_json(results: List[MidiValidationResult], path: str) -> None:
    """
    Guarda resultados intermedios o finales en disco.

    Parametros principales: results, path.
    """

    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)


def build_argparser() -> argparse.ArgumentParser:
    """Construye una estructura auxiliar usada por el resto del flujo."""

    parser = argparse.ArgumentParser(
        description="Valida archivos MIDI y genera métricas de calidad."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Ruta a un archivo MIDI o a un directorio con MIDIs."
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Busca MIDIs recursivamente dentro del directorio."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="Ruta de salida para guardar CSV."
    )
    parser.add_argument(
        "--json",
        type=str,
        default="",
        help="Ruta de salida para guardar JSON."
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Imprime resumen legible por consola."
    )
    parser.add_argument(
        "--min-piano-pitch",
        type=int,
        default=DEFAULT_MIN_PIANO_PITCH,
        help="Pitch MIDI mínimo esperado para piano. Por defecto 21 (A0)."
    )
    parser.add_argument(
        "--max-piano-pitch",
        type=int,
        default=DEFAULT_MAX_PIANO_PITCH,
        help="Pitch MIDI máximo esperado para piano. Por defecto 108 (C8)."
    )
    return parser


def main() -> None:
    """Punto de entrada del script cuando se ejecuta desde consola."""

    parser = build_argparser()
    args = parser.parse_args()

    midi_files = gather_midi_files(args.input_path, args.recursive)

    if not midi_files:
        print("No se encontraron archivos .mid o .midi en la ruta indicada.")
        return

    results = []
    for idx, midi_path in enumerate(midi_files, start=1):
        try:
            result = evaluate_midi(
                midi_path,
                min_piano_pitch=args.min_piano_pitch,
                max_piano_pitch=args.max_piano_pitch,
            )
            results.append(result)
            print(f"[{idx}/{len(midi_files)}] {result.status:7} score={result.score:6.2f} {midi_path}")
        except Exception as e:
            results.append(
                MidiValidationResult(
                    path=midi_path,
                    status="broken",
                    score=0.0,
                    parse_ok=False,
                    error=f"unexpected_error: {e}",
                    duration_sec=0.0,
                    note_count=0,
                    instrument_count=0,
                    non_drum_instrument_count=0,
                    drum_instrument_count=0,
                    tempo_mean_bpm=0.0,
                    tempo_change_count=0,
                    time_signature_count=0,
                    key_signature_count=0,
                    declared_key="",
                    estimated_key="unknown",
                    estimated_key_confidence=0.0,
                    suspicious_declared_key=False,
                    notes_per_second=0.0,
                    mean_note_duration=0.0,
                    short_note_ratio=0.0,
                    invalid_duration_count=0,
                    pitch_min=-1,
                    pitch_max=-1,
                    out_of_piano_range_ratio=0.0,
                    overlap_ratio=0.0,
                    quantization_error_beats=1.0,
                    warnings="unexpected_error",
                )
            )

    if args.pretty:
        print_pretty_summary(results)

    if args.csv:
        save_csv(results, args.csv)
        print(f"CSV guardado en: {args.csv}")

    if args.json:
        save_json(results, args.json)
        print(f"JSON guardado en: {args.json}")


# Ejecucion directa del script.
if __name__ == "__main__":
    main()