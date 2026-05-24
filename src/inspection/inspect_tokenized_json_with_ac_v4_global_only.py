#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
inspect_tokenized_json_with_ac_v4_global_only.py

Versión centrada SOLO en métricas globales.
Objetivo:
- No imprimir detalle por archivo ni por track.
- Mostrar un resumen global muy rico de familias de tokens.
- Mostrar con mucho detalle los Attribute Controls (AC):
    * familias AC
    * tokens AC exactos más frecuentes
    * valores observados por familia AC
    * cobertura por número de archivos
    * porcentajes globales
- Mantener la comprobación de ids exactos a nivel global.

Pensado para JSONs tokenizados con MidiTok y AC ya insertados.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from miditok import REMI
from symusic import Score


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TOKENIZER_PATH = (PROJECT_ROOT / "tokenizer" / "tokenizer_REMI_BPE_v4.json").resolve()
TOKENS_DIR = (PROJECT_ROOT / "data" / "interim" / "tokenized_json_bpe_v2").resolve()

SAMPLE_LIMIT = None          # None para todos
TOP_K_FAMILIES = 30
TOP_K_TOKENS = 30
TOP_K_AC_TOKENS = 50
TOP_K_VALUES_PER_AC_FAMILY = 35


# =============================================================================
# UTILIDADES
# =============================================================================

def list_json_files(root: Path) -> list[Path]:

    return sorted(p for p in root.rglob("*.json") if p.is_file())


def load_json(path: Path) -> dict[str, Any]:
    """Carga un JSON tokenizado para compararlo con la retokenización."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_saved_ids(ids: Any) -> list[list[int]]:

    if not isinstance(ids, list):
        raise TypeError(f"'ids' debe ser list, no {type(ids)}")

    if len(ids) == 0:
        return [[]]

    if isinstance(ids[0], int):
        return [ids]

    if isinstance(ids[0], list):
        return ids

    raise TypeError(f"Formato de 'ids' no reconocido: primer elemento {type(ids[0])}")


def normalize_attr_indexes(attr: Any) -> dict[int, dict[int, Any]] | None:

    if not attr:
        return None

    out: dict[int, dict[int, Any]] = {}
    for track_k, per_track in attr.items():
        track_idx = int(track_k)
        out[track_idx] = {}
        for ac_k, ac_val in per_track.items():
            out[track_idx][int(ac_k)] = ac_val
    return out


def normalize_tokseq_list(tokseq_or_list: Any) -> list[Any]:

    if isinstance(tokseq_or_list, list):
        return tokseq_or_list
    return [tokseq_or_list]


def compare_ids(saved_ids_nested: list[list[int]], generated_ids_nested: list[list[int]]) -> tuple[bool, str]:

    if len(saved_ids_nested) != len(generated_ids_nested):
        return False, f"n_streams distinto: saved={len(saved_ids_nested)} generated={len(generated_ids_nested)}"

    for i, (a, b) in enumerate(zip(saved_ids_nested, generated_ids_nested)):
        if len(a) != len(b):
            return False, f"stream {i}: longitud distinta saved={len(a)} generated={len(b)}"

        for j, (xa, xb) in enumerate(zip(a, b)):
            if xa != xb:
                return False, f"stream {i}: primer mismatch en pos {j}: saved={xa} generated={xb}"

    return True, "ok"


def token_family(tok: str) -> str:

    if "_" not in tok:
        return tok
    return tok.split("_", 1)[0]


def is_track_ac(tok: str) -> bool:

    return tok.startswith("ACTrack")


def is_bar_ac(tok: str) -> bool:

    return tok.startswith("ACBar")


def is_ac(tok: str) -> bool:

    return tok.startswith("ACTrack") or tok.startswith("ACBar")


def pct(num: int, den: int) -> float:

    return 0.0 if den == 0 else (100.0 * num / den)


def print_section(title: str) -> None:

    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)


def format_counter(counter: Counter, total: int, top_k: int) -> list[str]:

    rows = []
    for key, cnt in counter.most_common(top_k):
        rows.append(f"{str(key):<40} {cnt:>10}  ({pct(cnt, total):7.3f}%)")
    return rows


def parse_ac_token(tok: str) -> tuple[str, str] | None:
    """
    Devuelve:
      ("ACTrackRepetition", "0.00")
      ("ACBarPitchClass", "7")
      ...
    """
    if not is_ac(tok):
        return None

    if "_" not in tok:
        return tok, ""

    fam, value = tok.rsplit("_", 1)
    return fam, value


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Punto de entrada del script cuando se ejecuta desde consola."""

    print(f"[INFO] TOKENIZER = {TOKENIZER_PATH}")
    print(f"[INFO] TOKENS_DIR = {TOKENS_DIR}")

    tokenizer = REMI(params=TOKENIZER_PATH)

    json_files = list_json_files(TOKENS_DIR)
    if SAMPLE_LIMIT is not None:
        json_files = json_files[:SAMPLE_LIMIT]

    print(f"[INFO] JSONs a inspeccionar = {len(json_files)}")
    if not json_files:
        return

    # Resumen general
    inspected = 0
    n_exact_match = 0
    n_with_chords = 0
    n_with_track_ac = 0
    n_with_bar_ac = 0

    # Conteos globales
    global_fams = Counter()
    global_tokens = Counter()
    global_ac_tokens = Counter()

    # Cobertura por archivo
    fam_files = Counter()
    ac_family_files = Counter()
    ac_token_files = Counter()

    # Desglose específico
    global_specific = Counter()
    global_acs_coarse = Counter()   # ACTrack / ACBar

    # Desglose fino AC
    ac_family_counts = Counter()                       # ACBarPitchClass, ACTrackRepetition...
    ac_family_value_counts: dict[str, Counter] = defaultdict(Counter)  # familia -> valores
    ac_family_total_by_file: dict[str, list[int]] = defaultdict(list)   # stats de frecuencia por archivo

    total_global_tokens = 0

    for idx, path in enumerate(json_files, start=1):
        try:
            obj = load_json(path)

            if "ids" not in obj:
                raise KeyError("El JSON no contiene el campo 'ids'")
            if "source_midi" not in obj:
                raise KeyError("El JSON no contiene 'source_midi'")

            source_midi = Path(obj["source_midi"])
            if not source_midi.exists():
                raise FileNotFoundError(f"No existe source_midi: {source_midi}")

            saved_ids_nested = normalize_saved_ids(obj["ids"])
            attr_indexes = normalize_attr_indexes(obj.get("attribute_controls_indexes", None))

            score = Score(source_midi)
            score_pre = tokenizer.preprocess_score(score)

            # Exactitud ids
            seq_ids = tokenizer.encode(
                score_pre,
                encode_ids=True,
                no_preprocess_score=True,
                attribute_controls_indexes=attr_indexes,
            )
            seq_ids_list = normalize_tokseq_list(seq_ids)
            gen_ids_nested = [seq.ids for seq in seq_ids_list]
            exact_match, _ = compare_ids(saved_ids_nested, gen_ids_nested)

            # Tokens legibles
            seq_plain = tokenizer.encode(
                score_pre,
                encode_ids=False,
                no_preprocess_score=True,
                attribute_controls_indexes=attr_indexes,
            )
            seq_plain_list = normalize_tokseq_list(seq_plain)

            file_fams = Counter()
            file_ac_fams = Counter()
            file_ac_tokens = Counter()
            file_has_chords = False
            file_has_track_ac = False
            file_has_bar_ac = False

            # Para stats por archivo de cada familia AC
            file_ac_family_counts = Counter()

            for seq in seq_plain_list:
                tokenizer.complete_sequence(seq, complete_bytes=False)
                tokens = seq.tokens if seq.tokens is not None else []

                total_global_tokens += len(tokens)

                fams = Counter(token_family(t) for t in tokens)
                file_fams.update(fams)
                global_fams.update(fams)
                global_tokens.update(tokens)

                if fams.get("Chord", 0) > 0:
                    file_has_chords = True

                ac_coarse_track = 0
                ac_coarse_bar = 0

                for tok in tokens:
                    if tok.startswith("ACTrack"):
                        ac_coarse_track += 1
                    elif tok.startswith("ACBar"):
                        ac_coarse_bar += 1

                    parsed = parse_ac_token(tok)
                    if parsed is not None:
                        ac_family, ac_value = parsed
                        file_ac_fams[ac_family] += 1
                        file_ac_tokens[tok] += 1

                        ac_family_counts[ac_family] += 1
                        ac_family_value_counts[ac_family][ac_value] += 1
                        file_ac_family_counts[ac_family] += 1
                        global_ac_tokens[tok] += 1

                global_specific.update({
                    "Bar": fams.get("Bar", 0),
                    "Position": fams.get("Position", 0),
                    "Pitch": fams.get("Pitch", 0),
                    "Duration": fams.get("Duration", 0),
                    "Velocity": fams.get("Velocity", 0),
                    "Rest": fams.get("Rest", 0),
                    "Tempo": fams.get("Tempo", 0),
                    "TimeSig": fams.get("TimeSig", 0),
                    "Chord": fams.get("Chord", 0),
                    "Program": fams.get("Program", 0),
                    "Pedal": fams.get("Pedal", 0),
                    "PedalOff": fams.get("PedalOff", 0),
                    "PitchIntervalTime": fams.get("PitchIntervalTime", 0),
                    "PitchIntervalChord": fams.get("PitchIntervalChord", 0),
                })

                global_acs_coarse["ACTrack"] += ac_coarse_track
                global_acs_coarse["ACBar"] += ac_coarse_bar

                if ac_coarse_track > 0:
                    file_has_track_ac = True
                if ac_coarse_bar > 0:
                    file_has_bar_ac = True

            # Cobertura por archivo
            for fam in file_fams:
                fam_files[fam] += 1
            for acfam in file_ac_fams:
                ac_family_files[acfam] += 1
            for actok in file_ac_tokens:
                ac_token_files[actok] += 1
            for acfam, cnt in file_ac_family_counts.items():
                ac_family_total_by_file[acfam].append(cnt)

            inspected += 1
            if exact_match:
                n_exact_match += 1
            if file_has_chords:
                n_with_chords += 1
            if file_has_track_ac:
                n_with_track_ac += 1
            if file_has_bar_ac:
                n_with_bar_ac += 1

            if idx % 250 == 0 or idx == len(json_files):
                print(f"[INFO] {idx}/{len(json_files)} | ok={inspected} exact={n_exact_match}")

        except Exception as e:
            print(f"[ERROR] {path} -> {type(e).__name__}: {e}")

    # =========================================================================
    # SALIDA GLOBAL
    # =========================================================================
    print("\n" + "=" * 120)
    print("RESUMEN GLOBAL")
    print("=" * 120)
    print(f"Archivos inspeccionados   : {inspected}")
    print(f"Con ids exactos           : {n_exact_match}")
    print(f"Con Chord                 : {n_with_chords}")
    print(f"Con AC de track           : {n_with_track_ac}")
    print(f"Con AC de bar             : {n_with_bar_ac}")
    print(f"Tokens globales           : {total_global_tokens}")

    print_section(f"Top {TOP_K_FAMILIES} familias globales")
    for line in format_counter(global_fams, total_global_tokens, TOP_K_FAMILIES):
        print(line)

    print_section("Cobertura por familia (nº de archivos donde aparece)")
    cov_rows = sorted(fam_files.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_K_FAMILIES]
    for fam, cnt in cov_rows:
        print(f"{fam:<40} {cnt:>10}  ({pct(cnt, inspected):7.3f}% de archivos)")

    print_section(f"Top {TOP_K_TOKENS} tokens exactos globales")
    for tok, cnt in global_tokens.most_common(TOP_K_TOKENS):
        print(f"{tok:<40} {cnt:>10}  ({pct(cnt, total_global_tokens):7.3f}%)")

    print_section("Desglose global específico")
    for k, v in global_specific.most_common():
        print(f"{k:<40} {v:>10}  ({pct(v, total_global_tokens):7.3f}%)")

    print_section("Desglose global AC (grueso)")
    for k, v in global_acs_coarse.most_common():
        print(f"{k:<40} {v:>10}  ({pct(v, total_global_tokens):7.3f}%)")

    print_section("Familias AC detalladas")
    total_ac_tokens = sum(ac_family_counts.values())
    for acfam, cnt in ac_family_counts.most_common():
        coverage = ac_family_files.get(acfam, 0)
        vals = len(ac_family_value_counts[acfam])
        file_counts = ac_family_total_by_file.get(acfam, [])
        avg_per_file = (sum(file_counts) / len(file_counts)) if file_counts else 0.0
        print(
            f"{acfam:<40} "
            f"count={cnt:>10}  "
            f"global={pct(cnt, total_global_tokens):7.3f}%  "
            f"within_AC={pct(cnt, total_ac_tokens):7.3f}%  "
            f"files={coverage:>6} ({pct(coverage, inspected):6.2f}%)  "
            f"n_values={vals:>4}  "
            f"avg/file={avg_per_file:7.2f}"
        )

    print_section(f"Top {TOP_K_AC_TOKENS} tokens AC exactos")
    for tok, cnt in global_ac_tokens.most_common(TOP_K_AC_TOKENS):
        coverage = ac_token_files.get(tok, 0)
        parsed = parse_ac_token(tok)
        acfam, acval = parsed if parsed is not None else ("<?>", "<?>")
        print(
            f"{tok:<40} "
            f"count={cnt:>10}  "
            f"global={pct(cnt, total_global_tokens):7.3f}%  "
            f"files={coverage:>6} ({pct(coverage, inspected):6.2f}%)  "
            f"family={acfam:<28} value={acval}"
        )

    print_section("Valores observados por familia AC")
    for acfam, counter in sorted(ac_family_value_counts.items(), key=lambda kv: (-sum(kv[1].values()), kv[0])):
        fam_total = sum(counter.values())
        print(f"\n[{acfam}] total={fam_total}  files={ac_family_files.get(acfam, 0)}")
        for value, cnt in counter.most_common(TOP_K_VALUES_PER_AC_FAMILY):
            print(
                f"  value={value:<16} "
                f"count={cnt:>10}  "
                f"within_family={pct(cnt, fam_total):7.3f}%  "
                f"global={pct(cnt, total_global_tokens):7.3f}%"
            )

    print_section("Diagnóstico rápido")
    chord_pct = pct(global_specific["Chord"], total_global_tokens)
    actrack_pct = pct(global_acs_coarse["ACTrack"], total_global_tokens)
    acbar_pct = pct(global_acs_coarse["ACBar"], total_global_tokens)
    print(f"- Chord:   {chord_pct:.4f}%")
    print(f"- ACTrack: {actrack_pct:.4f}%")
    print(f"- ACBar:   {acbar_pct:.4f}%")
    if global_specific["Chord"] == 0:
        print("- No hay tokens Chord.")
    elif chord_pct < 0.05:
        print("- Los Chord son muy escasos frente al resto de la representación.")
    if global_acs_coarse["ACBar"] > global_acs_coarse["ACTrack"]:
        print("- Los AC bar-level dominan claramente sobre los track-level, como cabría esperar.")
    if n_exact_match == 0:
        print("- Los ids regenerados no coinciden exactamente con los guardados; úsalo solo como chequeo secundario.")
    else:
        print("- Hay coincidencia exacta de ids en al menos parte del corpus.")


# Ejecucion directa del script.
if __name__ == "__main__":
    main()
