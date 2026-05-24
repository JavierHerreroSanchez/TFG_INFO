#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
inspect_tokenized_json_with_ac_v3.py

Inspector detallado de JSON tokenizados con MidiTok + Attribute Controls.

Mejoras respecto a v2:
- Más detalle por archivo y por track.
- Top 10 de familias por track y global.
- Conteos absolutos y porcentajes.
- Desglose específico de ACTrack / ACBar / Chord / Bar / Position / Pitch / Duration / Velocity / Rest / Tempo / TimeSig.
- Posiciones de aparición de tokens especiales.
- Preview inicial y ventanas alrededor de los primeros tokens importantes.
- Comparación exacta de ids guardados vs regenerados.

Uso:
- Ajusta TOKENIZER_PATH y TOKENS_DIR si hace falta.
- Ejecuta el script.
"""

from __future__ import annotations

import json
from collections import Counter
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

SAMPLE_LIMIT = None        # None para todos
TOKENS_PREVIEW = 160       # preview inicial de tokens
TOP_K_FAMILIES = 20
FIRST_N_SPECIAL_POS = 20   # primeras posiciones a mostrar por tipo
WINDOW_RADIUS = 20          # tokens alrededor de eventos importantes



# =============================================================================
# UTILIDADES
# =============================================================================

def list_json_files(root: Path) -> list[Path]:
    """
    Implementa la logica de list json files dentro del pipeline del TFG.

    Parametros principales: root.
    """

    return sorted(p for p in root.rglob("*.json") if p.is_file())


def load_json(path: Path) -> dict[str, Any]:
    """
    Carga los recursos necesarios para esta fase del pipeline.

    Parametros principales: path.
    """

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_saved_ids(ids: Any) -> list[list[int]]:
    """
    Implementa la logica de normalize saved ids dentro del pipeline del TFG.

    Parametros principales: ids.
    """

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
    """
    Implementa la logica de normalize attr indexes dentro del pipeline del TFG.

    Parametros principales: attr.
    """

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
    """
    Implementa la logica de normalize tokseq list dentro del pipeline del TFG.

    Parametros principales: tokseq_or_list.
    """

    if isinstance(tokseq_or_list, list):
        return tokseq_or_list
    return [tokseq_or_list]


def compare_ids(saved_ids_nested: list[list[int]], generated_ids_nested: list[list[int]]) -> tuple[bool, str]:
    """
    Implementa la logica de compare ids dentro del pipeline del TFG.

    Parametros principales: saved_ids_nested, generated_ids_nested.
    """

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
    """
    Implementa la logica de token family dentro del pipeline del TFG.

    Parametros principales: tok.
    """

    if "_" not in tok:
        return tok
    return tok.split("_", 1)[0]


def is_track_ac(tok: str) -> bool:
    """
    Implementa la logica de is track ac dentro del pipeline del TFG.

    Parametros principales: tok.
    """

    return tok.startswith("ACTrack")


def is_bar_ac(tok: str) -> bool:
    """
    Implementa la logica de is bar ac dentro del pipeline del TFG.

    Parametros principales: tok.
    """

    return tok.startswith("ACBar")


def pct(num: int, den: int) -> float:
    """
    Implementa la logica de pct dentro del pipeline del TFG.

    Parametros principales: num, den.
    """

    return 0.0 if den == 0 else (100.0 * num / den)


def format_top(counter: Counter, total: int, top_k: int = TOP_K_FAMILIES) -> list[str]:
    """
    Implementa la logica de format top dentro del pipeline del TFG.

    Parametros principales: counter, total, top_k.
    """

    out = []
    for fam, cnt in counter.most_common(top_k):
        out.append(f"{fam:<18} {cnt:>6}  ({pct(cnt, total):6.2f}%)")
    return out


def first_positions(tokens: list[str], pred, limit: int = FIRST_N_SPECIAL_POS) -> list[int]:
    """
    Implementa la logica de first positions dentro del pipeline del TFG.

    Parametros principales: tokens, pred, limit.
    """

    out = []
    for i, tok in enumerate(tokens):
        if pred(tok):
            out.append(i)
            if len(out) >= limit:
                break
    return out


def windows_around_positions(tokens: list[str], positions: list[int], radius: int = WINDOW_RADIUS, max_windows: int = 5) -> list[str]:
    """
    Implementa la logica de windows around positions dentro del pipeline del TFG.

    Parametros principales: tokens, positions, radius, max_windows.
    """

    windows = []
    for pos in positions[:max_windows]:
        lo = max(0, pos - radius)
        hi = min(len(tokens), pos + radius + 1)
        chunk = tokens[lo:hi]
        rel = pos - lo
        if 0 <= rel < len(chunk):
            chunk[rel] = f">>>{chunk[rel]}<<<"
        windows.append(f"[pos {pos}] " + " | ".join(chunk))
    return windows


def family_counter(tokens: list[str]) -> Counter:
    """
    Implementa la logica de family counter dentro del pipeline del TFG.

    Parametros principales: tokens.
    """

    return Counter(token_family(t) for t in tokens)


def specific_token_stats(tokens: list[str]) -> dict[str, int]:
    """
    Implementa la logica de specific token stats dentro del pipeline del TFG.

    Parametros principales: tokens.
    """

    fams = family_counter(tokens)
    return {
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
    }


def ac_family_counter(tokens: list[str]) -> Counter:
    """
    Implementa la logica de ac family counter dentro del pipeline del TFG.

    Parametros principales: tokens.
    """

    c = Counter()
    for tok in tokens:
        if tok.startswith("ACTrack"):
            c["ACTrack"] += 1
        elif tok.startswith("ACBar"):
            c["ACBar"] += 1
    return c


def inspect_track(tokens: list[str]) -> dict[str, Any]:
    """
    Muestra informacion de diagnostico para revisar artefactos del proyecto.

    Parametros principales: tokens.
    """

    fams = family_counter(tokens)
    specific = specific_token_stats(tokens)
    acs = ac_family_counter(tokens)

    bar_positions = [i for i, t in enumerate(tokens) if t == "Bar_None"]
    chord_positions = [i for i, t in enumerate(tokens) if t.startswith("Chord")]
    track_ac_positions = [i for i, t in enumerate(tokens) if is_track_ac(t)]
    bar_ac_positions = [i for i, t in enumerate(tokens) if is_bar_ac(t)]

    leading_track_acs = 0
    i = 0
    while i < len(tokens) and is_track_ac(tokens[i]):
        leading_track_acs += 1
        i += 1

    bars_with_bar_ac = 0
    bar_ac_counts_after_bar: list[int] = []
    for bidx in bar_positions:
        j = bidx + 1
        cnt = 0
        while j < len(tokens) and is_bar_ac(tokens[j]):
            cnt += 1
            j += 1
        bar_ac_counts_after_bar.append(cnt)
        if cnt > 0:
            bars_with_bar_ac += 1

    n_tokens = len(tokens)

    return {
        "n_tokens": n_tokens,
        "families": fams,
        "specific": specific,
        "acs": acs,
        "n_bars": len(bar_positions),
        "n_track_ac": acs["ACTrack"],
        "n_bar_ac": acs["ACBar"],
        "n_chords": specific["Chord"],
        "leading_track_acs": leading_track_acs,
        "bars_with_bar_ac": bars_with_bar_ac,
        "bar_ac_counts_after_bar": bar_ac_counts_after_bar,
        "bar_positions_first": bar_positions[:FIRST_N_SPECIAL_POS],
        "chord_positions_first": chord_positions[:FIRST_N_SPECIAL_POS],
        "track_ac_positions_first": track_ac_positions[:FIRST_N_SPECIAL_POS],
        "bar_ac_positions_first": bar_ac_positions[:FIRST_N_SPECIAL_POS],
        "preview": tokens[:TOKENS_PREVIEW],
        "windows_track_ac": windows_around_positions(tokens, track_ac_positions),
        "windows_bar_ac": windows_around_positions(tokens, bar_ac_positions),
        "windows_chord": windows_around_positions(tokens, chord_positions),
    }


def print_section(title: str) -> None:
    """
    Implementa la logica de print section dentro del pipeline del TFG.

    Parametros principales: title.
    """

    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)


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

    global_fams = Counter()
    global_specific = Counter()
    global_acs = Counter()

    inspected = 0
    n_exact_match = 0
    n_with_chords = 0
    n_with_track_ac = 0
    n_with_bar_ac = 0

    for path in json_files:
        print("\n" + "=" * 120)
        print(path)

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

            # Re-tokenización con ids para comprobar exactitud
            seq_ids = tokenizer.encode(
                score_pre,
                encode_ids=True,
                no_preprocess_score=True,
                attribute_controls_indexes=attr_indexes,
            )
            seq_ids_list = normalize_tokseq_list(seq_ids)
            gen_ids_nested = [seq.ids for seq in seq_ids_list]

            exact_match, exact_msg = compare_ids(saved_ids_nested, gen_ids_nested)

            # Re-tokenización legible
            seq_plain = tokenizer.encode(
                score_pre,
                encode_ids=False,
                no_preprocess_score=True,
                attribute_controls_indexes=attr_indexes,
            )
            seq_plain_list = normalize_tokseq_list(seq_plain)

            print(f"[CHECK] ids exactos: {exact_match} ({exact_msg})")
            print(f"[META ] source_midi: {source_midi}")
            print(f"[META ] attribute_controls_inserted: {obj.get('attribute_controls_inserted', None)}")
            print(f"[META ] n_tracks_after_preprocess(json): {obj.get('n_tracks_after_preprocess', None)}")
            print(f"[META ] attribute_controls_indexes keys: {list(obj.get('attribute_controls_indexes', {}).keys()) if obj.get('attribute_controls_indexes') else []}")
            print(f"[META ] n_streams_saved: {len(saved_ids_nested)} | n_streams_regen: {len(seq_plain_list)}")

            file_has_chords = False
            file_has_track_ac = False
            file_has_bar_ac = False

            per_file_fams = Counter()
            per_file_specific = Counter()
            per_file_acs = Counter()
            total_tokens_file = 0

            for t_idx, seq in enumerate(seq_plain_list):
                tokenizer.complete_sequence(seq, complete_bytes=False)
                tokens = seq.tokens if seq.tokens is not None else []

                info = inspect_track(tokens)

                total_tokens_file += info["n_tokens"]
                per_file_fams.update(info["families"])
                per_file_specific.update(info["specific"])
                per_file_acs.update(info["acs"])

                global_fams.update(info["families"])
                global_specific.update(info["specific"])
                global_acs.update(info["acs"])

                file_has_chords |= info["n_chords"] > 0
                file_has_track_ac |= info["n_track_ac"] > 0
                file_has_bar_ac |= info["n_bar_ac"] > 0

                print_section(f"TRACK {t_idx}")

                print(f"n_tokens                 = {info['n_tokens']}")
                print(f"n_bars                   = {info['n_bars']}")
                print(f"n_track_ac               = {info['n_track_ac']} ({pct(info['n_track_ac'], info['n_tokens']):.2f}%)")
                print(f"n_bar_ac                 = {info['n_bar_ac']} ({pct(info['n_bar_ac'], info['n_tokens']):.2f}%)")
                print(f"n_chords                 = {info['n_chords']} ({pct(info['n_chords'], info['n_tokens']):.2f}%)")
                print(f"leading_track_acs        = {info['leading_track_acs']}")
                print(f"bars_with_bar_ac         = {info['bars_with_bar_ac']} / {info['n_bars']}")
                if info["bar_ac_counts_after_bar"]:
                    mean_bar_ac = sum(info["bar_ac_counts_after_bar"]) / len(info["bar_ac_counts_after_bar"])
                    print(f"media ACBar tras Bar     = {mean_bar_ac:.2f}")

                print_section("Resumen de familias principales")
                for k, v in info["specific"].items():
                    print(f"{k:<24} {v:>6}  ({pct(v, info['n_tokens']):6.2f}%)")

                print_section(f"Top {TOP_K_FAMILIES} familias del track")
                for line in format_top(info["families"], info["n_tokens"], TOP_K_FAMILIES):
                    print(line)

                print_section("Top AC del track")
                if info["acs"]:
                    for line in format_top(info["acs"], info["n_tokens"], TOP_K_FAMILIES):
                        print(line)
                else:
                    print("(sin AC)")

                print_section("Primeras posiciones relevantes")
                print(f"Bar_None       : {info['bar_positions_first']}")
                print(f"Chord_*        : {info['chord_positions_first']}")
                print(f"ACTrack*       : {info['track_ac_positions_first']}")
                print(f"ACBar*         : {info['bar_ac_positions_first']}")

                print_section("Preview inicial")
                print(" | ".join(info["preview"]))

                print_section("Ventanas alrededor de ACTrack")
                if info["windows_track_ac"]:
                    for row in info["windows_track_ac"]:
                        print(row)
                else:
                    print("(sin ACTrack)")

                print_section("Ventanas alrededor de ACBar")
                if info["windows_bar_ac"]:
                    for row in info["windows_bar_ac"]:
                        print(row)
                else:
                    print("(sin ACBar)")

                print_section("Ventanas alrededor de Chord")
                if info["windows_chord"]:
                    for row in info["windows_chord"]:
                        print(row)
                else:
                    print("(sin Chord)")

            print_section("Resumen agregado del archivo")
            print(f"tokens totales archivo    = {total_tokens_file}")
            print(f"tracks/streams            = {len(seq_plain_list)}")
            print(f"contiene chords           = {file_has_chords}")
            print(f"contiene ACTrack          = {file_has_track_ac}")
            print(f"contiene ACBar            = {file_has_bar_ac}")

            print_section(f"Top {TOP_K_FAMILIES} familias del archivo")
            for line in format_top(per_file_fams, total_tokens_file, TOP_K_FAMILIES):
                print(line)

            print_section("Desglose específico del archivo")
            for k, v in per_file_specific.items():
                print(f"{k:<24} {v:>6}  ({pct(v, total_tokens_file):6.2f}%)")

            inspected += 1
            if exact_match:
                n_exact_match += 1
            if file_has_chords:
                n_with_chords += 1
            if file_has_track_ac:
                n_with_track_ac += 1
            if file_has_bar_ac:
                n_with_bar_ac += 1

        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")

    print("\n" + "=" * 120)
    print("RESUMEN GLOBAL")
    print("=" * 120)
    print(f"Archivos inspeccionados   : {inspected}")
    print(f"Con ids exactos           : {n_exact_match}")
    print(f"Con Chord                 : {n_with_chords}")
    print(f"Con AC de track           : {n_with_track_ac}")
    print(f"Con AC de bar             : {n_with_bar_ac}")

    total_global = sum(global_fams.values())

    print_section(f"Top {TOP_K_FAMILIES} familias globales")
    for line in format_top(global_fams, total_global, TOP_K_FAMILIES):
        print(line)

    print_section("Desglose global específico")
    for k, v in global_specific.most_common():
        print(f"{k:<24} {v:>8}  ({pct(v, total_global):6.2f}%)")

    print_section("Desglose global AC")
    for k, v in global_acs.most_common():
        print(f"{k:<24} {v:>8}  ({pct(v, total_global):6.2f}%)")


# Ejecucion directa del script.
if __name__ == "__main__":
    main()
