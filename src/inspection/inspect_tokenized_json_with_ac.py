
"""
Inspeccion de JSONs tokenizados con attribute controls.

El script carga los JSON generados por el flujo de tokenizacion principal,
retokeniza el MIDI de origen y compara los ids guardados con los reconstruidos.
Tambien muestra una lectura resumida de AC, acordes, barras y otros eventos.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from miditok import REMI
from symusic import Score

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TOKENIZER_PATH = (PROJECT_ROOT / "tokenizer" / "tokenizer_REMI_BPE_v2.json").resolve()
TOKENS_DIR = (PROJECT_ROOT / "data" / "interim" / "tokenized_json_bpe_v2").resolve()

# Límite opcional de archivos a inspeccionar. None procesa todos.
SAMPLE_LIMIT = None
TOKENS_PREVIEW = 200


# =============================================================================
# UTILIDADES
# =============================================================================

def list_json_files(root: Path) -> list[Path]:

    return sorted(p for p in root.rglob("*.json") if p.is_file())


def token_family(tok: str) -> str:

    if "_" not in tok:
        return tok
    return tok.split("_", 1)[0]


def is_track_ac(tok: str) -> bool:

    return tok.startswith("ACTrack")

def is_bar_ac(tok: str) -> bool:

    return tok.startswith("ACBar")


def normalize_tokseq_list(tokseq_or_list: Any) -> list[Any]:

    if isinstance(tokseq_or_list, list):
        return tokseq_or_list
    return [tokseq_or_list]


def normalize_saved_ids(ids: Any) -> list[list[int]]:
    """
    Convierte el contenido del campo 'ids' a:
        [[...]] para un solo stream
        [[...], [...]] para multi-stream
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
    Convierte claves str -> int si vienen de JSON.
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


def inspect_track(tokens: list[str]) -> dict[str, Any]:
    """
    Muestra informacion de diagnostico para revisar artefactos del proyecto.

    """

    fams = Counter(token_family(t) for t in tokens)

    bar_positions = [i for i, t in enumerate(tokens) if t == "Bar_None"]

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

    return {
        "n_tokens": len(tokens),
        "n_bars": len(bar_positions),
        "n_track_ac": sum(1 for t in tokens if is_track_ac(t)),
        "n_bar_ac": sum(1 for t in tokens if is_bar_ac(t)),
        "n_chords": fams.get("Chord", 0),
        "n_rests": fams.get("Rest", 0),
        "n_positions": fams.get("Position", 0),
        "n_pitch": fams.get("Pitch", 0),
        "leading_track_acs": leading_track_acs,
        "bars_with_bar_ac": bars_with_bar_ac,
        "families": dict(fams),
        "preview": tokens[:TOKENS_PREVIEW],
    }


def load_json(path: Path) -> dict[str, Any]:
    """Carga un JSON tokenizado para compararlo con la retokenización."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
    n_exact_match = 0
    n_with_chords = 0
    n_with_track_ac = 0
    n_with_bar_ac = 0
    inspected = 0

    for path in json_files:
        print("\n" + "=" * 100)
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

            # Cargar y preprocesar Score
            score = Score(source_midi)
            score_pre = tokenizer.preprocess_score(score)

            # 1) Re-tokenización exacta con ids (BPE)
            seq_ids = tokenizer.encode(
                score_pre,
                encode_ids=True,
                no_preprocess_score=True,
                attribute_controls_indexes=attr_indexes,
            )
            seq_ids_list = normalize_tokseq_list(seq_ids)
            gen_ids_nested = [seq.ids for seq in seq_ids_list]

            exact_match, exact_msg = compare_ids(saved_ids_nested, gen_ids_nested)

            # 2) Re-tokenización legible sin BPE
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
            print(f"[META ] n_streams_saved: {len(saved_ids_nested)} | n_streams_regen: {len(seq_plain_list)}")

            file_has_chords = False
            file_has_track_ac = False
            file_has_bar_ac = False

            for t_idx, seq in enumerate(seq_plain_list):
                tokenizer.complete_sequence(seq, complete_bytes=False)
                tokens = seq.tokens if seq.tokens is not None else []
                info = inspect_track(tokens)

                global_fams.update(info["families"])
                file_has_chords |= info["n_chords"] > 0
                file_has_track_ac |= info["n_track_ac"] > 0
                file_has_bar_ac |= info["n_bar_ac"] > 0

                print(f"\n[TRACK {t_idx}]")
                print(f"  n_tokens           = {info['n_tokens']}")
                print(f"  n_bars             = {info['n_bars']}")
                print(f"  n_track_ac         = {info['n_track_ac']}")
                print(f"  n_bar_ac           = {info['n_bar_ac']}")
                print(f"  n_chords           = {info['n_chords']}")
                print(f"  n_rests            = {info['n_rests']}")
                print(f"  n_positions        = {info['n_positions']}")
                print(f"  n_pitch            = {info['n_pitch']}")
                print(f"  leading_track_acs  = {info['leading_track_acs']}")
                print(f"  bars_with_bar_ac   = {info['bars_with_bar_ac']} / {info['n_bars']}")
                top12 = dict(sorted(info["families"].items(), key=lambda kv: (-kv[1], kv[0]))[:12])
                print(f"  familias top       = {top12}")
                print("  preview:")
                print("   ", " | ".join(info["preview"]))

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

    print("\n" + "=" * 100)
    print("RESUMEN GLOBAL")
    print("=" * 100)
    print(f"Archivos inspeccionados : {inspected}")
    print(f"Con ids exactos         : {n_exact_match}")
    print(f"Con Chord               : {n_with_chords}")
    print(f"Con AC de track         : {n_with_track_ac}")
    print(f"Con AC de bar           : {n_with_bar_ac}")
    print("Familias globales top   :")
    for fam, cnt in global_fams.most_common(20):
        print(f"  {fam:<20} {cnt}")


# Ejecucion directa del script.
if __name__ == "__main__":
    main()
