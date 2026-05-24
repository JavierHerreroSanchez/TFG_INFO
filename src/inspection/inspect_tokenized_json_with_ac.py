
"""
inspect_tokenized_json_with_ac.py

Objetivo:
- Comprobar que los JSON tokenizados contienen:
  * AC de track al principio
  * AC de bar al comienzo de las barras
  * acordes, rests, time signatures, etc.
- Sacar un informe legible y una vista previa de tokens.

Uso típico:
- Ejecutar el script despues de tokenizar.
- Comprobar el resumen global.
- Para inspeccionar casos concretos, aumentar SAMPLE_FILES o SAMPLE_LIMIT.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from miditok import REMI

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TOKENIZER_PATH = (PROJECT_ROOT / "tokenizer" / "tokenizer_REMI_BPE_v4.json").resolve()
TOKENS_DIR = (PROJECT_ROOT / "data" / "interim" / "tokenized_json_bpe_v2").resolve()

SAMPLE_LIMIT = 20
TOKENS_PREVIEW = 80


# =============================================================================
# UTILIDADES
# =============================================================================

def list_json_files(root: Path) -> list[Path]:

    return sorted(p for p in root.rglob("*.json") if p.is_file())


def family(tok: str) -> str:

    return tok.split("_", 1)[0] if "_" in tok else tok


def is_track_ac(tok: str) -> bool:

    return tok.startswith("ACTrack")

def is_bar_ac(tok: str) -> bool:

    return tok.startswith("ACBar")


def normalize_loaded_tokens(tokenizer: REMI, path: Path) -> list[list[str]]:
    """
    Devuelve una lista de listas de tokens (una por pista).
    Soporta:
    - JSONs guardados con tokenizer.save_tokens(...)
    - JSONs multitrack guardados manualmente con {"ids": [[...], [...]]}
    """
    raw = tokenizer.load_tokens(path, raw=True)

    ids = raw.get("ids")
    if ids is None:
        raise ValueError(f"El JSON no contiene 'ids': {path}")

    # Caso 1: una sola pista => lista de ints
    if ids and isinstance(ids[0], int):
        seq = tokenizer.load_tokens(path, raw=False)
        tokenizer.decode_token_ids(seq)
        tokenizer.complete_sequence(seq, complete_bytes=False)
        return [seq.tokens or []]

    # Caso 2: varias pistas => lista de listas
    if ids and isinstance(ids[0], list):
        tracks_tokens: list[list[str]] = []
        for ids_track in ids:
            # Creamos una secuencia mínima usando la API del tokenizer
            # completando desde ids.
            from miditok import TokSequence
            seq = TokSequence(ids=ids_track)
            tokenizer.decode_token_ids(seq)
            tokenizer.complete_sequence(seq, complete_bytes=False)
            tracks_tokens.append(seq.tokens or [])
        return tracks_tokens

    raise ValueError(f"Formato de 'ids' no reconocido: {path}")


def inspect_track(tokens: list[str]) -> dict[str, Any]:
    """
    Muestra informacion de diagnostico para revisar artefactos del proyecto.

    """

    fams = Counter(family(t) for t in tokens)

    first_bar_idx = next((i for i, t in enumerate(tokens) if t == "Bar_None"), None)
    first_music_idx = next(
        (
            i for i, t in enumerate(tokens)
            if not is_track_ac(t) and not is_bar_ac(t)
        ),
        None,
    )

    # AC de track: deben ir antes de la música real
    leading_track_acs = 0
    i = 0
    while i < len(tokens) and is_track_ac(tokens[i]):
        leading_track_acs += 1
        i += 1

    # AC de bar: tras cada Bar_None, contamos el bloque continuo de ACBar*
    bar_positions = [i for i, t in enumerate(tokens) if t == "Bar_None"]
    bars_with_bar_ac = 0
    bar_ac_counts_after_bar: list[int] = []

    for bidx in bar_positions:
        j = bidx + 1
        count = 0
        while j < len(tokens) and is_bar_ac(tokens[j]):
            count += 1
            j += 1
        bar_ac_counts_after_bar.append(count)
        if count > 0:
            bars_with_bar_ac += 1

    return {
        "n_tokens": len(tokens),
        "families": dict(fams),
        "n_chords": fams.get("Chord", 0),
        "n_track_ac": sum(1 for t in tokens if is_track_ac(t)),
        "n_bar_ac": sum(1 for t in tokens if is_bar_ac(t)),
        "leading_track_acs": leading_track_acs,
        "first_bar_idx": first_bar_idx,
        "first_music_idx": first_music_idx,
        "n_bars": len(bar_positions),
        "bars_with_bar_ac": bars_with_bar_ac,
        "bar_ac_counts_after_bar": bar_ac_counts_after_bar,
        "preview": tokens[:TOKENS_PREVIEW],
    }


def main() -> None:
    """Punto de entrada del script cuando se ejecuta desde consola."""

    print(f"[INFO] TOKENIZER = {TOKENIZER_PATH}")
    print(f"[INFO] TOKENS_DIR = {TOKENS_DIR}")

    tokenizer = REMI(params=TOKENIZER_PATH)

    json_files = list_json_files(TOKENS_DIR)
    print(f"[INFO] JSONs encontrados = {len(json_files)}")
    if not json_files:
        return

    global_fams = Counter()
    files_with_chords = 0
    files_with_track_ac = 0
    files_with_bar_ac = 0
    inspected = 0

    for path in json_files[:SAMPLE_LIMIT]:
        try:
            per_track_tokens = normalize_loaded_tokens(tokenizer, path)

            print("\n" + "=" * 100)
            print(path)

            file_has_chords = False
            file_has_track_ac = False
            file_has_bar_ac = False

            for track_idx, tokens in enumerate(per_track_tokens):
                info = inspect_track(tokens)
                global_fams.update(info["families"])

                file_has_chords |= info["n_chords"] > 0
                file_has_track_ac |= info["n_track_ac"] > 0
                file_has_bar_ac |= info["n_bar_ac"] > 0

                print(f"\n[TRACK {track_idx}]")
                print(f"  n_tokens              = {info['n_tokens']}")
                print(f"  n_bars                = {info['n_bars']}")
                print(f"  n_track_ac            = {info['n_track_ac']}")
                print(f"  n_bar_ac              = {info['n_bar_ac']}")
                print(f"  n_chords              = {info['n_chords']}")
                print(f"  leading_track_acs     = {info['leading_track_acs']}")
                print(f"  first_bar_idx         = {info['first_bar_idx']}")
                print(f"  first_music_idx       = {info['first_music_idx']}")
                print(f"  bars_with_bar_ac      = {info['bars_with_bar_ac']} / {info['n_bars']}")
                print(f"  familias principales  = {dict(sorted(info['families'].items(), key=lambda kv: (-kv[1], kv[0]))[:12])}")
                print("  preview:")
                print("   ", " | ".join(info["preview"]))

            if file_has_chords:
                files_with_chords += 1
            if file_has_track_ac:
                files_with_track_ac += 1
            if file_has_bar_ac:
                files_with_bar_ac += 1

            inspected += 1

        except Exception as exc:
            print("\n" + "=" * 100)
            print(path)
            print(f"[ERROR] {type(exc).__name__}: {exc}")

    print("\n" + "=" * 100)
    print("RESUMEN GLOBAL")
    print("=" * 100)
    print(f"Archivos inspeccionados : {inspected}")
    print(f"Con Chord               : {files_with_chords}")
    print(f"Con AC de track         : {files_with_track_ac}")
    print(f"Con AC de bar           : {files_with_bar_ac}")
    print("Familias globales top   :")
    for fam, cnt in global_fams.most_common(20):
        print(f"  {fam:<20} {cnt}")


# Ejecucion directa del script.
if __name__ == "__main__":
    main()
