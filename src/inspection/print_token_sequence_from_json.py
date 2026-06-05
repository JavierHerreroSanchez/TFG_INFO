"""
print_token_sequence_from_json.py

Carga un JSON tokenizado por MidiTok y muestra secuencialmente:
- índice
- id
- token legible

Pensado para JSONs tokenizados con tokenizer_REMI_BPE_v2.json.

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from miditok import REMI
from miditok import TokSequence


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TOKENIZER_PATH = (PROJECT_ROOT / "tokenizer" / "tokenizer_REMI_BPE_v2.json").resolve()

# Ruta del JSON a inspeccionar.
JSON_PATH = (PROJECT_ROOT / "data" / "interim" / "tokenized_json_bpe_v2" / "ariamidi"  / "aa" / "000002_0.json"
).resolve()

# Stream a imprimir cuando el JSON contiene varios streams.
STREAM_INDEX = 0

# None => imprime todos
MAX_TOKENS: int | None = 6000

# True => añade metadatos del JSON al principio
PRINT_METADATA = True


# =============================================================================
# UTILIDADES
# =============================================================================

def load_json(path: Path) -> dict[str, Any]:
    """Carga un JSON tokenizado para inspeccionarlo."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_ids(ids_field: Any) -> list[list[int]]:
    """
    Normaliza el campo 'ids' a una lista de streams:
      [ [id, id, ...], [id, id, ...], ... ]
    """
    if not isinstance(ids_field, list):
        raise TypeError(f"El campo 'ids' no es una lista: {type(ids_field)}")

    if len(ids_field) == 0:
        return [[]]

    if isinstance(ids_field[0], int):
        return [ids_field]

    if isinstance(ids_field[0], list):
        return ids_field

    raise TypeError(f"Formato de 'ids' no reconocido: {type(ids_field[0])}")


def ids_to_tokens(tokenizer: REMI, ids: list[int]) -> list[str]:
    """
    Convierte ids a tokens legibles de forma robusta:
    1) crea TokSequence(ids=...)
    2) decode_token_ids => deshace BPE / modelo
    3) complete_sequence => rellena .tokens
    """
    seq = TokSequence(ids=ids)
    tokenizer.decode_token_ids(seq)
    tokenizer.complete_sequence(seq, complete_bytes=False)

    if seq.tokens is None:
        raise ValueError("No se pudieron reconstruir tokens desde los ids.")

    return seq.tokens


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Punto de entrada del script cuando se ejecuta desde consola."""

    print(f"[INFO] TOKENIZER_PATH = {TOKENIZER_PATH}")
    print(f"[INFO] JSON_PATH      = {JSON_PATH}")

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"No existe el tokenizer: {TOKENIZER_PATH}")
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"No existe el JSON: {JSON_PATH}")

    tokenizer = REMI(params=TOKENIZER_PATH)
    obj = load_json(JSON_PATH)

    if "ids" not in obj:
        raise KeyError("El JSON no contiene el campo 'ids'")

    ids_streams = normalize_ids(obj["ids"])

    if STREAM_INDEX < 0 or STREAM_INDEX >= len(ids_streams):
        raise IndexError(
            f"STREAM_INDEX={STREAM_INDEX} fuera de rango. "
            f"El JSON tiene {len(ids_streams)} stream(s)."
        )

    ids = ids_streams[STREAM_INDEX]
    tokens = ids_to_tokens(tokenizer, ids)

    if len(tokens) != len(ids):
        print(
            f"[WARN] nº tokens reconstruidos ({len(tokens)}) != nº ids originales ({len(ids)}). "
            "Esto puede ocurrir si el tokenizer aplica una decodificación de ids entrenados."
        )

    if PRINT_METADATA:
        print("\n" + "=" * 100)
        print("METADATA")
        print("=" * 100)
        for key in [
            "source_midi",
            "tokenizer_path",
            "attribute_controls_inserted",
            "n_tracks_after_preprocess",
            "bars_per_track_from_pretokenization",
            "total_ids",
            "total_tokens_after_bpe_decode",
        ]:
            if key in obj:
                print(f"{key}: {obj[key]}")
        if "attribute_controls_indexes" in obj:
            print(f"attribute_controls_indexes keys: {list(obj['attribute_controls_indexes'].keys())}")
        print(f"n_streams in json: {len(ids_streams)}")
        print(f"selected stream  : {STREAM_INDEX}")
        print(f"ids in stream    : {len(ids)}")
        print(f"tokens decoded   : {len(tokens)}")

    print("\n" + "=" * 100)
    print("SECUENCIA DE TOKENS")
    print("=" * 100)

    limit = len(tokens) if MAX_TOKENS is None else min(MAX_TOKENS, len(tokens))

    for i in range(limit):
        tok = tokens[i]
        id_val = ids[i] if i < len(ids) else "<?>"
        print(f"{i:06d} | {id_val:>6} | {tok}")

    if limit < len(tokens):
        print("\n" + "=" * 100)
        print(f"[INFO] Mostrados {limit} de {len(tokens)} tokens.")
        print("Establecer MAX_TOKENS = None para imprimirlos todos.")


# Ejecución directa del script.
if __name__ == "__main__":
    main()
