"""
Prepara la representacion simbolica que alimenta al modelo generativo.

El script convierte corpus MIDI en indices, tokens o secuencias de ids para que las fases de entrenamiento trabajen con tensores reproducibles.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


TOKEN_FIELD_CANDIDATES = ("ids", "ids_encoded")
INDEX_COLUMNS = ["path", "length", "min_id", "max_id"]


def flatten_ids(value) -> list[int]:
    """Devuelve una lista plana de ids desde JSONs single-track o multi-track."""
    if not isinstance(value, list):
        return []

    flat: list[int] = []
    for item in value:
        if isinstance(item, int):
            flat.append(item)
        elif isinstance(item, list):
            flat.extend(x for x in item if isinstance(x, int))
    return flat


def extract_ids(obj: dict) -> list[int]:
    """Busca el campo de ids compatible con las distintas versiones de MidiTok."""
    for field in TOKEN_FIELD_CANDIDATES:
        ids = flatten_ids(obj.get(field))
        if ids:
            return ids
    return []


def path_for_index(json_path: Path, project_root: Path | None) -> str:
    """Guarda rutas relativas al proyecto cuando es posible para evitar atarlas a una maquina."""
    resolved = json_path.resolve()
    if project_root is None:
        return str(resolved)

    try:
        return str(resolved.relative_to(project_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(resolved)


def iter_token_jsons(tokens_dir: Path) -> Iterable[Path]:
    """
    Implementa la logica de iter token jsons dentro del pipeline del TFG.

    Parametros principales: tokens_dir.
    """

    return sorted(p for p in tokens_dir.rglob("*.json") if p.is_file())


def build_token_index(tokens_dir: Path, out_csv: Path, project_root: Path | None = None) -> pd.DataFrame:
    """Construye el CSV de índice usado después por pretraining/finetuning.

    Columnas:
    - path: JSON tokenizado, preferiblemente relativo al root del proyecto
    - length: numero total de ids
    - min_id / max_id: comprobacion rapida del rango de vocabulario
    """
    if not tokens_dir.exists():
        raise FileNotFoundError(f"No existe TOKENS_DIR: {tokens_dir}")

    json_paths = list(iter_token_jsons(tokens_dir))
    print(f"[INDEX] JSON encontrados: {len(json_paths)}")
    print(f"[INDEX] TOKENS_DIR: {tokens_dir}")
    print(f"[INDEX] OUT_CSV: {out_csv}")

    rows = []
    bad = 0

    for i, json_path in enumerate(json_paths, start=1):
        try:
            obj = json.loads(json_path.read_text(encoding="utf-8"))
            ids = extract_ids(obj)
            if not ids:
                bad += 1
                print(f"[INDEX][WARN] sin ids o vacio: {json_path}")
                continue

            rows.append(
                {
                    "path": path_for_index(json_path, project_root),
                    "length": int(len(ids)),
                    "min_id": int(min(ids)),
                    "max_id": int(max(ids)),
                }
            )
        except Exception as exc:
            bad += 1
            print(f"[INDEX][WARN] no se pudo leer {json_path}: {type(exc).__name__}: {exc}")

        if i % 5000 == 0:
            print(f"[INDEX] procesados {i}/{len(json_paths)}")

    df = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    df = df.sort_values("path").reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    print("\n=== INDEX ===")
    print(f"Filas validas: {len(df)}")
    print(f"JSON problematicos: {bad}")
    print(f"CSV guardado en: {out_csv.resolve()}")
    return df
