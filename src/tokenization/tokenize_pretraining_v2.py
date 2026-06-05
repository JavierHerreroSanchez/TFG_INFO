"""
Tokeniza ARIAMidi y MAESTRO añadiendo attribute controls.

Objetivo:
1) Tokenizar MAESTRO correctamente con el tokenizer BPE ya entrenado.
2) Insertar Attribute Controls (AC) de forma explícita y robusta.
3) Guardar JSONs y un resumen claro de OK / SKIP / BAD.
4) Mantener el script listo para reutilizarlo con otros datasets cambiando solo DATASET_ROOT.

Notas importantes:
- Se usa tokenizer.encode(..., attribute_controls_indexes=...) en vez de tokenize_dataset(),
  porque aquí se controlan explícitamente los AC.
- Primero preprocesa el Score con tokenizer.preprocess_score(score), y luego tokeniza con
  no_preprocess_score=True. MidiTok recomienda esto cuando se usan attribute_controls_indexes.
- El conteo de barras para los AC bar-level se obtiene de una tokenización preliminar SIN AC,
  contando los tokens Bar_None. Así no se depende de utilidades internas.
"""

from __future__ import annotations

import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from miditok import REMI
from symusic import Score

from src.tokenization.indexing import build_token_index

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TOKENIZER_PATH = (PROJECT_ROOT / "tokenizer" / "tokenizer_REMI_BPE_v2.json").resolve()

# Rutas internas del proyecto.
DATASET_ROOT = (PROJECT_ROOT / "data" / "pretraining_raw").resolve()
OUT_DIR = (PROJECT_ROOT / "data" / "interim" / "tokenized_json_bpe_v2").resolve()
INDEX_CSV = (PROJECT_ROOT / "data" / "interim" / "indexes" / "index_pretraining_v2.csv").resolve()
BAD_LIST_PATH = (PROJECT_ROOT / "tokenizer" / "bad_midis.txt").resolve()

NUM_WORKERS = max(1, (os.cpu_count() or 8) - 2)
OVERWRITE = True
SAVE_TOKEN_STRINGS_PREVIEW = False  # True para guardar unas pocas tokens legibles por JSON
PREVIEW_TOKEN_COUNT = 128

MIDI_EXTS = {".mid", ".midi"}

# Caché por proceso.
_TOKENIZER: REMI | None = None


# =============================================================================
# UTILIDADES
# =============================================================================

def get_tokenizer() -> REMI:

    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = REMI(params=TOKENIZER_PATH)
    return _TOKENIZER


def list_midi_files(root: Path) -> list[Path]:

    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in MIDI_EXTS
    )


def is_bar_level_ac(ac_obj: Any) -> bool:
    """
    MidiTok documenta AC bar-level y track-level, pero aquí no se depende de internals frágiles.
    Heurística robusta:
    - si el nombre de clase contiene 'Bar' => bar-level
    - si contiene 'Track' => track-level
    - si expone tokens cuyo tipo empiece por ACBar => bar-level
    - si expone tokens cuyo tipo empiece por ACTrack => track-level
    """
    cls_name = ac_obj.__class__.__name__.lower()
    if "bar" in cls_name:
        return True
    if "track" in cls_name:
        return False

    tokens_attr = getattr(ac_obj, "tokens", None)
    if tokens_attr is not None:
        for tok in tokens_attr:
            # MidiTok suele guardar tuplas (type, value) o strings
            if isinstance(tok, (tuple, list)) and tok:
                tok_type = str(tok[0])
            else:
                tok_type = str(tok)
            if tok_type.startswith("ACBar"):
                return True
            if tok_type.startswith("ACTrack"):
                return False

    # Fallback conservador: si no lo sabemos, lo tratamos como track-level
    return False


def normalize_tokseq_list(tokseq_or_list: Any) -> list[Any]:

    if isinstance(tokseq_or_list, list):
        return tokseq_or_list
    return [tokseq_or_list]


def count_bars_per_track_from_pretokenization(tokenizer: REMI, score_preprocessed: Score) -> list[int]:
    """
    Hace una tokenización preliminar SIN AC y SIN BPE para contar cuántos Bar_None
    tiene cada pista. Ese valor da la cantidad de barras sobre las que insertar AC bar-level.
    """
    seqs = tokenizer.encode(
        score_preprocessed,
        encode_ids=False,
        no_preprocess_score=True,
        attribute_controls_indexes=None,
    )
    seqs = normalize_tokseq_list(seqs)

    bar_counts: list[int] = []
    for seq in seqs:
        tokenizer.complete_sequence(seq, complete_bytes=False)
        tokens = seq.tokens if seq.tokens is not None else []
        n_bars = sum(1 for t in tokens if t == "Bar_None")
        # Si por alguna razon no aparece ningun Bar_None, se conserva al menos 1.
        bar_counts.append(max(1, n_bars))
    return bar_counts


def build_attribute_controls_indexes(tokenizer: REMI, score_preprocessed: Score) -> dict[int, dict[int, Any]]:
    """
    Construye la estructura que espera MidiTok:
    {track_idx: {ac_idx: True (track-level) | [bar_idx, ...] (bar-level)}}
    """
    attribute_controls = getattr(tokenizer, "attribute_controls", None)
    if not attribute_controls:
        return {}

    bar_counts = count_bars_per_track_from_pretokenization(tokenizer, score_preprocessed)
    n_tracks = len(score_preprocessed.tracks)

    ac_map: dict[int, dict[int, Any]] = {}

    for track_idx in range(n_tracks):
        per_track: dict[int, Any] = {}
        n_bars = bar_counts[track_idx] if track_idx < len(bar_counts) else 1

        for ac_idx, ac_obj in enumerate(attribute_controls):
            if is_bar_level_ac(ac_obj):
                per_track[ac_idx] = list(range(n_bars))
            else:
                per_track[ac_idx] = True

        ac_map[track_idx] = per_track

    return ac_map


def save_bad_line(midi_path: Path, reason: str) -> None:
    """
    Guarda resultados intermedios o finales en disco.

    """

    BAD_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BAD_LIST_PATH.open("a", encoding="utf-8") as f:
        f.write(f"{midi_path}\t{reason}\n")


def output_json_path(midi_path: Path, dataset_root: Path, out_dir: Path) -> Path:

    rel = midi_path.relative_to(dataset_root)
    return (out_dir / rel).with_suffix(".json")


def token_family_counts_from_tokens(tokens: list[str]) -> dict[str, int]:

    counts: dict[str, int] = {}
    for tok in tokens:
        family = tok.split("_", 1)[0] if "_" in tok else tok
        counts[family] = counts.get(family, 0) + 1
    return counts


# =============================================================================
# PROCESADO DE UN FICHERO
# =============================================================================

@dataclass
class FileResult:
    """Representa FileResult dentro del flujo experimental del TFG."""

    status: str  # ok / skip / bad
    midi_path: str
    json_path: str | None = None
    reason: str | None = None


def process_one_file(midi_path_str: str) -> FileResult:

    midi_path = Path(midi_path_str)
    tokenizer = get_tokenizer()

    try:
        out_path = output_json_path(midi_path, DATASET_ROOT, OUT_DIR)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not OVERWRITE:
            return FileResult(status="skip", midi_path=str(midi_path), json_path=str(out_path), reason="exists")

        # 1) Cargar Score
        score = Score(midi_path)

        # 2) Preprocesar explícitamente
        score_pre = tokenizer.preprocess_score(score)

        if len(score_pre.tracks) == 0:
            save_bad_line(midi_path, "no_tracks_after_preprocess")
            return FileResult(status="bad", midi_path=str(midi_path), reason="no_tracks_after_preprocess")

        # 3) Construir AC map tras el preprocess
        ac_indexes = build_attribute_controls_indexes(tokenizer, score_pre)

        # 4) Tokenizar con AC
        seqs = tokenizer.encode(
            score_pre,
            encode_ids=True,
            no_preprocess_score=True,
            attribute_controls_indexes=ac_indexes if ac_indexes else None,
        )
        seqs_list = normalize_tokseq_list(seqs)

        # 5) Decodificar ids->tokens para inspección/metadata
        total_ids = 0
        total_tokens = 0
        per_track_bars: list[int] = []
        per_track_families: list[dict[str, int]] = []
        preview_tokens: list[list[str]] = []

        for seq in seqs_list:
            tokenizer.decode_token_ids(seq)
            tokenizer.complete_sequence(seq, complete_bytes=False)

            ids = seq.ids if seq.ids is not None else []
            tokens = seq.tokens if seq.tokens is not None else []

            if not ids:
                save_bad_line(midi_path, "empty_ids")
                return FileResult(status="bad", midi_path=str(midi_path), reason="empty_ids")

            total_ids += len(ids)
            total_tokens += len(tokens)
            per_track_bars.append(sum(1 for t in tokens if t == "Bar_None"))
            per_track_families.append(token_family_counts_from_tokens(tokens))

            if SAVE_TOKEN_STRINGS_PREVIEW:
                preview_tokens.append(tokens[:PREVIEW_TOKEN_COUNT])

        # 6) Guardar con metadata útil para comprobar AC / acordes / barras
        meta = {
            "source_midi": str(midi_path),
            "tokenizer_path": str(TOKENIZER_PATH),
            "n_tracks_after_preprocess": len(score_pre.tracks),
            "attribute_controls_inserted": bool(ac_indexes),
            "attribute_controls_indexes": ac_indexes,
            "bars_per_track_from_pretokenization": per_track_bars,
            "total_ids": total_ids,
            "total_tokens_after_bpe_decode": total_tokens,
            "per_track_token_families": per_track_families,
        }

        if len(seqs_list) == 1:
            tokenizer.save_tokens(seqs_list[0], out_path, **meta)
        else:
            # MidiTok save_tokens suele esperar una secuencia; para multitrack en REMI suele haber lista.
            # Guardado manual de una estructura simple si hay varias pistas.
            payload = {
                "ids": [seq.ids for seq in seqs_list],
                **meta,
            }
            if SAVE_TOKEN_STRINGS_PREVIEW:
                payload["tokens_preview"] = preview_tokens
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)

        return FileResult(status="ok", midi_path=str(midi_path), json_path=str(out_path))

    except Exception as exc:
        save_bad_line(midi_path, f"{type(exc).__name__}: {exc}")
        return FileResult(
            status="bad",
            midi_path=str(midi_path),
            reason=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Punto de entrada del script cuando se ejecuta desde consola."""

    print(f"[INFO] PROJECT_ROOT  = {PROJECT_ROOT}")
    print(f"[INFO] TOKENIZER     = {TOKENIZER_PATH}")
    print(f"[INFO] DATASET_ROOT  = {DATASET_ROOT}")
    print(f"[INFO] OUT_DIR       = {OUT_DIR}")
    print(f"[INFO] BAD_LIST      = {BAD_LIST_PATH}")
    print(f"[INFO] WORKERS       = {NUM_WORKERS}")

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"No existe el tokenizer: {TOKENIZER_PATH}")
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"No existe el dataset root: {DATASET_ROOT}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    midi_files = list_midi_files(DATASET_ROOT)
    print(f"[INFO] MIDIs totales = {len(midi_files)}")

    ok = 0
    skip = 0
    bad = 0

    # Limpieza opcional del bad list
    if BAD_LIST_PATH.exists():
        BAD_LIST_PATH.unlink()

    futures = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        for midi_path in midi_files:
            futures.append(ex.submit(process_one_file, str(midi_path)))

        for i, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            if res.status == "ok":
                ok += 1
            elif res.status == "skip":
                skip += 1
            else:
                bad += 1

            if i % 100 == 0 or i == len(futures):
                print(f"[INFO] {i}/{len(futures)} | ok={ok} skip={skip} bad={bad}")

    print("\n=== RESUMEN ===")
    print(f"OK:   {ok}")
    print(f"SKIP: {skip}")
    print(f"BAD:  {bad}")
    build_token_index(OUT_DIR, INDEX_CSV, PROJECT_ROOT)


# Ejecución directa del script.
if __name__ == "__main__":
    main()
