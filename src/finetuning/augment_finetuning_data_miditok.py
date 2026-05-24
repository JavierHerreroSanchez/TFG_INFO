"""
Aumentacion offline del corpus de fine-tuning usando MidiTok.

Este script es independiente de augment_finetuning_data.py. Usa las funciones
oficiales de MidiTok sobre symusic.Score:
    - pitch_offset: transposicion en semitonos
    - velocity_offset: desplazamiento aditivo de velocity
    - duration_offset: desplazamiento aditivo de duracion

Nota importante: MidiTok no implementa aqui el time-stretch multiplicativo del
script original. Su aumento de duracion suma/resta ticks o beats a cada nota.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from miditok.data_augmentation import augment_score_multiple_offsets
from symusic import Score


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# =============================================================================
# CONFIGURACION PARA EJECUTAR DESDE PYCHARM
# =============================================================================
IN_CLEAN_DIR = PROJECT_ROOT / "data" / "finetuning_v2" / "mozart_sonatas_merged"
OUT_AUG_DIR = PROJECT_ROOT / "data" / "finetuning_v2" / "mozart_sonatas_aug_miditok"
OUT_AUG_INDEX_CSV = PROJECT_ROOT / "output" / "generation_finetuning_tfg_third" / "finetuning_aug_miditok_index.csv"
OUT_AUG_REPORT_CSV = PROJECT_ROOT / "output" / "generation_finetuning_tfg_third" / "finetuning_aug_miditok_report.csv"

PRESERVE_TREE = True
COMMON_ROOT = IN_CLEAN_DIR

# MidiTok offsets. Con all_offset_combinations=True se generan combinaciones
# entre pitch, velocity y duration. Con False se aplican por separado.
PITCH_OFFSETS = [-3, -2, -1, 0, 1, 2, 3]
VELOCITY_OFFSETS: List[int] = []
DURATION_OFFSETS: List[int] = []
ALL_OFFSET_COMBINATIONS = False

K_VARIANTS_PER_FILE = 14
INCLUDE_ORIGINAL = True
SAMPLE_WITHOUT_REPLACEMENT = True
SEED = 1453

RESTRICT_ON_PROGRAM_TESSITURA = False
VELOCITY_RANGE = (1, 127)
DURATION_IN_TICKS = True
MIN_DURATION = 1

DRY_RUN = False
CONTINUE_ON_FAILURE = False
CLEAR_OUT_DIR_ON_START = False


def sanitize_stem(name: str) -> str:

    s = name.replace(",", "")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)


def list_midis_recursively(root: Path) -> List[Path]:

    exts = {".mid", ".midi"}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def fmt_offset(name: str, value: int | float) -> str:

    if isinstance(value, float):
        value_text = f"{value:.3f}".replace(".", "p")
    else:
        value_text = str(value)
    if not str(value_text).startswith("-") and value != 0:
        value_text = f"+{value_text}"
    return f"{name}{value_text}"


def make_out_path(src: Path, pitch: int, velocity: int, duration: int | float) -> Path:

    if PRESERVE_TREE:
        try:
            rel = src.relative_to(COMMON_ROOT)
        except ValueError:
            rel = Path(src.name)
        base = OUT_AUG_DIR / rel
    else:
        base = OUT_AUG_DIR / src.name

    stem = sanitize_stem(base.with_suffix("").name)
    suffix = "__".join(
        [
            fmt_offset("p", pitch),
            fmt_offset("v", velocity),
            fmt_offset("d", duration),
        ]
    )
    return base.parent / f"{stem}__miditok__{suffix}.mid"


def score_note_stats(score: Score) -> Dict[str, float | int | None]:

    pitches: List[int] = []
    velocities: List[int] = []
    durations: List[int] = []
    for track in score.tracks:
        if track.is_drum:
            continue
        for note in track.notes:
            pitches.append(int(note.pitch))
            velocities.append(int(note.velocity))
            durations.append(int(note.duration))

    if not pitches:
        return {
            "n_notes": 0,
            "pitch_min": None,
            "pitch_max": None,
            "velocity_mean": None,
            "duration_mean_ticks": None,
        }

    return {
        "n_notes": len(pitches),
        "pitch_min": min(pitches),
        "pitch_max": max(pitches),
        "velocity_mean": sum(velocities) / len(velocities),
        "duration_mean_ticks": sum(durations) / len(durations),
    }


def score_to_midi(score: Score, out_path: Path) -> None:

    out_path.parent.mkdir(parents=True, exist_ok=True)
    score.dump_midi(out_path)


def available_augmented_scores(score: Score) -> List[Tuple[Tuple[int, int, int | float], Score]]:

    return augment_score_multiple_offsets(
        score,
        pitch_offsets=PITCH_OFFSETS,
        velocity_offsets=VELOCITY_OFFSETS,
        duration_offsets=DURATION_OFFSETS,
        all_offset_combinations=ALL_OFFSET_COMBINATIONS,
        restrict_on_program_tessitura=RESTRICT_ON_PROGRAM_TESSITURA,
        velocity_range=VELOCITY_RANGE,
        duration_in_ticks=DURATION_IN_TICKS,
        min_duration=MIN_DURATION,
    )


def sample_augmented_scores(
    augmented_scores: List[Tuple[Tuple[int, int, int | float], Score]],
    rng: random.Random,
) -> List[Tuple[Tuple[int, int, int | float], Score]]:

    if not augmented_scores:
        return []

    remaining = list(augmented_scores)
    chosen: List[Tuple[Tuple[int, int, int | float], Score]] = []

    if INCLUDE_ORIGINAL:
        for item in list(remaining):
            if item[0] == (0, 0, 0):
                chosen.append(item)
                remaining.remove(item)
                break

    remaining_k = max(0, K_VARIANTS_PER_FILE - len(chosen))
    if remaining_k == 0:
        return chosen

    if SAMPLE_WITHOUT_REPLACEMENT:
        if remaining_k >= len(remaining):
            chosen.extend(remaining)
        else:
            chosen.extend(rng.sample(remaining, k=remaining_k))
    else:
        for _ in range(remaining_k):
            chosen.append(rng.choice(remaining))

    return chosen


def run_miditok_augmentation(
    in_clean_dir: Path = IN_CLEAN_DIR,
    out_aug_dir: Path = OUT_AUG_DIR,
    out_index_csv: Path = OUT_AUG_INDEX_CSV,
    out_report_csv: Path = OUT_AUG_REPORT_CSV,
    max_files: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    rng = random.Random(SEED)

    if not in_clean_dir.exists():
        raise FileNotFoundError(f"No existe IN_CLEAN_DIR: {in_clean_dir}")

    if CLEAR_OUT_DIR_ON_START and out_aug_dir.exists():
        import shutil
        shutil.rmtree(out_aug_dir)

    out_aug_dir.mkdir(parents=True, exist_ok=True)
    out_index_csv.parent.mkdir(parents=True, exist_ok=True)
    out_report_csv.parent.mkdir(parents=True, exist_ok=True)

    src_midis = list_midis_recursively(in_clean_dir)
    if max_files is not None:
        src_midis = src_midis[:max_files]

    print(f"[INFO] in_clean={in_clean_dir} files={len(src_midis)}")
    print(f"[INFO] out_aug={out_aug_dir}")
    print(
        "[INFO] offsets "
        f"pitch={PITCH_OFFSETS} velocity={VELOCITY_OFFSETS} duration={DURATION_OFFSETS} "
        f"all_combos={ALL_OFFSET_COMBINATIONS}"
    )

    report_rows: List[Dict] = []
    kept_rows: List[Dict] = []

    for i, src in enumerate(src_midis, start=1):
        try:
            score = Score(src)
        except Exception as exc:
            report_rows.append({
                "path_original": str(src),
                "status": "READ_FAIL",
                "pitch_offset": None,
                "velocity_offset": None,
                "duration_offset": None,
                "path_aug": "",
                "detail": type(exc).__name__,
            })
            if not CONTINUE_ON_FAILURE:
                raise
            continue

        original_stats = score_note_stats(score)
        augmented_scores = available_augmented_scores(score)
        chosen = sample_augmented_scores(augmented_scores, rng)

        if not chosen:
            report_rows.append({
                "path_original": str(src),
                "status": "SKIP_NO_VALID_OFFSET",
                "pitch_offset": None,
                "velocity_offset": None,
                "duration_offset": None,
                "path_aug": "",
                "detail": "",
                **{f"original_{k}": v for k, v in original_stats.items()},
            })
            continue

        for (pitch, velocity, duration), score_aug in chosen:
            dst = make_out_path(src, pitch, velocity, duration)
            if out_aug_dir != OUT_AUG_DIR:
                dst = out_aug_dir / dst.relative_to(OUT_AUG_DIR)

            aug_stats = score_note_stats(score_aug)
            try:
                if not DRY_RUN:
                    score_to_midi(score_aug, dst)
                status = "OK"
                detail = ""
            except Exception as exc:
                status = "DUMP_FAIL"
                detail = type(exc).__name__
                if not CONTINUE_ON_FAILURE:
                    raise

            row = {
                "path_original": str(src),
                "status": status,
                "pitch_offset": pitch,
                "velocity_offset": velocity,
                "duration_offset": duration,
                "path_aug": str(dst) if status == "OK" else "",
                "detail": detail,
                **{f"original_{k}": v for k, v in original_stats.items()},
                **{f"aug_{k}": v for k, v in aug_stats.items()},
            }
            report_rows.append(row)

            if status == "OK":
                kept_rows.append({
                    "path": str(dst),
                    "path_original": str(src),
                    "pitch_offset": pitch,
                    "velocity_offset": velocity,
                    "duration_offset": duration,
                })

        if i % 200 == 0:
            print(f"[PROC] {i}/{len(src_midis)}")

    report_df = pd.DataFrame(report_rows)
    kept_df = pd.DataFrame(kept_rows)
    if kept_df.empty:
        kept_df = pd.DataFrame(columns=["path", "path_original", "pitch_offset", "velocity_offset", "duration_offset"])

    report_df.to_csv(out_report_csv, index=False)
    kept_df.to_csv(out_index_csv, index=False)

    metadata = {
        "method": "miditok.data_augmentation.augment_score_multiple_offsets",
        "pitch_offsets": PITCH_OFFSETS,
        "velocity_offsets": VELOCITY_OFFSETS,
        "duration_offsets": DURATION_OFFSETS,
        "all_offset_combinations": ALL_OFFSET_COMBINATIONS,
        "restrict_on_program_tessitura": RESTRICT_ON_PROGRAM_TESSITURA,
        "velocity_range": VELOCITY_RANGE,
        "duration_in_ticks": DURATION_IN_TICKS,
        "min_duration": MIN_DURATION,
        "k_variants_per_file": K_VARIANTS_PER_FILE,
        "include_original": INCLUDE_ORIGINAL,
        "seed": SEED,
    }
    (out_report_csv.parent / "finetuning_aug_miditok_config.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] report -> {out_report_csv} rows={len(report_df)}")
    print(f"[OK] index  -> {out_index_csv} rows={len(kept_df)}")
    print(f"[OK] aug dir-> {out_aug_dir}")
    return kept_df, report_df


def main() -> None:
    """Punto de entrada del script cuando se ejecuta desde consola."""

    run_miditok_augmentation()


# Ejecucion directa del script.
if __name__ == "__main__":
    main()
