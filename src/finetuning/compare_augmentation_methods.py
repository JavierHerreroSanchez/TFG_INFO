"""
Comparacion pequena entre la aumentacion manual y la aumentacion de MidiTok.

Genera una muestra reducida en output/augmentation_comparison para inspeccionar
como cambian pitch y duraciones entre ambos metodos sin ejecutar el corpus entero.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import pandas as pd
from miditoolkit import MidiFile

import src.finetuning.augment_finetuning_data as manual
import src.finetuning.augment_finetuning_data_miditok as miditok_aug


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "data" / "finetuning_v2" / "mozart_sonatas_merged"
OUT_DIR = PROJECT_ROOT / "output" / "augmentation_comparison"

N_FILES = 2
K_VARIANTS = 4


def note_stats_miditoolkit(midi: MidiFile) -> Dict[str, float | int | None]:
    """
    Implementa la logica de note stats miditoolkit dentro del pipeline del TFG.

    Parametros principales: midi.
    """

    notes = [n for inst in midi.instruments if not inst.is_drum for n in inst.notes]
    if not notes:
        return {
            "n_notes": 0,
            "pitch_min": None,
            "pitch_max": None,
            "duration_mean_ticks": None,
        }
    return {
        "n_notes": len(notes),
        "pitch_min": min(n.pitch for n in notes),
        "pitch_max": max(n.pitch for n in notes),
        "duration_mean_ticks": sum(n.end - n.start for n in notes) / len(notes),
    }


def run_manual_sample(files: List[Path]) -> pd.DataFrame:
    """
    Implementa la logica de run manual sample dentro del pipeline del TFG.

    Parametros principales: files.
    """

    out_dir = OUT_DIR / "manual_miditoolkit"
    out_dir.mkdir(parents=True, exist_ok=True)

    manual.OUT_AUG_DIR = out_dir
    manual.COMMON_ROOT = SRC_DIR
    manual.PRESERVE_TREE = True

    rng = random.Random(manual.SEED)
    rows = []

    for src in files:
        base = MidiFile(str(src))
        manual.sanitize_miditoolkit(base)
        pairs = manual.sample_pairs(manual.valid_pairs_for_file(base), rng)[:K_VARIANTS]

        for transpose, stretch in pairs:
            aug = manual.rebuild_miditoolkit(base)
            if not manual.transpose_inplace(aug, transpose):
                continue
            manual.time_stretch_inplace(aug, stretch)
            manual.sanitize_miditoolkit(aug)

            dst = manual.make_out_path(src, transpose, stretch)
            dst.parent.mkdir(parents=True, exist_ok=True)
            ok = manual.safe_dump_miditoolkit(
                aug,
                dst,
                debug_tag=f"{src.name} tr={transpose} ts={stretch}",
            )

            rows.append({
                "method": "manual_miditoolkit",
                "source": src.name,
                "path": str(dst),
                "transpose": transpose,
                "stretch": stretch,
                "ok": ok,
                **note_stats_miditoolkit(aug),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "manual_report.csv", index=False)
    return df


def run_miditok_sample(files: List[Path]) -> pd.DataFrame:
    """
    Implementa la logica de run miditok sample dentro del pipeline del TFG.

    Parametros principales: files.
    """

    out_dir = OUT_DIR / "miditok"
    miditok_aug.OUT_AUG_DIR = out_dir
    miditok_aug.COMMON_ROOT = SRC_DIR
    miditok_aug.K_VARIANTS_PER_FILE = K_VARIANTS

    _, report_df = miditok_aug.run_miditok_augmentation(
        in_clean_dir=SRC_DIR,
        out_aug_dir=out_dir,
        out_index_csv=OUT_DIR / "miditok_index.csv",
        out_report_csv=OUT_DIR / "miditok_report.csv",
        max_files=len(files),
    )
    return report_df


def main() -> None:
    """Punto de entrada del script cuando se ejecuta desde consola."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC_DIR.glob("*.mid"))[:N_FILES]
    if not files:
        raise RuntimeError(f"No hay MIDIs en {SRC_DIR}")

    manual_df = run_manual_sample(files)
    miditok_df = run_miditok_sample(files)

    summary = {
        "manual_rows": len(manual_df),
        "miditok_rows": len(miditok_df),
        "manual_pitch_offsets": sorted(manual_df["transpose"].dropna().unique().tolist()),
        "manual_stretch_factors": sorted(manual_df["stretch"].dropna().unique().tolist()),
        "miditok_pitch_offsets": sorted(miditok_df["pitch_offset"].dropna().unique().tolist()),
        "miditok_duration_offsets": sorted(miditok_df["duration_offset"].dropna().unique().tolist()),
        "manual_duration_mean_ticks": float(manual_df["duration_mean_ticks"].mean()),
        "miditok_duration_mean_ticks": float(miditok_df["aug_duration_mean_ticks"].mean()),
    }
    pd.DataFrame([summary]).to_csv(OUT_DIR / "comparison_summary.csv", index=False)

    print("[SUMMARY]")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"[OK] comparison -> {OUT_DIR}")


# Ejecucion directa del script.
if __name__ == "__main__":
    main()
