"""
Evalua las piezas generadas mediante metricas simbolicas, espectrales o graficas.

Los resultados producidos aqui sirven para justificar experimentalmente la calidad del modelo en la memoria del TFG.
"""

from __future__ import annotations

"""
Genera los 7 graficos de evaluacion simbolica/espectral del TFG.

Preparado para PyCharm:
1) Ajusta SYMBOLIC_DATA_DIR y SPECTRAL_DATA_DIR, o pasalos por consola.
2) Ejecuta el script directamente desde PyCharm.
3) Los PNG se guardan en OUTPUT_DIR.

Ejemplo por consola:
python src/evaluation/generar_graficos_evaluacion.py ^
  --symbolic-dir "output/generation_finetuning_tfg_second/midi_eval_windows" ^
  --spectral-dir "output/generation_finetuning_tfg_second/midi_spectral_eval_windows" ^
  --output-dir "output/graficos_finetuning_v2"

Ficheros simbolicos esperados:
- per_piece_details.json
- per_piece_evaluation.csv
- reference_features.csv
- summary_compact.csv

Ficheros espectrales esperados:
- spectral_features_all.csv
- spectral_per_piece_details.json
- spectral_summary.json
- spectral_evaluation.csv
"""

import json
import math
import re
import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import entropy as scipy_entropy
except Exception:
    scipy_entropy = None


# ============================================================
# CONFIGURACION PYCHARM
# ============================================================

# Opcion 1: si este script esta en la misma carpeta que los CSV/JSON, deja esto asi.
DATA_DIR = Path(__file__).resolve().parent

# Opcion 2: si los datos estan en carpetas separadas, ajusta estas dos rutas.
# Tambien puedes pasarlas por consola con --symbolic-dir y --spectral-dir.
SYMBOLIC_DATA_DIR = Path(r"../../output/generation_finetuning_tfg_second/midi_eval_windows")
SPECTRAL_DATA_DIR = Path(r"../../output/generation_finetuning_tfg_second/midi_spectral_eval_windows")

# Si es None, se usa SYMBOLIC_DATA_DIR / "graficos_evaluacion".
OUTPUT_DIR: Path | None = None
PREFIX = "finetuning"
OUTPUT_SUFFIX = "_v2"
MAX_PIECES = 10
DPI = 150

# Si tienes un global_report_windows.csv junto a estos ficheros, el script lo usara
# para OA/KLD simbolico. Si no existe, calcula OA/KLD desde reference_features.csv
# y per_piece_details.json.
PREFER_OPTIONAL_SYMBOLIC_GLOBAL_REPORT = True
OPTIONAL_SYMBOLIC_GLOBAL_REPORT_NAME = "global_report_windows.csv"


# ============================================================
# ESTILO VISUAL
# ============================================================

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 24,
    "axes.titleweight": "bold",
    "axes.labelsize": 18,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})

SYMBOLIC_ORDER = [
    "pitch_range",
    "n_pitch_classes_used",
    "n_pitches_used",
    "pitch_class_entropy",
    "scale_consistency",
    "polyphony",
    "polyphony_rate",
    "empty_beat_rate",
    "empty_measure_rate",
    "groove_consistency",
    "consecutive_pitch_repetition_ratio_custom",
    "mean_velocity_custom",
    "std_velocity_custom",
    "mean_duration_beats_custom",
    "std_duration_beats_custom",
]

SYMBOLIC_LABELS = {
    "pitch_range": "Rango de pitch",
    "n_pitch_classes_used": "Clases de pitch usadas",
    "n_pitches_used": "Pitches usados",
    "pitch_class_entropy": "Entropía clases pitch",
    "scale_consistency": "Consistencia tonal",
    "polyphony": "Polifonía",
    "polyphony_rate": "Tasa de polifonía",
    "empty_beat_rate": "Tasa de beats vacíos",
    "empty_measure_rate": "Tasa compases vacíos",
    "groove_consistency": "consistencia groove",
    "consecutive_pitch_repetition_ratio_custom": "Repetición consecutiva pitch",
    "mean_velocity_custom": "Velocidad media",
    "std_velocity_custom": "Desv. velocidad",
    "mean_duration_beats_custom": "Duración media nota",
    "std_duration_beats_custom": "Desv. duración nota",
}

SPECTRAL_ORDER = [
    "rms_mean",
    "rms_std",
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    "spectral_rolloff_mean",
    "spectral_rolloff_std",
    "spectral_flatness_mean",
    "spectral_flatness_std",
    "spectral_contrast_mean",
    "spectral_contrast_std",
]

SPECTRAL_LABELS = {
    "rms_mean": "RMS medio",
    "rms_std": "Desv. RMS",
    "spectral_centroid_mean": "Centroide medio",
    "spectral_centroid_std": "Desv. centroide",
    "spectral_bandwidth_mean": "Ancho banda medio",
    "spectral_bandwidth_std": "Desv. ancho banda",
    "spectral_rolloff_mean": "Rolloff medio",
    "spectral_rolloff_std": "Desv. rolloff",
    "spectral_flatness_mean": "Planitud media",
    "spectral_flatness_std": "Desv. planitud",
    "spectral_contrast_mean": "Contraste medio",
    "spectral_contrast_std": "Desv. contraste",
}

SYMBOLIC_REQUIRED_FILES = [
    "per_piece_details.json",
    "per_piece_evaluation.csv",
    "reference_features.csv",
    "summary_compact.csv",
]

SPECTRAL_REQUIRED_FILES = [
    "spectral_features_all.csv",
    "spectral_per_piece_details.json",
    "spectral_summary.json",
    "spectral_evaluation.csv",
]


# ============================================================
# UTILIDADES
# ============================================================

def check_required_files(data_dir: Path, required_files: List[str], label: str) -> None:
    """
    Implementa la logica de check required files dentro del pipeline del TFG.

    Parametros principales: data_dir, required_files, label.
    """

    missing = [name for name in required_files if not (data_dir / name).exists()]
    if missing:
        msg = "\n".join(f"- {m}" for m in missing)
        raise FileNotFoundError(
            f"Faltan ficheros en {label}={data_dir}:\n{msg}\n\n"
            "Ajusta --symbolic-dir/--spectral-dir o las constantes de CONFIGURACION PYCHARM."
        )


def resolve_dir(value: str | Path | None, default: Path) -> Path:
    """Normaliza rutas de entrada para usarlas desde PyCharm o consola."""

    path = Path(value) if value is not None else default
    return path.expanduser().resolve()


def parse_args() -> argparse.Namespace:
    """Lee la configuracion opcional de consola."""

    parser = argparse.ArgumentParser(
        description="Genera los 7 graficos de evaluacion simbolica/espectral del TFG."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Carpeta compartida para simbolico y espectral. Mantiene compatibilidad con el modo antiguo.",
    )
    parser.add_argument(
        "--symbolic-dir",
        type=Path,
        default=None,
        help="Carpeta con per_piece_details.json, per_piece_evaluation.csv, reference_features.csv y summary_compact.csv.",
    )
    parser.add_argument(
        "--spectral-dir",
        type=Path,
        default=None,
        help="Carpeta con spectral_features_all.csv, spectral_per_piece_details.json, spectral_summary.json y spectral_evaluation.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Carpeta donde se guardan los PNG. Por defecto: symbolic-dir/graficos_evaluacion.",
    )
    parser.add_argument("--prefix", default=PREFIX, help="Prefijo de los PNG generados.")
    parser.add_argument("--suffix", default=OUTPUT_SUFFIX, help="Sufijo de los PNG generados, por ejemplo _v2.")
    parser.add_argument("--max-pieces", type=int, default=MAX_PIECES, help="Numero maximo de piezas a graficar.")
    parser.add_argument("--dpi", type=int, default=DPI, help="Resolucion de salida de los PNG.")
    parser.add_argument(
        "--no-symbolic-global-report",
        action="store_false",
        dest="prefer_symbolic_global_report",
        help="Ignora global_report_windows.csv aunque exista y recalcula OA/KLD simbolico.",
    )
    parser.set_defaults(prefer_symbolic_global_report=PREFER_OPTIONAL_SYMBOLIC_GLOBAL_REPORT)
    return parser.parse_args()


def read_json(path: Path):
    """
    Lee datos de entrada y los normaliza para el procesamiento posterior.

    Parametros principales: path.
    """

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_piece_id(path_or_name: str) -> int | None:
    """
    Implementa la logica de extract piece id dentro del pipeline del TFG.

    Parametros principales: path_or_name.
    """

    s = str(path_or_name).replace("\\", "/")
    patterns = [
        r"generated_from_json(\d+)",
        r"\bg(\d+)\b",
        r"json(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            return int(m.group(1))
    nums = re.findall(r"(\d+)", Path(s).stem)
    return int(nums[-1]) if nums else None


def add_piece_id(df: pd.DataFrame, file_col: str = "file") -> pd.DataFrame:
    """
    Implementa la logica de add piece id dentro del pipeline del TFG.

    Parametros principales: df, file_col.
    """

    out = df.copy()
    out["piece_id"] = out[file_col].apply(extract_piece_id)
    out = out[out["piece_id"].notna()].copy()
    out["piece_id"] = out["piece_id"].astype(int)
    return out.sort_values("piece_id").reset_index(drop=True)


def finite_values(arr: Iterable[float]) -> np.ndarray:
    """
    Implementa la logica de finite values dentro del pipeline del TFG.

    Parametros principales: arr.
    """

    a = np.asarray(list(arr), dtype=float)
    return a[np.isfinite(a)]


def safe_float(x, default=np.nan) -> float:
    """
    Implementa la logica de safe float dentro del pipeline del TFG.

    Parametros principales: x, default.
    """

    try:
        v = float(x)
        return v if math.isfinite(v) else float(default)
    except Exception:
        return float(default)


def finite_stats(arr: Iterable[float]) -> Tuple[float, float, int]:
    """
    Implementa la logica de finite stats dentro del pipeline del TFG.

    Parametros principales: arr.
    """

    a = finite_values(arr)
    if a.size == 0:
        return np.nan, np.nan, 0
    return float(a.mean()), float(a.std()) if a.size > 1 else 0.0, int(a.size)


def entropy_real_to_gen(p_real: np.ndarray, q_gen: np.ndarray) -> float:
    """
    Implementa la logica de entropy real to gen dentro del pipeline del TFG.

    Parametros principales: p_real, q_gen.
    """

    if p_real.size == 0 or q_gen.size == 0:
        return np.nan
    if scipy_entropy is not None:
        return float(scipy_entropy(p_real, q_gen))
    p = np.asarray(p_real, dtype=float)
    q = np.asarray(q_gen, dtype=float)
    return float(np.sum(p * np.log(p / q)))


def normalized_hist_pair(a: Iterable[float], b: Iterable[float], bins: int | str = "auto", max_auto_bins: int = 48):
    """
    Implementa la logica de normalized hist pair dentro del pipeline del TFG.

    Parametros principales: a, b, bins, max_auto_bins.
    """

    a = finite_values(a)
    b = finite_values(b)
    if a.size == 0 or b.size == 0:
        return np.asarray([]), np.asarray([])

    lo = min(float(np.min(a)), float(np.min(b)))
    hi = max(float(np.max(a)), float(np.max(b)))
    if lo == hi:
        lo -= 0.5
        hi += 0.5

    if bins == "auto":
        nb = int(np.clip(np.sqrt(a.size + b.size), 8, max_auto_bins))
    else:
        nb = int(bins)

    edges = np.linspace(lo, hi, nb + 1)
    ha, _ = np.histogram(a, bins=edges, density=False)
    hb, _ = np.histogram(b, bins=edges, density=False)

    ha = ha.astype(float) + 1e-12
    hb = hb.astype(float) + 1e-12
    ha /= ha.sum()
    hb /= hb.sum()
    return ha, hb


def global_distribution_report(ref_df: pd.DataFrame, gen_df: pd.DataFrame, features: List[str], max_auto_bins: int) -> pd.DataFrame:
    """
    Implementa la logica de global distribution report dentro del pipeline del TFG.

    Parametros principales: ref_df, gen_df, features, max_auto_bins.
    """

    rows = []
    for feat in features:
        if feat not in ref_df.columns or feat not in gen_df.columns:
            print(f"[WARN] feature omitida porque no existe en ambos dataframes: {feat}")
            continue
        ref_vals = pd.to_numeric(ref_df[feat], errors="coerce").to_numpy(dtype=float)
        gen_vals = pd.to_numeric(gen_df[feat], errors="coerce").to_numpy(dtype=float)
        p, q = normalized_hist_pair(ref_vals, gen_vals, max_auto_bins=max_auto_bins)
        real_mean, real_std, n_real = finite_stats(ref_vals)
        gen_mean, gen_std, n_gen = finite_stats(gen_vals)
        rows.append({
            "feature": feat,
            "oa": float(np.minimum(p, q).sum()) if p.size and q.size else np.nan,
            "kld_real_to_gen": entropy_real_to_gen(p, q),
            "real_mean": safe_float(real_mean),
            "gen_mean": safe_float(gen_mean),
            "real_std": safe_float(real_std),
            "gen_std": safe_float(gen_std),
            "n_real": n_real,
            "n_gen": n_gen,
        })
    return pd.DataFrame(rows)


def parse_json_metric_payload(value: str) -> dict | None:
    """
    Implementa la logica de parse json metric payload dentro del pipeline del TFG.

    Parametros principales: value.
    """

    try:
        payload = json.loads(value)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def parse_summary_compact(summary_compact: pd.DataFrame, section_prefix: str = "global_report") -> pd.DataFrame:
    """
    Implementa la logica de parse summary compact dentro del pipeline del TFG.

    Parametros principales: summary_compact, section_prefix.
    """

    rows = []
    for _, row in summary_compact.iterrows():
        section = str(row.get("section", ""))
        if not section.startswith(section_prefix):
            continue
        feature = str(row.get("key", ""))
        payload = parse_json_metric_payload(str(row.get("value", "")))
        if not payload:
            continue
        payload = payload.copy()
        payload["feature"] = feature
        rows.append(payload)
    return pd.DataFrame(rows)


def symbolic_generated_features_from_details(per_piece_details: list, per_piece_eval: pd.DataFrame) -> pd.DataFrame:
    """
    Implementa la logica de symbolic generated features from details dentro del pipeline del TFG.

    Parametros principales: per_piece_details, per_piece_eval.
    """

    rows = []
    for item in per_piece_details:
        d: Dict[str, float | str] = {"file": item["file"]}
        details = item.get("reference_based", {}).get("details", {})
        for feat, payload in details.items():
            if isinstance(payload, dict) and "x" in payload:
                d[feat] = payload["x"]
        rows.append(d)

    gen_df = add_piece_id(pd.DataFrame(rows))

    # Algunas columnas auxiliares, si hacen falta para diagnostico.
    aux_cols = ["duration_beats_custom", "n_notes_custom"]
    eval_aux = add_piece_id(per_piece_eval)
    cols = ["piece_id"] + [c for c in aux_cols if c in eval_aux.columns]
    if len(cols) > 1:
        gen_df = gen_df.merge(eval_aux[cols], on="piece_id", how="left")
    return gen_df


def spectral_global_report_from_evaluation(spectral_eval: pd.DataFrame) -> pd.DataFrame:
    """
    Implementa la logica de spectral global report from evaluation dentro del pipeline del TFG.

    Parametros principales: spectral_eval.
    """

    rows = []
    global_rows = spectral_eval[spectral_eval["section"].eq("global_metric")]
    for _, row in global_rows.iterrows():
        payload = parse_json_metric_payload(str(row.get("strengths", "")))
        if not payload:
            continue
        payload = payload.copy()
        payload["feature"] = row["file"]
        rows.append(payload)
    return pd.DataFrame(rows)


def spectral_global_report_from_summary(spectral_summary: dict) -> pd.DataFrame:
    """
    Implementa la logica de spectral global report from summary dentro del pipeline del TFG.

    Parametros principales: spectral_summary.
    """

    rows = []
    for payload in spectral_summary.get("top_global_matches", []):
        if isinstance(payload, dict) and "feature" in payload:
            rows.append(payload.copy())
    return pd.DataFrame(rows)


def spectral_global_report_from_features(spectral_features_all: pd.DataFrame) -> pd.DataFrame:
    """
    Implementa la logica de spectral global report from features dentro del pipeline del TFG.

    Parametros principales: spectral_features_all.
    """

    gen_df = spectral_features_all[spectral_features_all["set_type"].eq("generated")].copy()
    ref_df = spectral_features_all[spectral_features_all["set_type"].eq("reference")].copy()
    return global_distribution_report(ref_df, gen_df, SPECTRAL_ORDER + ["duration_s"], max_auto_bins=40)


def ensure_ordered_report(report: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    """
    Implementa la logica de ensure ordered report dentro del pipeline del TFG.

    Parametros principales: report, order.
    """

    if report.empty:
        raise ValueError("El reporte global esta vacio.")
    out = report.copy()
    out = out[out["feature"].isin(order)].copy()
    missing = [f for f in order if f not in set(out["feature"])]
    if missing:
        raise ValueError(
            "Faltan metricas en el reporte global:\n" +
            "\n".join(f"- {m}" for m in missing)
        )
    out["feature"] = pd.Categorical(out["feature"], categories=order, ordered=True)
    return out.sort_values("feature").reset_index(drop=True)


# ============================================================
# ESTILO DE GRAFICOS
# ============================================================

def add_plausibility_bands(ax) -> None:
    """
    Implementa la logica de add plausibility bands dentro del pipeline del TFG.

    Parametros principales: ax.
    """

    ax.axhspan(0, 45, color="#efefef", alpha=0.65, zorder=0)
    ax.axhspan(45, 58, color="#ead6d6", alpha=0.60, zorder=0)
    ax.axhspan(58, 72, color="#e8dfbf", alpha=0.60, zorder=0)
    ax.axhspan(72, 85, color="#dce6d7", alpha=0.60, zorder=0)
    ax.axhspan(85, 100, color="#d9e3ef", alpha=0.60, zorder=0)

    for y, label in [
        (45, "45 debil"),
        (58, "58 aceptable"),
        (72, "72 plausible"),
        (85, "85 muy plausible"),
    ]:
        ax.axhline(y, color="gray", linestyle="--", linewidth=1.1, alpha=0.9)
        ax.text(
            0.965,
            y + 0.7,
            label,
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=11,
            color="dimgray",
        )


def annotate_vertical_bars(ax, bars) -> None:
    """
    Implementa la logica de annotate vertical bars dentro del pipeline del TFG.

    Parametros principales: ax, bars.
    """

    for bar in bars:
        h = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.8,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=9.5,
        )


def annotate_horizontal_bars(ax, bars, fmt: str, offset: float) -> None:
    """
    Implementa la logica de annotate horizontal bars dentro del pipeline del TFG.

    Parametros principales: ax, bars, fmt, offset.
    """

    for bar in bars:
        value = float(bar.get_width())
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(value),
            va="center",
            ha="left",
            fontsize=10,
        )


def style_hbar_axis(ax) -> None:
    """
    Implementa la logica de style hbar axis dentro del pipeline del TFG.

    Parametros principales: ax.
    """

    ax.grid(axis="x", linestyle="--", alpha=0.30)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)


# ============================================================
# PLOTS
# ============================================================

def plot_score_comparison(piece_df: pd.DataFrame, out_path: Path) -> None:
    """
    Dibuja una visualizacion usada durante la evaluacion.

    Parametros principales: piece_df, out_path.
    """

    x = np.arange(len(piece_df))
    labels = [f"g{i}" for i in piece_df["piece_id"]]
    width = 0.36

    fig, ax = plt.subplots(figsize=(16, 8))
    add_plausibility_bands(ax)

    b1 = ax.bar(x - width / 2, piece_df["symbolic_global_score"], width, color=BLUE, label="Score global simbólico", alpha=0.95)
    b2 = ax.bar(x + width / 2, piece_df["spectral_global_score"], width, color=ORANGE, label="Score global espectral", alpha=0.95)

    ax.set_title("Comparación por pieza: score simbólico vs score espectral")
    ax.set_ylabel("Puntuación (0-100)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.18)
    annotate_vertical_bars(ax, b1)
    annotate_vertical_bars(ax, b2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_componentes(piece_df: pd.DataFrame, out_path: Path, kind: str) -> None:
    """
    Dibuja una visualizacion usada durante la evaluacion.

    Parametros principales: piece_df, out_path, kind.
    """

    x = np.arange(len(piece_df))
    labels = [f"g{i}" for i in piece_df["piece_id"]]
    width = 0.25

    if kind == "simbolico":
        rb = "symbolic_reference_based_score"
        rf = "symbolic_reference_free_score"
        gl = "symbolic_global_score"
        title = "Evaluación simbólica por pieza: referencia, plausibilidad interna y score global"
    elif kind == "espectral":
        rb = "spectral_reference_based_score"
        rf = "spectral_reference_free_score"
        gl = "spectral_global_score"
        title = "Evaluación espectral por pieza: referencia, plausibilidad interna y score global"
    else:
        raise ValueError("kind debe ser 'simbolico' o 'espectral'")

    fig, ax = plt.subplots(figsize=(16, 8))
    add_plausibility_bands(ax)

    ax.bar(x - width, piece_df[rb], width, color=BLUE, label="Reference-based", alpha=0.95)
    ax.bar(x, piece_df[rf], width, color=ORANGE, label="Reference-free", alpha=0.95)
    ax.bar(x + width, piece_df[gl], width, color=GREEN, label="Global ponderado", alpha=0.98)

    ax.set_title(title)
    ax.set_ylabel("Puntuación (0-100)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper left", ncol=3)
    ax.grid(axis="y", alpha=0.18)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_hbar(report: pd.DataFrame, order: List[str], labels_map: Dict[str, str], value_col: str, title: str, xlabel: str, fmt: str, out_path: Path) -> None:
    """
    Dibuja una visualizacion usada durante la evaluacion.

    Parametros principales: report, order, labels_map, value_col, title, xlabel, fmt, out_path.
    """

    plot_df = ensure_ordered_report(report, order)
    y_labels = [labels_map[f] for f in plot_df["feature"].astype(str)]
    values = pd.to_numeric(plot_df[value_col], errors="coerce").to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(16, 10))
    bars = ax.barh(y_labels, values, color=BLUE, alpha=0.95)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    style_hbar_axis(ax)

    if value_col == "oa":
        ax.set_xlim(0, 1.02)
        offset = 0.01
    else:
        max_v = float(np.nanmax(values)) if np.isfinite(values).any() else 1.0
        ax.set_xlim(0, max_v * 1.12)
        offset = max_v * 0.015

    annotate_horizontal_bars(ax, bars, fmt, offset)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Punto de entrada configurable del script."""

    global DPI

    args = parse_args()
    shared_data_dir = resolve_dir(args.data_dir, DATA_DIR)
    symbolic_data_dir = resolve_dir(args.symbolic_dir, SYMBOLIC_DATA_DIR if args.data_dir is None else shared_data_dir)
    spectral_data_dir = resolve_dir(args.spectral_dir, SPECTRAL_DATA_DIR if args.data_dir is None else shared_data_dir)
    output_dir = resolve_dir(args.output_dir, OUTPUT_DIR or (symbolic_data_dir / "graficos_evaluacion"))
    prefix = args.prefix
    suffix = args.suffix
    max_pieces = args.max_pieces
    DPI = args.dpi

    check_required_files(symbolic_data_dir, SYMBOLIC_REQUIRED_FILES, "SYMBOLIC_DATA_DIR")
    check_required_files(spectral_data_dir, SPECTRAL_REQUIRED_FILES, "SPECTRAL_DATA_DIR")
    output_dir.mkdir(parents=True, exist_ok=True)

    per_piece_details = read_json(symbolic_data_dir / "per_piece_details.json")
    per_piece_eval = pd.read_csv(symbolic_data_dir / "per_piece_evaluation.csv")
    reference_features = pd.read_csv(symbolic_data_dir / "reference_features.csv")
    summary_compact = pd.read_csv(symbolic_data_dir / "summary_compact.csv")

    spectral_features_all = pd.read_csv(spectral_data_dir / "spectral_features_all.csv")
    spectral_per_piece_details = read_json(spectral_data_dir / "spectral_per_piece_details.json")
    spectral_summary = read_json(spectral_data_dir / "spectral_summary.json")
    spectral_eval = pd.read_csv(spectral_data_dir / "spectral_evaluation.csv")

    print(f"[OK] Cargados ficheros simbólicos desde: {symbolic_data_dir}")
    print(f"[OK] Cargados ficheros espectrales desde: {spectral_data_dir}")
    print(f"[INFO] per_piece_details: {len(per_piece_details)} piezas")
    print(f"[INFO] spectral_per_piece_details: {len(spectral_per_piece_details)} piezas")
    print(f"[INFO] spectral_summary keys: {list(spectral_summary.keys())[:6]}...")
    print(f"[INFO] summary_compact rows: {len(summary_compact)}")

    sym_piece = add_piece_id(per_piece_eval)
    sym_piece = sym_piece.rename(columns={
        "reference_based_score": "symbolic_reference_based_score",
        "reference_free_score": "symbolic_reference_free_score",
        "global_score": "symbolic_global_score",
    })

    spec_piece = spectral_eval[spectral_eval["section"].eq("per_piece")].copy()
    spec_piece = add_piece_id(spec_piece)

    piece_cols_sym = [
        "piece_id",
        "symbolic_reference_based_score",
        "symbolic_reference_free_score",
        "symbolic_global_score",
    ]
    piece_cols_spec = [
        "piece_id",
        "spectral_reference_based_score",
        "spectral_reference_free_score",
        "spectral_global_score",
    ]
    piece_df = sym_piece[piece_cols_sym].merge(spec_piece[piece_cols_spec], on="piece_id", how="inner")
    piece_df = piece_df.sort_values("piece_id").head(max_pieces).reset_index(drop=True)

    if piece_df.empty:
        raise ValueError("No se han podido emparejar piezas simbólicas y espectrales por generated_from_jsonN.")

    optional_symbolic_global_report = symbolic_data_dir / OPTIONAL_SYMBOLIC_GLOBAL_REPORT_NAME
    if args.prefer_symbolic_global_report and optional_symbolic_global_report.exists():
        symbolic_report = pd.read_csv(optional_symbolic_global_report)
        print(f"[INFO] OA/KLD simbólico desde reporte opcional: {optional_symbolic_global_report.name}")
    else:
        gen_symbolic_features = symbolic_generated_features_from_details(per_piece_details, per_piece_eval)
        symbolic_report = global_distribution_report(
            reference_features,
            gen_symbolic_features,
            SYMBOLIC_ORDER,
            max_auto_bins=48,
        )
        print("[INFO] OA/KLD simbólico calculado desde reference_features.csv + per_piece_details.json")

    spectral_report = spectral_global_report_from_evaluation(spectral_eval)
    if not spectral_report.empty:
        print("[INFO] OA/KLD espectral desde spectral_evaluation.csv")
    else:
        spectral_report = spectral_global_report_from_summary(spectral_summary)
        if not spectral_report.empty:
            print("[INFO] OA/KLD espectral desde spectral_summary.json")
        else:
            spectral_report = spectral_global_report_from_features(spectral_features_all)
            print("[INFO] OA/KLD espectral calculado desde spectral_features_all.csv")

    def chart_path(number: int, stem: str) -> Path:
        return output_dir / f"{prefix}_{number:02d}_{stem}{suffix}.png"

    plot_score_comparison(
        piece_df,
        chart_path(1, "score_global_simbolico_vs_espectral_umbral"),
    )
    plot_componentes(
        piece_df,
        chart_path(2, "componentes_score_simbolico"),
        kind="simbolico",
    )
    plot_componentes(
        piece_df,
        chart_path(3, "componentes_score_espectral"),
        kind="espectral",
    )
    plot_hbar(
        symbolic_report,
        SYMBOLIC_ORDER,
        SYMBOLIC_LABELS,
        value_col="oa",
        title="OA por métrica simbólica",
        xlabel="Overlapping Area (0-1, mayor es mejor)",
        fmt="{:.3f}",
        out_path=chart_path(4, "oa_metricas_simbolicas"),
    )
    plot_hbar(
        symbolic_report,
        SYMBOLIC_ORDER,
        SYMBOLIC_LABELS,
        value_col="kld_real_to_gen",
        title="KLD por métrica simbólica",
        xlabel="KLD referencia→generado (menor es mejor)",
        fmt="{:.2f}",
        out_path=chart_path(5, "kld_metricas_simbolicas"),
    )
    plot_hbar(
        spectral_report,
        SPECTRAL_ORDER,
        SPECTRAL_LABELS,
        value_col="oa",
        title="OA por métrica espectral",
        xlabel="Overlapping Area (0-1, mayor es mejor)",
        fmt="{:.3f}",
        out_path=chart_path(6, "oa_metricas_espectrales"),
    )
    plot_hbar(
        spectral_report,
        SPECTRAL_ORDER,
        SPECTRAL_LABELS,
        value_col="kld_real_to_gen",
        title="KLD por métrica espectral",
        xlabel="KLD referencia→generado (menor es mejor)",
        fmt="{:.2f}",
        out_path=chart_path(7, "kld_metricas_espectrales"),
    )

    print("\n[OK] Gráficos generados:")
    for p in sorted(output_dir.glob(f"{prefix}_*{suffix}.png")):
        print(f" - {p}")


# Ejecucion directa del script.
if __name__ == "__main__":
    main()
