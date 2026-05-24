"""
Genera una imagen de analisis musicologico para generated_from_json1 del
primer finetuning.

La lectura formal evita forzar una reexposicion: se entiende como una exposicion
larga, con material de desarrollo interno, seguida de un desarrollo posterior.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pretty_midi


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]

MIDI_PATH = ROOT_DIR / "output" / "generation_finetuning_tfg_first" / "generated_from_json1.mid"
OUT_PATH = SCRIPT_DIR / "musical_analisis_finetuning_v1_generated_from_json1.png"

# Estructura tentativa segun la escucha: no se fuerza reexposicion clara.
FORM_SECTIONS = [
    {"label": "Exposicion", "start": 0.0, "end": 154.0, "color": "#d9edf7"},
    {"label": "Desarrollo", "start": 154.0, "end": 999.0, "color": "#dff0d8"},
]

SUBSECTIONS = [
    {"label": "Seccion A", "start": 0.0, "end": 51.0, "color": "#222222"},
    {"label": "Seccion B", "start": 51.0, "end": 106.0, "color": "#222222"},
    {"label": "zona cadencial", "start": 106.0, "end": 154.0, "color": "#222222"},
]

ANALYTIC_BRACKETS = [
    {"label": "motivo original", "start": 0.44, "end": 12.0, "color": "#222222"},
]

# Motivos amplios senalados manualmente por su recurrencia o contraste.
# mode="top" colorea solo la voz superior de cada ventana.
MOTIF_ANNOTATIONS = [
    {"label": "motivo A", "start": 12.0, "end": 23.0, "color": "#d62728", "mode": "top", "pitch_min": 57},
    {"label": "A'", "start": 26.0, "end": 36.0, "color": "#d62728", "mode": "top", "pitch_min": 62},
    {"label": "", "start": 36.0, "end": 40.0, "color": "#d62728", "mode": "top", "pitch_min": 55},
    {"label": "motivo B", "start": 40.0, "end": 51.0, "color": "#1f77b4", "mode": "top", "pitch_min": 60},
    {"label": "motivo C", "start": 69.0, "end": 73.0, "color": "#9467bd", "mode": "top", "pitch_min": 68},
]

MOTIF_A_COLOR = "#d62728"
MOTIF_B_COLOR = "#1f77b4"
MOTIF_C_COLOR = "#9467bd"
BACKGROUND_NOTE_COLOR = "#222222"
BACKGROUND_ALPHA = 0.18
ANNOTATED_ALPHA = 0.96
TOP_ONSET_TOL = 0.045


def collect_notes(pm):
    notes = []
    for inst_i, inst in enumerate(pm.instruments):
        if inst.is_drum:
            continue
        for note in inst.notes:
            notes.append({
                "inst": inst_i,
                "pitch": int(note.pitch),
                "start": float(note.start),
                "end": float(note.end),
                "duration": float(note.end - note.start),
                "velocity": int(note.velocity),
            })
    notes.sort(key=lambda n: (n["start"], n["pitch"], n["end"]))
    return notes


def top_voice_note_ids(notes, start, end, pitch_min=None):
    candidates = [
        (i, n) for i, n in enumerate(notes)
        if start - 1e-6 <= n["start"] <= end + 1e-6
        and (pitch_min is None or n["pitch"] >= pitch_min)
    ]
    selected = set()
    used = [False] * len(candidates)
    for a, (idx, note) in enumerate(candidates):
        if used[a]:
            continue
        group = [(idx, note)]
        used[a] = True
        for b in range(a + 1, len(candidates)):
            if used[b]:
                continue
            idx2, note2 = candidates[b]
            if abs(note2["start"] - note["start"]) <= TOP_ONSET_TOL:
                group.append((idx2, note2))
                used[b] = True
        top_idx, _ = max(group, key=lambda pair: pair[1]["pitch"])
        selected.add(top_idx)
    return selected


def bounded_end(item, end_time):
    return min(item["end"], end_time)


def build_note_annotations(notes):
    note_styles = {}
    label_positions = []

    for ann in MOTIF_ANNOTATIONS:
        ids = top_voice_note_ids(notes, ann["start"], ann["end"], ann.get("pitch_min"))
        selected_notes = [notes[i] for i in ids]
        for i in ids:
            note_styles[i] = {"color": ann["color"], "height": 0.96, "linewidth": 0.85}
        if selected_notes:
            label_positions.append({
                "x": min(n["start"] for n in selected_notes),
                "y": max(n["pitch"] for n in selected_notes) + 1.4,
                "label": ann["label"],
                "color": ann["color"],
            })

    return note_styles, label_positions


def spread_labels(label_positions, min_x_gap=4.8, min_y_gap=2.0):
    placed = []
    for label in sorted(label_positions, key=lambda item: (item["x"], item["y"])):
        y = label["y"]
        for _ in range(8):
            collision = any(
                abs(label["x"] - other["x"]) < min_x_gap and abs(y - other["y"]) < min_y_gap
                for other in placed
            )
            if not collision:
                break
            y += min_y_gap
        placed.append({**label, "y": y})
    return placed


def draw_bracket(ax, label, start, end, y, color, zorder=7, fontsize=9.5):
    ax.hlines(y=y, xmin=start, xmax=end, color=color, linewidth=1.7, alpha=0.95, zorder=zorder)
    ax.vlines(x=[start, end], ymin=y - 1.0, ymax=y + 1.0, color=color, linewidth=1.7, alpha=0.95, zorder=zorder)
    ax.text(
        (start + end) / 2,
        y + 1.25,
        label,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        zorder=zorder + 1,
    )


def plot_from_midi(midi_path: Path, out_path: Path):
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = collect_notes(pm)
    if not notes:
        raise RuntimeError("No se encontraron notas en el MIDI.")

    note_styles, label_positions = build_note_annotations(notes)
    min_pitch = min(n["pitch"] for n in notes) - 5
    max_pitch = max(n["pitch"] for n in notes) + 11
    end_time = max(n["end"] for n in notes)

    fig, ax = plt.subplots(figsize=(24, 7.8))

    for sec in FORM_SECTIONS:
        start = sec["start"]
        end = bounded_end(sec, end_time)
        if start >= end:
            continue
        ax.axvspan(start, end, color=sec["color"], alpha=0.48, linewidth=0)
        ax.text(
            (start + end) / 2,
            max_pitch - 2.1,
            sec["label"],
            ha="center",
            va="top",
            fontsize=13,
            fontweight="bold",
            color="#222222",
        )

    for sub in SUBSECTIONS:
        start = sub["start"]
        end = bounded_end(sub, end_time)
        if start < end:
            draw_bracket(ax, sub["label"], start, end, max_pitch - 7.0, sub["color"], fontsize=9)

    for bracket in ANALYTIC_BRACKETS:
        start = bracket["start"]
        end = bounded_end(bracket, end_time)
        if start < end:
            draw_bracket(ax, bracket["label"], start, end, max_pitch - 12.2, bracket["color"], fontsize=9)

    for i, note in enumerate(notes):
        style = note_styles.get(i)
        if style:
            face = style["color"]
            edge = style["color"]
            alpha = ANNOTATED_ALPHA
            linewidth = style["linewidth"]
            height = style["height"]
        else:
            face = BACKGROUND_NOTE_COLOR
            edge = BACKGROUND_NOTE_COLOR
            alpha = BACKGROUND_ALPHA
            linewidth = 0.10
            height = 0.70

        ax.add_patch(
            patches.Rectangle(
                (note["start"], note["pitch"] - height / 2),
                max(note["duration"], 0.018),
                height,
                facecolor=face,
                edgecolor=edge,
                linewidth=linewidth,
                alpha=alpha,
            )
        )

    for label in spread_labels(label_positions):
        ax.text(
            label["x"],
            label["y"],
            label["label"],
            color=label["color"],
            fontsize=9.5,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    legend_handles = [
        patches.Patch(color=MOTIF_A_COLOR, label="motivo A"),
        patches.Patch(color=MOTIF_B_COLOR, label="motivo B"),
        patches.Patch(color=MOTIF_C_COLOR, label="motivo C"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, framealpha=0.9, fontsize=9.2)

    ax.set_xlim(0, end_time)
    ax.set_ylim(min_pitch, max_pitch)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Altura MIDI")
    ax.set_title("Analisis estructural", fontsize=14)
    ax.grid(axis="y", alpha=0.18)
    ax.grid(axis="x", alpha=0.08)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {out_path}")


def plot():
    if MIDI_PATH.exists():
        plot_from_midi(MIDI_PATH, OUT_PATH)
    else:
        raise FileNotFoundError(f"No se encontro el MIDI de entrada: {MIDI_PATH}")


if __name__ == "__main__":
    plot()
