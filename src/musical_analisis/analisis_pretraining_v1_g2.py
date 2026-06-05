"""
Genera una imagen de análisis motívico para generated_from_json2 del primer
pretraining.

No marca estructura formal. Solo dibuja, en ventanas manuales, las seis notas
superiores del motivo inicial tipo "Here comes the sun" y sus repeticiones.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pretty_midi


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]

MIDI_PATH = ROOT_DIR / "output" / "generation_pretraining_tfg_first" / "generated_from_json2.mid"
OUT_PATH = SCRIPT_DIR / "musical_analisis_pretraining_v1_generated_from_json2.png"

# Notas concretas del motivo. Se usan tiempos aproximados de ataque y altura
# MIDI para no depender de filtros generales que puedan coger notas ajenas.
MOTIF_OCCURRENCES = [
    {
        "label": "motivo principal",
        "notes": [(9.50, 85), (9.75, 81), (10.06, 83), (10.38, 85)],
    },
    {
        "label": "repeticion",
        "notes": [(57.31, 85), (57.62, 81), (57.94, 83), (58.25, 85)],
    },
    {
        "label": "repeticion",
        "notes": [(71.12, 80), (71.44, 78), (72.69, 81), (73.25, 80)],
    },
    {
        "label": "repeticion",
        "notes": [(96.56, 80), (96.88, 85), (97.12, 83), (98.00, 85)],
    },
    {
        "label": "repeticion",
        "notes": [(108.75, 73), (109.06, 71), (109.69, 73)],
    },
]

PIECE_MOTIF_BRACKET = {
    "label": "motivo original",
    "start": 0.62,
    "end": 35.0,
    "color": "#222222",
}

MOTIF_COLOR = "#d62728"
BACKGROUND_NOTE_COLOR = "#222222"
BACKGROUND_ALPHA = 0.18
ANNOTATED_ALPHA = 0.98


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


def find_note_id(notes, start, pitch, tolerance=0.04):
    candidates = [
        (i, n) for i, n in enumerate(notes)
        if n["pitch"] == pitch and abs(n["start"] - start) <= tolerance
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda pair: abs(pair[1]["start"] - start))[0]


def build_note_annotations(notes):
    note_styles = {}
    label_positions = []

    for occurrence in MOTIF_OCCURRENCES:
        note_ids = {
            note_id
            for start, pitch in occurrence["notes"]
            if (note_id := find_note_id(notes, start, pitch)) is not None
        }
        if not note_ids:
            continue

        for note_id in note_ids:
            note_styles[note_id] = {"color": MOTIF_COLOR, "height": 0.98, "linewidth": 0.85}

        selected_notes = [notes[i] for i in note_ids]
        label_positions.append({
            "x": min(n["start"] for n in selected_notes),
            "y": max(n["pitch"] for n in selected_notes) + 1.4,
            "label": occurrence["label"],
            "color": MOTIF_COLOR,
        })

    return note_styles, label_positions


def spread_labels(label_positions, min_x_gap=8.5, min_y_gap=2.2):
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


def draw_bracket(ax, bracket, y):
    start = bracket["start"]
    end = bracket["end"]
    color = bracket["color"]
    ax.hlines(y=y, xmin=start, xmax=end, color=color, linewidth=1.8, alpha=0.95, zorder=7)
    ax.vlines(x=[start, end], ymin=y - 1.0, ymax=y + 1.0, color=color, linewidth=1.8, alpha=0.95, zorder=7)
    ax.text(
        (start + end) / 2,
        y + 1.2,
        bracket["label"],
        ha="center",
        va="bottom",
        fontsize=9.8,
        fontweight="bold",
        color=color,
        zorder=8,
    )


def plot_from_midi(midi_path: Path, out_path: Path):
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = collect_notes(pm)
    if not notes:
        raise RuntimeError("No se encontraron notas en el MIDI.")

    note_styles, label_positions = build_note_annotations(notes)
    min_pitch = min(n["pitch"] for n in notes) - 5
    max_pitch = max(n["pitch"] for n in notes) + 9
    end_time = max(n["end"] for n in notes)

    fig, ax = plt.subplots(figsize=(22, 7.2))

    draw_bracket(ax, PIECE_MOTIF_BRACKET, max_pitch - 5.0)

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
            fontsize=9.8,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    ax.legend(
        handles=[patches.Patch(color=MOTIF_COLOR, label="Motivo principal y repeticiones")],
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        fontsize=9.5,
    )
    ax.set_xlim(0, end_time)
    ax.set_ylim(min_pitch, max_pitch)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Altura MIDI")
    ax.set_title("Analisis de motivos", fontsize=14)
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
