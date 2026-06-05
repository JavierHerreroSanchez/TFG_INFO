"""
Genera una imagen de análisis para generated_from_json7 del segundo pretraining.

No plantea una estructura formal completa: solo señala el motivo original y dos
zonas de alta riqueza armónica con direccionalidad visible.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pretty_midi


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]

MIDI_PATH = ROOT_DIR / "output" / "generation_pretraining_tfg_second" / "generated_from_json7.mid"
OUT_PATH = SCRIPT_DIR / "musical_analisis_pretraining_v2_generated_from_json7.png"

MOTIVE_BRACKET = {
    "label": "motivo original",
    "start": 0.0,
    "end": 13.0,
    "color": "#222222",
}

RICH_HARMONY_ZONES = [
    {
        "label": "zona de alta riqueza armonica\ny direccionalidad",
        "start": 36.0,
        "end": 44.0,
        "color": "#ff7f0e",
    },
    {
        "label": "zona de alta riqueza armonica\ny direccionalidad",
        "start": 60.0,
        "end": 65.0,
        "color": "#ff7f0e",
    },
]

BACKGROUND_NOTE_COLOR = "#222222"
BACKGROUND_ALPHA = 0.20


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


def bounded_end(item, end_time):
    return min(item["end"], end_time)


def draw_bracket(ax, item, y, zorder=7):
    start = item["start"]
    end = item["end"]
    color = item["color"]
    ax.hlines(y=y, xmin=start, xmax=end, color=color, linewidth=1.9, alpha=0.95, zorder=zorder)
    ax.vlines(x=[start, end], ymin=y - 1.0, ymax=y + 1.0, color=color, linewidth=1.9, alpha=0.95, zorder=zorder)
    ax.text(
        (start + end) / 2,
        y + 1.25,
        item["label"],
        ha="center",
        va="bottom",
        fontsize=9.8,
        fontweight="bold",
        color=color,
        zorder=zorder + 1,
    )


def plot_from_midi(midi_path: Path, out_path: Path):
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = collect_notes(pm)
    if not notes:
        raise RuntimeError("No se encontraron notas en el MIDI.")

    min_pitch = min(n["pitch"] for n in notes) - 5
    max_pitch = max(n["pitch"] for n in notes) + 10
    end_time = max(n["end"] for n in notes)

    fig, ax = plt.subplots(figsize=(18, 7.2))

    for zone in RICH_HARMONY_ZONES:
        start = zone["start"]
        end = bounded_end(zone, end_time)
        if start >= end:
            continue
        ax.axvspan(start, end, color=zone["color"], alpha=0.18, linewidth=0, zorder=0)
        draw_bracket(ax, {**zone, "end": end}, max_pitch - 6.5, zorder=8)

    draw_bracket(ax, {**MOTIVE_BRACKET, "end": bounded_end(MOTIVE_BRACKET, end_time)}, max_pitch - 2.8)

    for note in notes:
        ax.add_patch(
            patches.Rectangle(
                (note["start"], note["pitch"] - 0.35),
                max(note["duration"], 0.018),
                0.70,
                facecolor=BACKGROUND_NOTE_COLOR,
                edgecolor=BACKGROUND_NOTE_COLOR,
                linewidth=0.10,
                alpha=BACKGROUND_ALPHA,
            )
        )

    legend_handles = [
        patches.Patch(
            color=RICH_HARMONY_ZONES[0]["color"],
            alpha=0.45,
            label="Alta riqueza armonica con direccionalidad visible",
        )
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, framealpha=0.9, fontsize=9.5)

    ax.set_xlim(0, end_time)
    ax.set_ylim(min_pitch, max_pitch)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Altura MIDI")
    ax.set_title("Analisis: motivo original y zonas de riqueza armonica", fontsize=14)
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
