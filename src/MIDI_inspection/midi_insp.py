from pathlib import Path
import pretty_midi
import matplotlib.pyplot as plt
import matplotlib.patches as patches

MIDI_PATH = Path(r"../../output/generation_finetuning_tfg_third/best_train/generated_from_json9.mid")
OUT_PATH = Path(r"finetuning_second_g9.png")

TOL = 0.08  # tolerancia en segundos para emparejar notas

MOTIFS = [
    {
        "label": "Motivo A",
        "color": "red",
        "notes": [
            {"pitch": 72, "start": 3.20, "end": 3.60},
            {"pitch": 74, "start": 3.60, "end": 4.00},
            {"pitch": 76, "start": 4.00, "end": 4.45},
        ],
    },
    {
        "label": "A'",
        "color": "red",
        "notes": [
            {"pitch": 72, "start": 10.30, "end": 10.70},
            {"pitch": 74, "start": 10.70, "end": 11.10},
            {"pitch": 76, "start": 11.10, "end": 11.55},
        ],
    },
]

SECTIONS = [
    {"start": 0, "end": 18, "label": "Exposición", "color": "tab:blue"},
    {"start": 18, "end": 34, "label": "Desarrollo", "color": "tab:orange"},
]


def matches_annotation(note, ann):
    return (
        note.pitch == ann["pitch"]
        and abs(note.start - ann["start"]) <= TOL
        and abs(note.end - ann["end"]) <= TOL
    )


def get_note_annotation(note):
    for motif in MOTIFS:
        for ann in motif["notes"]:
            if matches_annotation(note, ann):
                return motif["color"], motif["label"]
    return "black", None


def plot():
    pm = pretty_midi.PrettyMIDI(str(MIDI_PATH))
    fig, ax = plt.subplots(figsize=(18, 7))

    # Fondos de sección
    for sec in SECTIONS:
        ax.axvspan(sec["start"], sec["end"], color=sec["color"], alpha=0.10)
        ax.text(
            (sec["start"] + sec["end"]) / 2,
            111,
            sec["label"],
            ha="center",
            fontweight="bold",
            color=sec["color"],
        )

    # Piano roll
    label_positions = {}

    for inst in pm.instruments:
        if inst.is_drum:
            continue

        for note in inst.notes:
            color, label = get_note_annotation(note)

            rect = patches.Rectangle(
                (note.start, note.pitch - 0.4),
                note.end - note.start,
                0.8,
                facecolor=color,
                edgecolor=color,
                alpha=0.85 if label else 0.35,
                linewidth=1.2 if label else 0.2,
            )
            ax.add_patch(rect)

            if label:
                label_positions.setdefault(label, []).append((note.start, note.pitch, color))

    # Etiquetas de motivo
    for label, positions in label_positions.items():
        x = min(p[0] for p in positions)
        y = max(p[1] for p in positions) + 2
        color = positions[0][2]
        ax.text(x, y, label, color=color, fontsize=12, fontweight="bold")

    ax.set_title(f"Piano roll anotado: {MIDI_PATH.name}")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Pitch MIDI")
    ax.set_ylim(20, 115)
    ax.set_xlim(0, pm.get_end_time())
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300)
    plt.show()


if __name__ == "__main__":
    plot()