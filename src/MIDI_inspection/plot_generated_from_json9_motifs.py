from pathlib import Path
from collections import defaultdict
import pretty_midi
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ============================================================
# MIDI analizado: generated_from_json9.mid
# Duración aprox.: 72.625 s | 1 pista piano | tempo 120 BPM
# Resolución visual pensada para motivos dentro de textura polifónica.
# Solo se colorea la voz superior de cada motivo, no todo lo que suena debajo.
# ============================================================

MIDI_PATH = Path("generated_from_json9.mid")
OUT_PATH = Path("generated_from_json9_motifs.png")

# El MIDI está a 120 BPM: 1 negra = 0.5 s. La cuantización observada encaja con semicorcheas: 0.125 s.
QUANT_STEP = 0.125

# Secciones de fondo. Puedes cambiarlas cuando decidas tu análisis formal.
# Están en segundos. No colorean notas: solo sombrean el fondo.
SECTIONS = [
    {"start": 5.625, "end": 24.0, "label": "Inicio / presentación", "color": "tab:blue"},
    {"start": 24.0, "end": 48.0, "label": "Zona central", "color": "tab:orange"},
    {"start": 48.0, "end": 72.625, "label": "Repetición / cierre", "color": "tab:green"},
]

# Motivos detectados en la voz superior.
# IMPORTANTE: cada ocurrencia colorea únicamente la nota más aguda en cada onset dentro de la ventana,
# por eso sirve aunque haya acordes/bajo/otras voces simultáneas.
MOTIFS = [
    {
        "label": "A",
        "color": "crimson",
        "occurrences": [
            {"start": 48.00, "end": 49.76},
            {"start": 61.00, "end": 62.39},
        ],
        "description": "descenso 78-76-73-71-69-68-66-64 en voz superior",
    },
    {
        "label": "B",
        "color": "purple",
        "occurrences": [
            {"start": 40.25, "end": 41.65},
            {"start": 46.50, "end": 47.64},
        ],
        "description": "gesto descendente agudo similar, con variación final",
    },
]


def qtime(t: float, step: float = QUANT_STEP) -> float:
    return round(t / step) * step


def build_onset_groups(pm: pretty_midi.PrettyMIDI):
    groups = defaultdict(list)
    for inst_i, inst in enumerate(pm.instruments):
        if inst.is_drum:
            continue
        for note_i, note in enumerate(inst.notes):
            note._inst_i = inst_i
            note._note_i = note_i
            groups[qtime(note.start)].append(note)
    return groups


def top_voice_note_ids_in_window(groups, start: float, end: float):
    """Devuelve IDs de notas de la voz superior por onset, dentro de [start, end]."""
    selected = set()
    for onset in sorted(groups):
        if start <= onset <= end:
            notes = groups[onset]
            if not notes:
                continue
            top = max(notes, key=lambda n: (n.pitch, n.end - n.start))
            selected.add((top._inst_i, top._note_i))
    return selected


def build_highlight_map(pm, groups):
    highlight = {}
    label_positions = []

    for motif in MOTIFS:
        for occ_i, occ in enumerate(motif["occurrences"], start=1):
            ids = top_voice_note_ids_in_window(groups, occ["start"], occ["end"])
            for note_id in ids:
                highlight[note_id] = {
                    "color": motif["color"],
                    "label": f'{motif["label"]}{occ_i}',
                    "motif": motif["label"],
                }
            label_positions.append({
                "x": occ["start"],
                "y": 91,
                "text": f'{motif["label"]}{occ_i}',
                "color": motif["color"],
            })

    return highlight, label_positions


def plot_piano_roll():
    pm = pretty_midi.PrettyMIDI(str(MIDI_PATH))
    groups = build_onset_groups(pm)
    highlight, label_positions = build_highlight_map(pm, groups)

    all_notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    min_pitch = min(n.pitch for n in all_notes) - 3
    max_pitch = max(n.pitch for n in all_notes) + 6

    fig, ax = plt.subplots(figsize=(20, 8))

    # Fondo por secciones formales
    for sec in SECTIONS:
        ax.axvspan(sec["start"], sec["end"], color=sec["color"], alpha=0.08, zorder=0)
        ax.text(
            (sec["start"] + sec["end"]) / 2,
            max_pitch - 1,
            sec["label"],
            ha="center",
            va="top",
            fontsize=11,
            color=sec["color"],
            fontweight="bold",
        )

    # Notas
    for inst_i, inst in enumerate(pm.instruments):
        if inst.is_drum:
            continue
        for note_i, note in enumerate(inst.notes):
            key = (inst_i, note_i)
            h = highlight.get(key)

            if h:
                facecolor = h["color"]
                edgecolor = h["color"]
                alpha = 0.95
                linewidth = 1.3
                zorder = 3
                height = 0.92
            else:
                facecolor = "0.15"
                edgecolor = "0.15"
                alpha = 0.28
                linewidth = 0.2
                zorder = 2
                height = 0.78

            rect = patches.Rectangle(
                (note.start, note.pitch - height / 2),
                note.end - note.start,
                height,
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=alpha,
                linewidth=linewidth,
                zorder=zorder,
            )
            ax.add_patch(rect)

    # Cajas finas alrededor de cada ocurrencia para que se vea el bloque motivico sin colorear el acompañamiento
    for motif in MOTIFS:
        for occ_i, occ in enumerate(motif["occurrences"], start=1):
            ax.add_patch(
                patches.Rectangle(
                    (occ["start"], 62),
                    occ["end"] - occ["start"],
                    max_pitch - 64,
                    fill=False,
                    edgecolor=motif["color"],
                    linewidth=1.2,
                    linestyle="--",
                    alpha=0.8,
                    zorder=4,
                )
            )

    # Etiquetas de motivos
    for lab in label_positions:
        ax.text(
            lab["x"], lab["y"], lab["text"],
            color=lab["color"], fontsize=13, fontweight="bold",
            ha="left", va="bottom", zorder=5,
        )

    # Guías de compás aproximadas: 4/4 a 120 BPM => compás = 2 s
    bar_len = 2.0
    t = 0.0
    while t <= pm.get_end_time() + 0.01:
        ax.axvline(t, color="0.85", linewidth=0.5, zorder=1)
        t += bar_len

    ax.set_title("Piano roll anotado — motivos coloreados en voz superior", fontsize=15, fontweight="bold")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Altura MIDI")
    ax.set_xlim(0, pm.get_end_time())
    ax.set_ylim(min_pitch, max_pitch)
    ax.grid(axis="y", alpha=0.15)

    legend_text = " | ".join([f'{m["label"]}: {m["description"]}' for m in MOTIFS])
    fig.text(0.01, 0.01, legend_text, fontsize=9, ha="left", va="bottom")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Guardado en: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    plot_piano_roll()
