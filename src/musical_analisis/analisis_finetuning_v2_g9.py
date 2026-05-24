"""
Calcula analisis musicales agregados sobre las muestras generadas.

Se usa para obtener evidencias cuantitativas y visuales del comportamiento musical del sistema.
"""

from pathlib import Path
import pretty_midi
import matplotlib.pyplot as plt
import matplotlib.patches as patches

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]

MIDI_PATH = ROOT_DIR / "output" / "generation_finetuning_tfg_second" / "generated_from_json9.mid"
OUT_PATH = SCRIPT_DIR / "musical_analisis_finetuning_v2_generated_from_json9.png"

# Estructura formal aproximada, en segundos.
SECTIONS = [
    {"label": "Sección A", "start": 0.0, "end": 31.0, "color": "#9ecae1"},
    {"label": "Sección B", "start": 31.0, "end": 56.0, "color": "#fdd0a2"},
    {"label": "Zona cadencial", "start": 56.0, "end": 72.7, "color": "#c7e9c0"},
]

# Anotaciones concretas.
# mode="top": colorea solo la voz superior de la ventana.
# mode="lower": colorea solo notas graves/medias dentro de la ventana, útil para acompañamiento.
ANNOTATIONS = [
    # Motivo y contestacion localizado entre 35 y 37 s.
    {"label": "Motivo", "start": 35.56, "end": 36.25, "color": "#d62728", "mode": "top"},
    {"label": "Contestación", "start": 36.68, "end": 37.44, "color": "#9467bd", "mode": "top"},

    # Motivo repetido posterior.
    {"label": "A", "start": 48.00, "end": 49.76, "color": "#1f77b4", "mode": "top"},
    {"label": "A'", "start": 61.00, "end": 62.39, "color": "#1f77b4", "mode": "top"},


    # Motivos descendentes entre 40 y 50 s.
    # Marcados en la voz superior para no colorear el acompañamiento.
    {"label": "Rep. desc.", "start": 40.06, "end": 41.44, "color": "#e377c2", "mode": "top"},
    {"label": "Rep. desc.", "start": 42.12, "end": 43.38, "color": "#e377c2", "mode": "top"},
    {"label": "Rep. desc.", "start": 43.68, "end": 44.56, "color": "#e377c2", "mode": "top"},
    {"label": "Rep. desc.", "start": 44.93, "end": 45.75, "color": "#e377c2", "mode": "top"},
    {"label": "Rep. desc.", "start": 46.18, "end": 47.63, "color": "#e377c2", "mode": "top"},


    # Bajo Alberti / acompañamiento Alberti-like marcado manualmente como capa de notas graves.
    # Rangos ajustables para estrechar o desplazar el marcado.
    {"label": "Bajo arpegiado", "start": 10.5, "end": 18.50, "color": "#2ca02c", "mode": "lower", "pitch_max": 70},
    {"label": "Bajo arpegiado", "start": 32.31, "end": 38.69, "color": "#2ca02c", "mode": "lower", "pitch_max": 60},
    {"label": "Bajo arpegiado", "start": 41.06, "end": 47.92, "color": "#2ca02c", "mode": "lower", "pitch_max": 60},
]


# Subsecciones internas dentro de la estructura formal.
# Se dibujan como una línea horizontal con dos barritas verticales.
SUBSECTIONS = [
    {"label": "Motivo original", "start": 0.0, "end": 18.0, "color": "#222222"},
]

BACKGROUND_NOTE_COLOR = "#222222"
BACKGROUND_ALPHA = 0.23
ANNOTATED_ALPHA = 0.96
TOP_ONSET_TOL = 0.045

def draw_subsections(ax, subsections, y):
    """Dibuja subsecciones como línea horizontal + barritas verticales."""
    for sub in subsections:
        start = sub["start"]
        end = sub["end"]
        color = sub.get("color", "#222222")
        label = sub["label"]

        ax.hlines(y=y, xmin=start, xmax=end, color=color, linewidth=1.8, alpha=0.95, zorder=5)
        ax.vlines(x=[start, end], ymin=y - 1.2, ymax=y + 1.2, color=color, linewidth=1.8, alpha=0.95, zorder=5)
        ax.text(
            (start + end) / 2,
            y + 1.6,
            label,
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color=color,
            zorder=6,
        )

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


def top_voice_note_ids(notes, start, end):

    candidates = [
        (i, n) for i, n in enumerate(notes)
        if start - 1e-6 <= n["start"] <= end + 1e-6
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


def lower_note_ids(notes, start, end, pitch_max=60):

    return {
        i for i, n in enumerate(notes)
        if start - 1e-6 <= n["start"] <= end + 1e-6 and n["pitch"] <= pitch_max
    }


def build_note_annotations(notes):
    """
    Construye una estructura auxiliar usada por el resto del flujo.

    """

    note_styles = {}
    label_positions = []

    for ann in ANNOTATIONS:
        mode = ann.get("mode")
        if mode == "top":
            ids = top_voice_note_ids(notes, ann["start"], ann["end"])
        elif mode == "lower":
            ids = lower_note_ids(notes, ann["start"], ann["end"], ann.get("pitch_max", 60))
        else:
            ids = {
                i for i, n in enumerate(notes)
                if ann["start"] <= n["start"] and n["end"] <= ann["end"]
            }

        selected_notes = [notes[i] for i in ids]
        for i in ids:
            note_styles[i] = ann

        if selected_notes:
            x = min(n["start"] for n in selected_notes)
            y = max(n["pitch"] for n in selected_notes) + 1.6
            label_positions.append((x, y, ann["label"], ann["color"]))

    return note_styles, label_positions


def plot():
    """Dibuja una visualizacion usada durante la evaluacion."""

    pm = pretty_midi.PrettyMIDI(str(MIDI_PATH))
    notes = collect_notes(pm)
    note_styles, label_positions = build_note_annotations(notes)

    min_pitch = min(n["pitch"] for n in notes) - 4
    max_pitch = max(n["pitch"] for n in notes) + 8
    end_time = pm.get_end_time()

    fig, ax = plt.subplots(figsize=(18, 7))

    # Bandas de estructura, con texto dentro del piano roll.
    for sec in SECTIONS:
        ax.axvspan(sec["start"], sec["end"], color=sec["color"], alpha=0.23, linewidth=0)
        ax.text(
            (sec["start"] + sec["end"]) / 2,
            max_pitch - 2,
            sec["label"],
            ha="center",
            va="top",
            fontsize=13,
            fontweight="bold",
            color="#222222",
        )

    # Subsecciones internas, colocadas por debajo del texto de la sección.
    draw_subsections(ax, SUBSECTIONS, y=max_pitch - 7)

    # Notas del piano roll.
    for i, n in enumerate(notes):
        ann = note_styles.get(i)
        if ann:
            face = ann["color"]
            edge = ann["color"]
            alpha = ANNOTATED_ALPHA
            lw = 0.75
            height = 0.9
        else:
            face = BACKGROUND_NOTE_COLOR
            edge = BACKGROUND_NOTE_COLOR
            alpha = BACKGROUND_ALPHA
            lw = 0.12
            height = 0.74

        ax.add_patch(
            patches.Rectangle(
                (n["start"], n["pitch"] - height / 2),
                max(n["duration"], 0.02),
                height,
                facecolor=face,
                edgecolor=edge,
                linewidth=lw,
                alpha=alpha,
            )
        )

    # Etiquetas dentro del piano roll.
    for x, y, label, color in label_positions:
        ax.text(
            x,
            y,
            label,
            color=color,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    legend_handles = [
        patches.Patch(color="#d62728", label="Motivo principal"),
        patches.Patch(color="#9467bd", label="Contestacion"),
        patches.Patch(color="#1f77b4", label="Motivo A y repeticion"),
        patches.Patch(color="#e377c2", label="Repeticiones descendentes"),
        patches.Patch(color="#2ca02c", label="Bajo arpegiado"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, framealpha=0.9, fontsize=9.5)

    ax.set_xlim(0, end_time)
    ax.set_ylim(min_pitch, max_pitch)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Altura MIDI")
    ax.set_title("Análisis estructural", fontsize=14)
    ax.grid(axis="y", alpha=0.18)
    ax.grid(axis="x", alpha=0.08)

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {OUT_PATH}")


# Ejecucion directa del script.
if __name__ == "__main__":
    plot()
