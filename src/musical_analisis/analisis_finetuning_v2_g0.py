"""
Genera una imagen de analisis musicologico para generated_from_json0.

Las zonas formales y los elementos analiticos se fijan manualmente para que la
figura refleje una lectura musicologica concreta: motivo original, estructuras
de subida/bajada en B, bajo arpegiado y cierre cadencial.
"""

from pathlib import Path
import struct
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]

MIDI_PATH = ROOT_DIR / "output" / "generation_finetuning_tfg_second" / "generated_from_json0.mid"
OUT_PATH = SCRIPT_DIR / "musical_analisis_finetuning_v2_generated_from_json0.png"

# Fallback para ejecuciones fuera del repositorio.
SANDBOX_MIDI_PATH = Path("/mnt/data/generated_from_json0.mid")
SANDBOX_OUT_PATH = Path("/mnt/data/analisis_generated_from_json0_musicologico.png")

# Estructura formal fijada manualmente.
SECTIONS = [
    {"label": "Sección A", "start": 0.0, "end": 19.2, "color": "#d9edf7"},
    {"label": "Sección B", "start": 19.2, "end": 45.0, "color": "#ece8f4"},
    {"label": "Zona cadencial", "start": 45.0, "end": 999.0, "color": "#fff0df"},
]

# Llaves analiticas superiores: no colorean notas, solo senalan zonas.
ANALYTIC_BRACKETS = [
    {"label": "motivo original", "start": 4.31, "end": 19.2, "color": "#222222"},
    {"label": "impulso cadencial", "start": 45.0, "end": 68.0, "color": "#222222"},
]

# Anotaciones melodicas. mode="top" colorea solo la voz superior del intervalo.
TOP_ANNOTATIONS = [
    {"label": "", "start": 4.31, "end": 5.38, "color": "#d62728", "mode": "top"},
    {"label": "rep. motivo", "start": 9.75, "end": 10.69, "color": "#d62728", "mode": "top"},
    {"label": "rep. motivo", "start": 15.25, "end": 16.38, "color": "#d62728", "mode": "top"},

    # Estructuras ascendente-descendentes que articulan la seccion B.
    {"label": "asc.-desc.", "start": 21.38, "end": 23.95, "color": "#1f77b4", "mode": "top"},
    {"label": "motivo A", "start": 24.94, "end": 26.70, "color": "#1f77b4", "mode": "top"},
    {"label": "A'", "start": 28.38, "end": 30.76, "color": "#1f77b4", "mode": "top"},
    {"label": "A''", "start": 33.50, "end": 35.32, "color": "#1f77b4", "mode": "top"},
]

# Momentos en los que el bajo actua como patron arpegiado, no como ostinato.
BASS_ARPEGGIO_ANNOTATIONS = [
    {"label": "bajo arpegiado", "start": 5.31, "end": 45.0, "pitch_max": 67},
    {"label": "bajo arp. cad.", "start": 57.31, "end": 63.38, "pitch_max": 67, "color": "#17becf"},
    {"label": "", "start": 63.88, "end": 67.32, "pitch_max": 67, "color": "#17becf"},
    {"label": "", "start": 68.19, "end": 72.45, "pitch_max": 67, "color": "#17becf"},
]

MOTIVE_COLOR = "#d62728"
SEQUENCE_COLOR = "#1f77b4"
BASS_COLOR = "#2ca02c"
CADENTIAL_BASS_COLOR = "#17becf"
BACKGROUND_NOTE_COLOR = "#222222"
BACKGROUND_ALPHA = 0.20
ANNOTATED_ALPHA = 0.96
TOP_ONSET_TOL = 0.045


def read_var(data, i):
    value = 0
    while True:
        b = data[i]
        i += 1
        value = (value << 7) | (b & 0x7F)
        if not (b & 0x80):
            break
    return value, i


def parse_midi(path: Path):
    data = path.read_bytes()
    if data[:4] != b"MThd":
        raise ValueError("No parece un archivo MIDI valido: falta cabecera MThd.")
    header_len = struct.unpack(">I", data[4:8])[0]
    _fmt, n_tracks, division = struct.unpack(">HHH", data[8:14])
    if division & 0x8000:
        raise ValueError("Este script solo contempla MIDI con ticks por negra, no SMPTE.")

    i = 8 + header_len
    tempos = [(0, 500000)]
    raw_notes = []

    for track_idx in range(n_tracks):
        if data[i:i + 4] != b"MTrk":
            raise ValueError("Falta cabecera MTrk.")
        i += 4
        track_len = struct.unpack(">I", data[i:i + 4])[0]
        i += 4
        track_end = i + track_len
        tick = 0
        running_status = None
        active = defaultdict(list)

        while i < track_end:
            delta, i = read_var(data, i)
            tick += delta
            status = data[i]
            if status < 0x80:
                if running_status is None:
                    raise ValueError("Running status sin estado previo.")
                status = running_status
            else:
                i += 1
                if status < 0xF0:
                    running_status = status

            if status == 0xFF:
                meta_type = data[i]
                i += 1
                length, i = read_var(data, i)
                payload = data[i:i + length]
                i += length
                if meta_type == 0x51 and length == 3:
                    tempos.append((tick, int.from_bytes(payload, "big")))
                continue
            if status in (0xF0, 0xF7):
                length, i = read_var(data, i)
                i += length
                continue

            event_type = status & 0xF0
            channel = status & 0x0F
            if event_type in (0x80, 0x90):
                pitch = data[i]
                velocity = data[i + 1]
                i += 2
                key = (channel, pitch)
                if event_type == 0x90 and velocity > 0:
                    active[key].append((tick, velocity, track_idx))
                elif active[key]:
                    start_tick, start_velocity, start_track = active[key].pop(0)
                    if tick > start_tick:
                        raw_notes.append({
                            "track": start_track,
                            "channel": channel,
                            "pitch": int(pitch),
                            "velocity": int(start_velocity),
                            "start_tick": int(start_tick),
                            "end_tick": int(tick),
                        })
            elif event_type in (0xA0, 0xB0, 0xE0):
                i += 2
            elif event_type in (0xC0, 0xD0):
                i += 1
            else:
                raise ValueError(f"Evento MIDI no soportado: {hex(status)}")
        i = track_end

    tempos = sorted(set(tempos))

    def tick_to_seconds(tick_value):
        seconds = 0.0
        last_tick = 0
        tempo = tempos[0][1]
        for tempo_tick, new_tempo in tempos[1:]:
            if tempo_tick >= tick_value:
                break
            seconds += (tempo_tick - last_tick) * tempo / 1_000_000.0 / division
            last_tick = tempo_tick
            tempo = new_tempo
        seconds += (tick_value - last_tick) * tempo / 1_000_000.0 / division
        return seconds

    notes = []
    for raw in raw_notes:
        start = tick_to_seconds(raw["start_tick"])
        end = tick_to_seconds(raw["end_tick"])
        notes.append({
            "pitch": raw["pitch"],
            "velocity": raw["velocity"],
            "start": start,
            "end": end,
            "duration": max(end - start, 0.0),
        })
    notes.sort(key=lambda x: (x["start"], x["pitch"], x["end"]))
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


def bass_note_ids(notes, start, end, pitch_max=67):
    return {
        i for i, n in enumerate(notes)
        if start - 1e-6 <= n["start"] <= end + 1e-6 and n["pitch"] <= pitch_max
    }


def bounded_end(item, end_time):
    return min(item["end"], end_time)


def build_note_annotations(notes):
    note_styles = {}
    label_positions = []

    for ann in TOP_ANNOTATIONS:
        ids = top_voice_note_ids(notes, ann["start"], ann["end"])
        selected_notes = [notes[i] for i in ids]
        for i in ids:
            note_styles[i] = {"color": ann["color"], "height": 0.95, "linewidth": 0.85}
        if selected_notes and ann["label"]:
            label_positions.append({
                "x": min(n["start"] for n in selected_notes),
                "y": max(n["pitch"] for n in selected_notes) + 1.55,
                "label": ann["label"],
                "color": ann["color"],
            })

    for ann in BASS_ARPEGGIO_ANNOTATIONS:
        ids = bass_note_ids(notes, ann["start"], ann["end"], ann.get("pitch_max", 67))
        selected_notes = [notes[i] for i in ids]
        color = ann.get("color", BASS_COLOR)
        for i in ids:
            note_styles.setdefault(i, {"color": color, "height": 0.86, "linewidth": 0.65})
        if selected_notes and ann["label"]:
            label_positions.append({
                "x": min(n["start"] for n in selected_notes),
                "y": min(max(n["pitch"] for n in selected_notes) + 1.1, 67.0),
                "label": ann["label"],
                "color": color,
            })

    return note_styles, label_positions


def spread_labels(label_positions, min_x_gap=4.2, min_y_gap=2.2):
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


def draw_bracket(ax, label, start, end, y, color, zorder=7):
    ax.hlines(y=y, xmin=start, xmax=end, color=color, linewidth=1.9, alpha=0.95, zorder=zorder)
    ax.vlines(x=[start, end], ymin=y - 1.15, ymax=y + 1.15, color=color, linewidth=1.9, alpha=0.95, zorder=zorder)
    ax.text(
        (start + end) / 2,
        y + 1.45,
        label,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=color,
        zorder=zorder + 1,
    )


def plot_from_midi(midi_path: Path, out_path: Path):
    notes = parse_midi(midi_path)
    if not notes:
        raise RuntimeError("No se encontraron notas en el MIDI.")

    note_styles, label_positions = build_note_annotations(notes)
    min_pitch = min(n["pitch"] for n in notes) - 5
    max_pitch = max(n["pitch"] for n in notes) + 9
    end_time = max(n["end"] for n in notes)

    fig, ax = plt.subplots(figsize=(18, 7.2))

    for sec in SECTIONS:
        start = sec["start"]
        end = bounded_end(sec, end_time)
        if start >= end:
            continue
        ax.axvspan(start, end, color=sec["color"], alpha=0.48, linewidth=0)
        ax.text(
            (start + end) / 2,
            max_pitch - 2.0,
            sec["label"],
            ha="center",
            va="top",
            fontsize=12.2,
            fontweight="bold",
            color="#222222",
        )

    for bracket in ANALYTIC_BRACKETS:
        start = bracket["start"]
        end = bounded_end(bracket, end_time)
        if start < end:
            draw_bracket(ax, bracket["label"], start, end, max_pitch - 6.4, bracket["color"])

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
            linewidth = 0.12
            height = 0.72

        ax.add_patch(
            patches.Rectangle(
                (note["start"], note["pitch"] - height / 2),
                max(note["duration"], 0.02),
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
            fontsize=10.2,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    legend_handles = [
        patches.Patch(color=MOTIVE_COLOR, label="motivo original y repeticiones"),
        patches.Patch(color=SEQUENCE_COLOR, label="subidas/bajadas en B"),
        patches.Patch(color=BASS_COLOR, label="bajo arpegiado"),
        patches.Patch(color=CADENTIAL_BASS_COLOR, label="bajo arpegiado cadencial"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, framealpha=0.9, fontsize=9.5)

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
    elif SANDBOX_MIDI_PATH.exists():
        plot_from_midi(SANDBOX_MIDI_PATH, SANDBOX_OUT_PATH)
    else:
        raise FileNotFoundError("No se encontro el MIDI de entrada.")


if __name__ == "__main__":
    plot()
