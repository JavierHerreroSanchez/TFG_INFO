from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

MIDI_PATH = Path('../../output/generation_pretraining_tfg_second/generated_from_json7.mid')
OUT_PATH = Path('../musical_analisis/musical_analisis_pretraining_v2_generated_from_json9.png')

# Analisis ligero de pseudopieza jazz.
# Sin dividir en demasiadas partes: solo se marcan ideas/gestos relevantes.
ANNOTATIONS = [
    {"label": "Motivo inicial", "start": 1.80, "end": 6.05, "color": "#d62728", "mode": "top"},
    {"label": "Continuación", "start": 6.05, "end": 18.00, "color": "#1f77b4", "mode": "top"},

    # Pequeñas repeticiones / ecos motívicos en la zona central.
    {"label": "Rep.", "start": 18.70, "end": 21.00, "color": "#9467bd", "mode": "top"},
    {"label": "Rep.", "start": 21.30, "end": 23.90, "color": "#9467bd", "mode": "top"},
    {"label": "Rep.", "start": 24.40, "end": 27.10, "color": "#9467bd", "mode": "top"},
    {"label": "Rep.", "start": 27.10, "end": 30.60, "color": "#9467bd", "mode": "top"},

    # Ascenso armonico/registral especialmente logrado, coloreando la textura implicada.
    {"label": "Subida armónica", "start": 34.10, "end": 42.00, "color": "#ff7f0e", "mode": "band", "pitch_min": 36, "pitch_max": 84},
]

BACKGROUND_NOTE_COLOR = "#222222"
BACKGROUND_ALPHA = 0.23
ANNOTATED_ALPHA = 0.96
TOP_ONSET_TOL = 0.045


def read_varlen(data, i):
    value = 0
    while True:
        b = data[i]
        i += 1
        value = (value << 7) | (b & 0x7F)
        if not (b & 0x80):
            break
    return value, i


def parse_midi_notes(path):
    data = Path(path).read_bytes()
    i = 0
    if data[i:i+4] != b'MThd':
        raise ValueError('No parece un archivo MIDI valido.')
    i += 4
    header_len = int.from_bytes(data[i:i+4], 'big')
    i += 4
    fmt = int.from_bytes(data[i:i+2], 'big')
    n_tracks = int.from_bytes(data[i+2:i+4], 'big')
    ticks_per_beat = int.from_bytes(data[i+4:i+6], 'big')
    i += header_len

    raw_notes = []
    tempos = [(0, 500000)]  # microsegundos por negra, 120 BPM por defecto

    for track_idx in range(n_tracks):
        if data[i:i+4] != b'MTrk':
            raise ValueError('Chunk MTrk no encontrado.')
        i += 4
        track_len = int.from_bytes(data[i:i+4], 'big')
        i += 4
        end = i + track_len
        tick = 0
        running_status = None
        active = {}

        while i < end:
            delta, i = read_varlen(data, i)
            tick += delta

            status = data[i]
            if status < 0x80:
                if running_status is None:
                    raise ValueError('Running status inesperado.')
                status = running_status
            else:
                i += 1
                running_status = status

            if status == 0xFF:
                meta_type = data[i]
                i += 1
                length, i = read_varlen(data, i)
                payload = data[i:i+length]
                i += length
                if meta_type == 0x51 and length == 3:
                    tempos.append((tick, int.from_bytes(payload, 'big')))
                continue

            if status in (0xF0, 0xF7):
                length, i = read_varlen(data, i)
                i += length
                continue

            event_type = status & 0xF0
            channel = status & 0x0F

            if event_type in (0x80, 0x90):
                pitch = data[i]
                velocity = data[i+1]
                i += 2
                key = (channel, pitch)
                if event_type == 0x90 and velocity > 0:
                    active.setdefault(key, []).append((tick, velocity))
                else:
                    if key in active and active[key]:
                        start_tick, vel = active[key].pop(0)
                        if tick > start_tick:
                            raw_notes.append((start_tick, tick, pitch, vel, channel, track_idx))
            elif event_type in (0xA0, 0xB0, 0xE0):
                i += 2
            elif event_type in (0xC0, 0xD0):
                i += 1
            else:
                raise ValueError(f'Evento MIDI no soportado: {hex(status)}')

    tempos = sorted(set(tempos), key=lambda x: x[0])

    def tick_to_sec(t):
        sec = 0.0
        prev_tick = 0
        tempo = 500000
        for tempo_tick, tempo_value in tempos:
            if tempo_tick > t:
                break
            sec += (tempo_tick - prev_tick) * tempo / 1_000_000 / ticks_per_beat
            prev_tick = tempo_tick
            tempo = tempo_value
        sec += (t - prev_tick) * tempo / 1_000_000 / ticks_per_beat
        return sec

    notes = []
    for start_tick, end_tick, pitch, velocity, channel, track_idx in raw_notes:
        start = tick_to_sec(start_tick)
        end = tick_to_sec(end_tick)
        notes.append({
            'inst': track_idx,
            'pitch': int(pitch),
            'start': float(start),
            'end': float(end),
            'duration': float(end - start),
            'velocity': int(velocity),
        })

    notes.sort(key=lambda n: (n['start'], n['pitch'], n['end']))
    end_time = max((n['end'] for n in notes), default=0.0)
    return notes, end_time


def top_voice_note_ids(notes, start, end):
    candidates = [
        (i, n) for i, n in enumerate(notes)
        if start - 1e-6 <= n['start'] <= end + 1e-6
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
            if abs(note2['start'] - note['start']) <= TOP_ONSET_TOL:
                group.append((idx2, note2))
                used[b] = True
        top_idx, _ = max(group, key=lambda pair: pair[1]['pitch'])
        selected.add(top_idx)
    return selected


def band_note_ids(notes, start, end, pitch_min, pitch_max):
    return {
        i for i, n in enumerate(notes)
        if start - 1e-6 <= n['start'] <= end + 1e-6
        and pitch_min <= n['pitch'] <= pitch_max
    }


def build_note_annotations(notes):
    note_styles = {}
    label_positions = []

    for ann in ANNOTATIONS:
        mode = ann.get('mode')
        if mode == 'top':
            ids = top_voice_note_ids(notes, ann['start'], ann['end'])
        elif mode == 'band':
            ids = band_note_ids(notes, ann['start'], ann['end'], ann.get('pitch_min', 0), ann.get('pitch_max', 127))
        else:
            ids = {
                i for i, n in enumerate(notes)
                if ann['start'] <= n['start'] and n['end'] <= ann['end']
            }

        selected_notes = [notes[i] for i in ids]
        for i in ids:
            note_styles[i] = ann

        if selected_notes:
            x = min(n['start'] for n in selected_notes)
            y = max(n['pitch'] for n in selected_notes) + 1.6
            label_positions.append((x, y, ann['label'], ann['color']))

    return note_styles, label_positions


def draw_subsection_line(ax, label, start, end, y, color):
    ax.hlines(y=y, xmin=start, xmax=end, color=color, linewidth=1.8, alpha=0.95, zorder=6)
    ax.vlines(x=[start, end], ymin=y-1.2, ymax=y+1.2, color=color, linewidth=1.8, alpha=0.95, zorder=6)
    ax.text((start + end) / 2, y + 1.5, label, ha='center', va='bottom', fontsize=10.5,
            fontweight='bold', color=color, zorder=7)


def plot():
    notes, end_time = parse_midi_notes(MIDI_PATH)
    if not notes:
        raise RuntimeError('No se encontraron notas MIDI.')

    note_styles, label_positions = build_note_annotations(notes)
    min_pitch = min(n['pitch'] for n in notes) - 4
    max_pitch = max(n['pitch'] for n in notes) + 8

    fig, ax = plt.subplots(figsize=(18, 7))

    # Marcador lineal de la subida armónica, sin crear una sección formal extra.
    draw_subsection_line(ax, 'subida armónica', 34.10, 42.00, max_pitch - 7, '#ff7f0e')

    for i, n in enumerate(notes):
        ann = note_styles.get(i)
        if ann:
            face = ann['color']
            edge = ann['color']
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
                (n['start'], n['pitch'] - height / 2),
                max(n['duration'], 0.02),
                height,
                facecolor=face,
                edgecolor=edge,
                linewidth=lw,
                alpha=alpha,
            )
        )

    for x, y, label, color in label_positions:
        ax.text(x, y, label, color=color, fontsize=11, fontweight='bold', ha='left', va='bottom')

    ax.set_xlim(0, end_time)
    ax.set_ylim(min_pitch, max_pitch)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Altura MIDI')
    ax.set_title('Análisis ligero de pseudopieza jazz', fontsize=14)
    ax.grid(axis='y', alpha=0.18)
    ax.grid(axis='x', alpha=0.08)

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Guardado: {OUT_PATH}')


if __name__ == '__main__':
    plot()
