from pathlib import Path
import pretty_midi


# =========================
# CONFIGURACIÓN
# =========================
INPUT_DIR = Path(r"/data/finetuning_v3/mozart_sonatas_raw")
OUTPUT_DIR = Path(r"/data/finetuning_v2\mozart_sonatas_merged")

RECURSIVE = True  # True = busca también en subcarpetas
OVERWRITE = True  # False = salta archivos ya procesados


def merge_piano_tracks(in_path: Path, out_path: Path) -> bool:
    """
    Lee un MIDI, fusiona todas las pistas no-drum con program=0 en una sola pista de piano,
    y escribe un nuevo MIDI compatible con tokenización single-stream.
    """
    try:
        pm = pretty_midi.PrettyMIDI(str(in_path))
    except Exception as e:
        print(f"[ERROR] No se pudo leer: {in_path.name} | {e}")
        return False

    # Nuevo MIDI. Usamos el tempo estimado porque pretty_midi no copia directamente
    # todos los eventos de tempo en una API pública simple.
    try:
        initial_tempo = pm.estimate_tempo()
    except Exception:
        initial_tempo = 120.0

    merged = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo)

    # Copiar metadatos útiles si existen
    merged.time_signature_changes = list(pm.time_signature_changes)
    merged.key_signature_changes = list(pm.key_signature_changes)
    merged.lyrics = list(pm.lyrics)
    merged.text_events = list(pm.text_events)

    piano = pretty_midi.Instrument(
        program=0,
        is_drum=False,
        name="Merged Piano"
    )

    n_source_tracks = 0
    n_notes = 0

    for inst in pm.instruments:
        # Solo pistas de piano acústico program 0, no batería
        if not inst.is_drum and inst.program == 0:
            n_source_tracks += 1

            piano.notes.extend(inst.notes)
            piano.control_changes.extend(inst.control_changes)
            piano.pitch_bends.extend(inst.pitch_bends)

            n_notes += len(inst.notes)

    if n_notes == 0:
        print(f"[WARN] Sin notas de piano program=0: {in_path.name}")
        return False

    # Ordenar eventos
    piano.notes.sort(key=lambda n: (n.start, n.pitch, n.end))
    piano.control_changes.sort(key=lambda c: c.time)
    piano.pitch_bends.sort(key=lambda b: b.time)

    merged.instruments.append(piano)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not OVERWRITE:
        print(f"[SKIP] Ya existe: {out_path.name}")
        return True

    try:
        merged.write(str(out_path))
    except Exception as e:
        print(f"[ERROR] No se pudo escribir: {out_path.name} | {e}")
        return False

    print(
        f"[OK] {in_path.name} -> {out_path.name} | "
        f"tracks_piano={n_source_tracks} notes={n_notes}"
    )
    return True


def find_midi_files(root: Path, recursive: bool = True):
    patterns = ["*.mid", "*.midi"]
    files = []

    for pattern in patterns:
        if recursive:
            files.extend(root.rglob(pattern))
        else:
            files.extend(root.glob(pattern))

    return sorted(set(p for p in files if p.is_file()))


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"No existe INPUT_DIR: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    midi_files = find_midi_files(INPUT_DIR, RECURSIVE)

    print(f"[INFO] MIDIs encontrados: {len(midi_files)}")
    print(f"[INFO] Carpeta entrada: {INPUT_DIR}")
    print(f"[INFO] Carpeta salida:  {OUTPUT_DIR}")

    ok = 0
    fail = 0

    for in_path in midi_files:
        # Mantiene estructura de subcarpetas si RECURSIVE=True
        rel_path = in_path.relative_to(INPUT_DIR)
        out_path = OUTPUT_DIR / rel_path.with_name(rel_path.stem + "_merged_piano.mid")

        success = merge_piano_tracks(in_path, out_path)

        if success:
            ok += 1
        else:
            fail += 1

    print("\n========== RESUMEN ==========")
    print(f"Procesados OK: {ok}")
    print(f"Fallidos / sin piano: {fail}")
    print(f"Salida: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()