from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd

from symusic import Score


# =========================
# CONFIG
# =========================
MAESTRO_MIDI_DIR = Path(r"data/raw/maestro-v3.0.0")      # <-- AJUSTA a tus .mid/.midi
LAKH_MIDI_DIR    = Path(r"data/raw/clean_midi")     # <-- AJUSTA a tus .mid/.midi
OUT_CSV = Path(r"data/interim/index_piano_midis.csv")

INCLUDE_LAKH = True

# Piano family en General MIDI: 0..7
ALLOWED_PROGRAMS = set(range(0, 8))

# Consideramos “piano-only” si no hay drums y todos los programas son piano.
# Si el MIDI no tiene program changes, puede venir como "0" o "None" según parser;
# en ese caso lo aceptamos salvo evidencia de otros instrumentos.
MAX_NONDRUM_TRACKS = 4  # opcional: limita multitrack
# =========================


@dataclass
class MidiCheck:
    ok: bool
    reason: str
    n_tracks: int
    n_non_drum_tracks: int
    programs: List[int]
    has_drums: bool


def collect_midis(root: Path) -> List[Path]:
    return sorted(
        [p for p in root.rglob("*.mid") if p.is_file()] +
        [p for p in root.rglob("*.midi") if p.is_file()]
    )


def analyze_midi(path: Path) -> MidiCheck:
    try:
        score = Score(path)
    except Exception as e:
        return MidiCheck(False, f"parse_error:{e}", 0, 0, [], False)

    tracks = score.tracks
    n_tracks = len(tracks)

    has_drums = False
    programs = []
    n_non_drum = 0

    for tr in tracks:
        # symusic suele exponer tr.is_drum (si no existiera en tu versión, te doy fallback)
        is_drum = getattr(tr, "is_drum", False)
        if is_drum:
            # si hay percusión -> fuera
            has_drums = True
            continue

        n_non_drum += 1

        # program puede ser None en algunos midis sin program change
        prog = getattr(tr, "program", None)
        if prog is not None:
            programs.append(int(prog))

    if has_drums:
        return MidiCheck(False, "has_drums", n_tracks, n_non_drum, sorted(set(programs)), True)

    if n_non_drum == 0:
        return MidiCheck(False, "no_non_drum_tracks", n_tracks, n_non_drum, sorted(set(programs)), False)

    if n_non_drum > MAX_NONDRUM_TRACKS:
        return MidiCheck(False, f"too_many_tracks:{n_non_drum}", n_tracks, n_non_drum, sorted(set(programs)), False)

    # Si no hay programas explícitos, asumimos piano (muchos midis lo hacen)
    if len(programs) == 0:
        return MidiCheck(True, "ok_no_programs", n_tracks, n_non_drum, [], False)

    # Si todos los programas están en familia piano -> ok
    bad = [p for p in programs if p not in ALLOWED_PROGRAMS]
    if bad:
        return MidiCheck(False, f"non_piano_programs:{sorted(set(bad))}", n_tracks, n_non_drum, sorted(set(programs)), False)

    return MidiCheck(True, "ok", n_tracks, n_non_drum, sorted(set(programs)), False)


def main():
    rows = []

    maestro = collect_midis(MAESTRO_MIDI_DIR)
    print(f"[MAESTRO] found={len(maestro)}")
    kept_m = 0
    for p in maestro:
        chk = analyze_midi(p)
        if chk.ok:
            kept_m += 1
            rows.append({
                "path": str(p.resolve()),
                "dataset": "maestro",
                "reason": chk.reason,
                "n_tracks": chk.n_tracks,
                "n_non_drum_tracks": chk.n_non_drum_tracks,
                "programs": ";".join(map(str, chk.programs)),
                "has_drums": int(chk.has_drums),
            })
    print(f"[MAESTRO] kept={kept_m}")

    if INCLUDE_LAKH:
        lakh = collect_midis(LAKH_MIDI_DIR)
        print(f"[LAKH] found={len(lakh)}")
        kept_l = 0
        for p in lakh:
            chk = analyze_midi(p)
            if chk.ok:
                kept_l += 1
                rows.append({
                    "path": str(p.resolve()),
                    "dataset": "lakh_piano",
                    "reason": chk.reason,
                    "n_tracks": chk.n_tracks,
                    "n_non_drum_tracks": chk.n_non_drum_tracks,
                    "programs": ";".join(map(str, chk.programs)),
                    "has_drums": int(chk.has_drums),
                })
        print(f"[LAKH] kept={kept_l}")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"[DONE] saved {OUT_CSV} total={len(df)}")


if __name__ == "__main__":
    main()