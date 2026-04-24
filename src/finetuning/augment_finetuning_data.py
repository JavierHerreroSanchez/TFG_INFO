"""
Aumentación OFFLINE al estilo Music Transformer (sin sobrerrepresentar obras)

Idea
----
En Music Transformer se aplica, por ejemplo en Piano-e-Competition:
  - Transposición uniforme en semitonos: {-3,-2,-1,0,1,2,3}
  - Time-stretch uniforme: {0.95, 0.975, 1.0, 1.025, 1.05}

En vez de generar el producto cartesiano completo (35 variantes por obra),
este script genera K variantes *muestreadas* por obra (K pequeño), para
mantener el espíritu del paper pero evitando sobrerrepresentación.

Salida
------
- OUT_AUG_DIR: MIDIs aumentados
- OUT_AUG_INDEX_CSV: índice con columna 'path' apuntando a los MIDIs aumentados
- OUT_AUG_REPORT_CSV: reporte con OK / SKIP / FAIL

"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from miditoolkit import MidiFile, Instrument, Note, TempoChange, ControlChange, PitchBend


import re

def sanitize_stem(name: str) -> str:
    """
    Convierte un nombre a uno "filesystem-friendly":
    - quita caracteres problemáticos,
    - comprime espacios,
    - reemplaza comas por nada y espacios por guiones bajos.
    """
    s = name
    s = s.replace(",", "")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace(" ", "_")
    # conserva letras, números, '_', '-', '.'
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
    return s


# =============================================================================
# RUTAS
# =============================================================================
IN_CLEAN_DIR = Path(r"../../data/finetuning_v2/finetuning_sonatas_clean")
OUT_AUG_DIR = Path(r"../../data/finetuning_v2/finetuning_sonatas_aug")
OUT_AUG_INDEX_CSV = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\output\finetuning_v2\finetuning_aug_index.csv")
OUT_AUG_REPORT_CSV = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\output\finetuning_v2\finetuning_aug_report.csv")

# Mantener estructura de carpetas dentro de OUT_AUG_DIR
PRESERVE_TREE = True
COMMON_ROOT = IN_CLEAN_DIR

# =============================================================================
# PARÁMETROS (paper-style, pero muestreado)
# =============================================================================
TRANSPOSE_SHIFTS = [-3, -2, -1, 0, 1, 2, 3]
TIME_STRETCH_FACTORS = [0.95, 0.975, 1.0, 1.025, 1.05]

# Nº de variantes por obra (recomendación práctica: 4–8)
K_VARIANTS_PER_FILE = 6

# Si True, siempre incluye (transpose=0, stretch=1.0) como una de las variantes.
# Si ya está en el muestreo, no se duplica.
INCLUDE_ORIGINAL = True

# Muestreo sin reemplazo sobre el conjunto de pares válidos (recomendado).
# Si False, permite repetir combinaciones (normalmente no interesa).
SAMPLE_WITHOUT_REPLACEMENT = True

# Semilla para reproducibilidad
SEED = 1453

# Seguridad / IO
DRY_RUN = False
CONTINUE_ON_FAILURE = True
CLEAR_OUT_DIR_ON_START = False

# Rango piano (A0..C8). Si la transposición sale de rango, se descarta esa combinación.
PITCH_MIN = 21
PITCH_MAX = 108


# =============================================================================
# Helpers básicos
# =============================================================================
def _ls(obj, attr: str):
    x = getattr(obj, attr, None)
    return x if x is not None else []


def list_midis_recursively(root: Path) -> List[Path]:
    exts = {".mid", ".midi"}
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def fmt_transpose(s: int) -> str:
    if s == 0:
        return "tr0"
    sign = "+" if s > 0 else ""
    return f"tr{sign}{s}"


def fmt_stretch(f: float) -> str:
    # 1.025 -> "ts1p025"
    return "ts" + f"{f:.3f}".replace(".", "p")


def make_out_path(src: Path, transpose: int, stretch: float) -> Path:
    if PRESERVE_TREE:
        try:
            rel = src.relative_to(COMMON_ROOT)
        except Exception:
            rel = Path(src.name)
        base = OUT_AUG_DIR / rel
    else:
        base = OUT_AUG_DIR / src.name

    stem = base.with_suffix("").name
    stem = sanitize_stem(stem)
    fname = f"{stem}__{fmt_transpose(transpose)}__{fmt_stretch(stretch)}.mid"
    return base.parent / fname


# =============================================================================
# Escritura robusta (misma idea que en tu limpieza)
# =============================================================================
def sanitize_miditoolkit(m: MidiFile) -> None:
    # tempo
    m.tempo_changes = [tc for tc in _ls(m, "tempo_changes") if tc.time >= 0]
    m.tempo_changes.sort(key=lambda tc: tc.time)
    if not m.tempo_changes:
        m.tempo_changes = [TempoChange(120.0, 0)]

    # metas opcionales
    for attr in ["time_signature_changes", "key_signature_changes", "markers", "lyrics"]:
        lst = getattr(m, attr, None)
        if lst is None:
            continue
        lst = [x for x in lst if getattr(x, "time", 0) >= 0]
        lst.sort(key=lambda x: x.time)
        setattr(m, attr, lst)

    # instrumentos
    new_insts = []
    for inst in _ls(m, "instruments"):
        inst.notes = [n for n in _ls(inst, "notes") if n.start >= 0 and n.end > n.start]
        inst.notes.sort(key=lambda n: (n.start, n.end, n.pitch, n.velocity))

        inst.control_changes = [cc for cc in _ls(inst, "control_changes") if cc.time >= 0]
        inst.control_changes.sort(key=lambda cc: cc.time)

        inst.pitch_bends = [pb for pb in _ls(inst, "pitch_bends") if pb.time >= 0]
        inst.pitch_bends.sort(key=lambda pb: pb.time)

        if inst.notes or inst.control_changes or inst.pitch_bends:
            new_insts.append(inst)

    m.instruments = new_insts


def rebuild_miditoolkit(m: MidiFile) -> MidiFile:
    """Copia profunda segura."""
    m2 = MidiFile()
    m2.ticks_per_beat = m.ticks_per_beat

    tcs = [TempoChange(tc.tempo, max(0, int(tc.time))) for tc in _ls(m, "tempo_changes")]
    if not tcs:
        tcs = [TempoChange(120.0, 0)]
    tcs.sort(key=lambda tc: tc.time)
    m2.tempo_changes = tcs

    for attr in ["time_signature_changes", "key_signature_changes", "markers", "lyrics"]:
        lst = getattr(m, attr, None)
        if lst is None:
            continue
        clean = [x for x in lst if getattr(x, "time", 0) >= 0]
        clean.sort(key=lambda x: x.time)
        setattr(m2, attr, clean)

    m2.instruments = []
    for inst in _ls(m, "instruments"):
        inst2 = Instrument(program=inst.program, is_drum=inst.is_drum, name=getattr(inst, "name", ""))

        inst2.notes = [
            Note(int(n.velocity), int(n.pitch), int(n.start), int(n.end))
            for n in _ls(inst, "notes")
            if n.start >= 0 and n.end > n.start
        ]
        inst2.notes.sort(key=lambda n: (n.start, n.end, n.pitch, n.velocity))

        inst2.control_changes = [
            ControlChange(int(cc.number), int(cc.value), int(cc.time))
            for cc in _ls(inst, "control_changes")
            if cc.time >= 0
        ]
        inst2.control_changes.sort(key=lambda cc: cc.time)

        inst2.pitch_bends = [
            PitchBend(int(pb.pitch), int(pb.time))
            for pb in _ls(inst, "pitch_bends")
            if pb.time >= 0
        ]
        inst2.pitch_bends.sort(key=lambda pb: pb.time)

        if inst2.notes or inst2.control_changes or inst2.pitch_bends:
            m2.instruments.append(inst2)

    return m2


def safe_dump_miditoolkit(m: MidiFile, out_path: Path, debug_tag: str = "") -> bool:
    try:
        sanitize_miditoolkit(m)
        m.dump(str(out_path))
        return True
    except ValueError as e:
        try:
            m2 = rebuild_miditoolkit(m)
            sanitize_miditoolkit(m2)
            m2.dump(str(out_path))
            return True
        except Exception as e2:
            print(f"[DUMP][FAIL] {debug_tag} -> {out_path} | {type(e).__name__}: {e} | retry_err={type(e2).__name__}: {e2}")
            return False
    except Exception as e:
        print(f"[DUMP][FAIL] {debug_tag} -> {out_path} | {type(e).__name__}: {e}")
        return False


# =============================================================================
# Augmentations
# =============================================================================
def piano_pitch_range(m: MidiFile) -> Optional[Tuple[int, int]]:
    pitches: List[int] = []
    for inst in _ls(m, "instruments"):
        if getattr(inst, "is_drum", False):
            continue
        for n in _ls(inst, "notes"):
            pitches.append(int(n.pitch))
    if not pitches:
        return None
    return min(pitches), max(pitches)


def transpose_inplace(m: MidiFile, semitones: int) -> bool:
    """Transpone. Devuelve False si se sale del rango del piano."""
    if semitones == 0:
        return True

    pr = piano_pitch_range(m)
    if pr is None:
        return False

    mn, mx = pr
    if mn + semitones < PITCH_MIN or mx + semitones > PITCH_MAX:
        return False

    for inst in _ls(m, "instruments"):
        if getattr(inst, "is_drum", False):
            continue
        for n in _ls(inst, "notes"):
            n.pitch = int(n.pitch) + semitones
    return True


def _scale_time_int(x: int, factor: float) -> int:
    return int(round(x * factor))


def time_stretch_inplace(m: MidiFile, factor: float) -> None:
    """Escala tiempos (ticks) de notas, CC, PB, tempo y metadatos."""
    if abs(factor - 1.0) < 1e-9:
        return

    for inst in _ls(m, "instruments"):
        for n in _ls(inst, "notes"):
            n.start = _scale_time_int(int(n.start), factor)
            n.end = _scale_time_int(int(n.end), factor)
            if n.end <= n.start:
                n.end = n.start + 1
        for cc in _ls(inst, "control_changes"):
            cc.time = _scale_time_int(int(cc.time), factor)
        for pb in _ls(inst, "pitch_bends"):
            pb.time = _scale_time_int(int(pb.time), factor)

    for tc in _ls(m, "tempo_changes"):
        tc.time = _scale_time_int(int(tc.time), factor)

    for attr in ["time_signature_changes", "key_signature_changes", "markers", "lyrics"]:
        lst = getattr(m, attr, None)
        if lst is None:
            continue
        for x in lst:
            x.time = _scale_time_int(int(x.time), factor)


# =============================================================================
# Muestreo de combinaciones (paper-style)
# =============================================================================
def valid_pairs_for_file(m: MidiFile) -> List[Tuple[int, float]]:
    """Todas las parejas (transpose, stretch) que respetan el rango piano."""
    pr = piano_pitch_range(m)
    if pr is None:
        return []
    mn, mx = pr

    valid_t = []
    for t in TRANSPOSE_SHIFTS:
        if mn + t >= PITCH_MIN and mx + t <= PITCH_MAX:
            valid_t.append(t)

    return [(t, s) for t in valid_t for s in TIME_STRETCH_FACTORS]


def sample_pairs(all_pairs: List[Tuple[int, float]], rng: random.Random) -> List[Tuple[int, float]]:
    if not all_pairs:
        return []

    pairs = list(all_pairs)

    # fuerza incluir el original si procede
    original = (0, 1.0)
    chosen: List[Tuple[int, float]] = []
    if INCLUDE_ORIGINAL and original in pairs:
        chosen.append(original)
        pairs.remove(original)

    remaining_k = max(0, K_VARIANTS_PER_FILE - len(chosen))
    if remaining_k == 0:
        return chosen

    if SAMPLE_WITHOUT_REPLACEMENT:
        # sin reemplazo
        if remaining_k >= len(pairs):
            chosen.extend(pairs)
        else:
            chosen.extend(rng.sample(pairs, k=remaining_k))
    else:
        # con reemplazo
        for _ in range(remaining_k):
            chosen.append(rng.choice(pairs))

    return chosen


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    rng = random.Random(SEED)

    if not IN_CLEAN_DIR.exists():
        raise FileNotFoundError(f"No existe IN_CLEAN_DIR: {IN_CLEAN_DIR}")

    if CLEAR_OUT_DIR_ON_START and OUT_AUG_DIR.exists():
        import shutil
        shutil.rmtree(OUT_AUG_DIR)

    OUT_AUG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_AUG_INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_AUG_REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)

    src_midis = list_midis_recursively(IN_CLEAN_DIR)
    print(f"[INFO] in_clean={IN_CLEAN_DIR} files={len(src_midis)}")
    print(f"[INFO] out_aug={OUT_AUG_DIR}")
    print(f"[INFO] K={K_VARIANTS_PER_FILE} include_original={INCLUDE_ORIGINAL} seed={SEED} no_repl={SAMPLE_WITHOUT_REPLACEMENT}")

    report_rows: List[Dict] = []
    kept_rows: List[Dict] = []

    for i, src in enumerate(src_midis, start=1):
        try:
            base = MidiFile(str(src))
        except Exception as e:
            report_rows.append({
                "path_original": str(src),
                "status": "READ_FAIL",
                "transpose": None,
                "stretch": None,
                "path_aug": "",
                "detail": f"{type(e).__name__}",
            })
            if not CONTINUE_ON_FAILURE:
                raise
            continue

        sanitize_miditoolkit(base)

        all_pairs = valid_pairs_for_file(base)
        chosen_pairs = sample_pairs(all_pairs, rng)

        if not chosen_pairs:
            report_rows.append({
                "path_original": str(src),
                "status": "SKIP_NO_VALID_PAIR",
                "transpose": None,
                "stretch": None,
                "path_aug": "",
                "detail": "",
            })
            continue

        for t, s in chosen_pairs:
            aug = rebuild_miditoolkit(base)

            if not transpose_inplace(aug, t):
                report_rows.append({
                    "path_original": str(src),
                    "status": "SKIP_PITCH_RANGE",
                    "transpose": t,
                    "stretch": s,
                    "path_aug": "",
                    "detail": "",
                })
                continue

            time_stretch_inplace(aug, s)
            sanitize_miditoolkit(aug)

            dst = make_out_path(src, t, s)
            dst.parent.mkdir(parents=True, exist_ok=True)

            ok = True
            if not DRY_RUN:
                ok = safe_dump_miditoolkit(aug, dst, debug_tag=f"{src} tr={t} ts={s}")

            if ok:
                kept_rows.append({
                    "path": str(dst),
                    "path_original": str(src),
                    "transpose": t,
                    "stretch": s,
                })
                report_rows.append({
                    "path_original": str(src),
                    "status": "OK",
                    "transpose": t,
                    "stretch": s,
                    "path_aug": str(dst),
                    "detail": "",
                })
            else:
                report_rows.append({
                    "path_original": str(src),
                    "status": "DUMP_FAIL",
                    "transpose": t,
                    "stretch": s,
                    "path_aug": "",
                    "detail": "",
                })
                if not CONTINUE_ON_FAILURE:
                    raise RuntimeError(f"DUMP_FAIL: {src}")

        if i % 200 == 0:
            print(f"[PROC] {i}/{len(src_midis)}")

    pd.DataFrame(report_rows).to_csv(OUT_AUG_REPORT_CSV, index=False)

    kept_df = pd.DataFrame(kept_rows)
    if len(kept_df) == 0:
        kept_df = pd.DataFrame(columns=["path", "path_original", "transpose", "stretch"])
    kept_df.to_csv(OUT_AUG_INDEX_CSV, index=False)

    print(f"[OK] report -> {OUT_AUG_REPORT_CSV} rows={len(report_rows)}")
    print(f"[OK] index  -> {OUT_AUG_INDEX_CSV} rows={len(kept_df)}")
    print(f"[OK] aug dir-> {OUT_AUG_DIR}")


if __name__ == "__main__":
    main()
