from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from miditok import REMI


# =========================
# CONFIGURA ESTAS RUTAS
# =========================
TOKENIZER_PATH = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\tokenizer\tokenizer_REMI_BPE_v3.json")
JSON_DIR = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\data\interim\tokenized_json_bpe")

# Si quieres limitar el número de archivos para una prueba rápida
MAX_FILES = None   # por ejemplo: 20


def token_family(token_str: str) -> str:
    """
    Devuelve la 'familia' del token:
    'Chord_maj' -> 'Chord'
    'Pitch_60'  -> 'Pitch'
    'Bar_None'  -> 'Bar'
    """
    if "_" not in token_str:
        return token_str
    return token_str.split("_", 1)[0]


def load_decoded_tokens(tokenizer: REMI, json_path: Path) -> list[str]:
    """
    Carga un JSON tokenizado, decodifica BPE si hace falta
    y devuelve la lista de tokens base legibles.
    """
    seq = tokenizer.load_tokens(json_path)

    # Si los ids están codificados (BPE), los descomponemos
    if getattr(seq, "are_ids_encoded", False):
        tokenizer.decode_token_ids(seq)

    # Completa tokens a partir de ids
    tokenizer.complete_sequence(seq, complete_bytes=False)

    return seq.tokens


def analyze_folder(tokenizer_path: Path, json_dir: Path, max_files: int | None = None):
    tokenizer = REMI(params=tokenizer_path)

    json_files = sorted(json_dir.rglob("*.json"))
    if max_files is not None:
        json_files = json_files[:max_files]

    if not json_files:
        raise FileNotFoundError(f"No se encontraron JSON en: {json_dir}")

    family_counter = Counter()
    token_counter = Counter()
    files_with_family = Counter()
    per_file_presence = defaultdict(set)

    total_files = 0
    total_tokens = 0

    for i, path in enumerate(json_files, start=1):
        try:
            tokens = load_decoded_tokens(tokenizer, path)
        except Exception as e:
            print(f"[WARN] No se pudo procesar {path.name}: {e}")
            continue

        total_files += 1
        total_tokens += len(tokens)

        local_families = set()

        for tok in tokens:
            fam = token_family(tok)
            family_counter[fam] += 1
            token_counter[tok] += 1
            local_families.add(fam)

        for fam in local_families:
            files_with_family[fam] += 1

        if i % 25 == 0:
            print(f"[INFO] procesados {i}/{len(json_files)} archivos...")

    print("\n" + "=" * 100)
    print("RESUMEN GLOBAL")
    print("=" * 100)
    print(f"Archivos procesados: {total_files}")
    print(f"Tokens totales:      {total_tokens:,}")
    print()

    print("=" * 100)
    print("FAMILIAS DE TOKENS")
    print("=" * 100)
    for fam, count in family_counter.most_common():
        pct = 100 * count / max(total_tokens, 1)
        nfiles = files_with_family[fam]
        print(f"{fam:15s} | {count:12,d} | {pct:7.3f}% | en {nfiles:5d} archivos")

    print("\n" + "=" * 100)
    print("TOKENS MÁS FRECUENTES")
    print("=" * 100)
    for tok, count in token_counter.most_common(100):
        pct = 100 * count / max(total_tokens, 1)
        print(f"{tok:30s} | {count:12,d} | {pct:7.3f}%")

    # Resumen específico de lo que más te interesa
    interesting_families = [
        "Chord",
        "Program",
        "Bar",
        "Position",
        "Pitch",
        "Velocity",
        "Duration",
        "Rest",
        "TimeSig",
        "Tempo",
    ]

    print("\n" + "=" * 100)
    print("RESUMEN DE FAMILIAS CLAVE")
    print("=" * 100)
    for fam in interesting_families:
        count = family_counter.get(fam, 0)
        pct = 100 * count / max(total_tokens, 1)
        nfiles = files_with_family.get(fam, 0)
        print(f"{fam:15s} | {count:12,d} | {pct:7.3f}% | en {nfiles:5d} archivos")

    print("\n" + "=" * 100)
    print("TOP TOKENS POR FAMILIA")
    print("=" * 100)
    for fam in interesting_families:
        fam_tokens = [(tok, c) for tok, c in token_counter.items() if token_family(tok) == fam]
        fam_tokens.sort(key=lambda x: x[1], reverse=True)

        print(f"\n[{fam}]")
        if not fam_tokens:
            print("  (ninguno)")
            continue

        for tok, c in fam_tokens[:20]:
            pct = 100 * c / max(total_tokens, 1)
            print(f"  {tok:30s} | {c:10,d} | {pct:7.4f}%")

    print("\n" + "=" * 100)
    print("DIAGNÓSTICO RÁPIDO")
    print("=" * 100)

    chord_pct = 100 * family_counter.get("Chord", 0) / max(total_tokens, 1)
    program_pct = 100 * family_counter.get("Program", 0) / max(total_tokens, 1)

    if chord_pct == 0:
        print("- No aparecen tokens Chord en la muestra analizada.")
    else:
        print(f"- Los tokens Chord representan {chord_pct:.3f}% del total.")

    if program_pct > 5:
        print(f"- Program pesa bastante ({program_pct:.3f}%): puede estar metiendo ruido.")
    else:
        print(f"- Program no parece dominar ({program_pct:.3f}%).")


def main():
    analyze_folder(
        tokenizer_path=TOKENIZER_PATH,
        json_dir=JSON_DIR,
        max_files=MAX_FILES,
    )


if __name__ == "__main__":
    main()