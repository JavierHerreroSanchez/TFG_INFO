from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from miditok import REMI


TOKENIZER_PATH = Path(r"/tokenizer/tokenizer_REMI_BPE_v4.json")
JSON_DIR = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\data\interim\tokenized_json_bpe_v2")

MAX_FILES = 40000

# Si quieres limitar la salida a ciertas familias, rellena esta lista.
# Ejemplo: ["Chord", "Rest", "TimeSig", "BarPitchClass"]
ONLY_FAMILIES: list[str] | None = None

TOP_FAMILIES_TO_SHOW = 30
TOP_TOKENS_PER_FAMILY = 25


def token_family(token_str: str) -> str:
    """
    Devuelve la familia del token tomando la parte previa al primer "_".
    Ejemplos:
      Pitch_60 -> Pitch
      Bar_None -> Bar
      Chord_C:maj -> Chord
      TimeSig_4/4 -> TimeSig
    """
    if "_" not in token_str:
        return token_str
    return token_str.split("_", 1)[0]


def chord_subfamily(token_str: str) -> str:
    """
    Intenta extraer la 'calidad' del acorde de tokens como:
      Chord_C:maj
      Chord_G:maj8
      Chord_D:maj_open
    """
    if not token_str.startswith("Chord_"):
        return ""
    body = token_str[len("Chord_"):]
    if ":" in body:
        return body.split(":", 1)[1]
    return body


def _decode_one_sequence(tokenizer: REMI, seq) -> list[str]:
    if getattr(seq, "are_ids_encoded", False):
        tokenizer.decode_token_ids(seq)
    tokenizer.complete_sequence(seq, complete_bytes=False)
    return seq.tokens


def load_decoded_tokens(tokenizer: REMI, json_path: Path) -> list[str]:
    loaded = tokenizer.load_tokens(json_path)

    # Caso TokSequence único
    if hasattr(loaded, "ids"):
        return _decode_one_sequence(tokenizer, loaded)

    # Caso lista de TokSequence
    if isinstance(loaded, list):
        all_tokens = []
        for seq in loaded:
            all_tokens.extend(_decode_one_sequence(tokenizer, seq))
        return all_tokens

    raise TypeError(f"Formato inesperado devuelto por load_tokens: {type(loaded)}")


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
    tokens_by_family = defaultdict(Counter)
    chord_quality_counter = Counter()

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
            tokens_by_family[fam][tok] += 1
            local_families.add(fam)

            if fam == "Chord":
                chord_quality_counter[chord_subfamily(tok)] += 1

        for fam in local_families:
            files_with_family[fam] += 1

        if i % 1000 == 0:
            print(f"[INFO] procesados {i}/{len(json_files)} archivos...")

    print("\n" + "=" * 100)
    print("RESUMEN GLOBAL")
    print("=" * 100)
    print(f"Archivos procesados: {total_files}")
    print(f"Tokens totales:      {total_tokens:,}")

    print("\n" + "=" * 100)
    print("FAMILIAS DE TOKENS")
    print("=" * 100)
    for fam, count in family_counter.most_common():
        pct = 100 * count / max(total_tokens, 1)
        nfiles = files_with_family[fam]
        print(f"{fam:20s} | {count:12,d} | {pct:7.3f}% | en {nfiles:6d} archivos")

    families_to_show = [fam for fam, _ in family_counter.most_common(TOP_FAMILIES_TO_SHOW)]
    if ONLY_FAMILIES is not None:
        families_to_show = [fam for fam in families_to_show if fam in ONLY_FAMILIES]

    print("\n" + "=" * 100)
    print("TOP TOKENS POR FAMILIA")
    print("=" * 100)

    for fam in families_to_show:
        fam_total = family_counter[fam]
        print(f"\n[{fam}] total={fam_total:,} ({100 * fam_total / max(total_tokens, 1):.3f}%)")

        fam_tokens = tokens_by_family[fam].most_common(TOP_TOKENS_PER_FAMILY)
        if not fam_tokens:
            print("  (ninguno)")
            continue

        for tok, c in fam_tokens:
            pct_global = 100 * c / max(total_tokens, 1)
            pct_family = 100 * c / max(fam_total, 1)
            print(f"  {tok:40s} | {c:12,d} | global={pct_global:7.4f}% | familia={pct_family:7.3f}%")

    print("\n" + "=" * 100)
    print("DETALLE EXTRA DE CHORD")
    print("=" * 100)
    chord_total = family_counter.get("Chord", 0)
    if chord_total == 0:
        print("No aparecen tokens Chord.")
    else:
        print(f"Total Chord: {chord_total:,} ({100 * chord_total / max(total_tokens, 1):.4f}%)")
        print("\n[Calidades / subfamilias de acorde]")
        for quality, c in chord_quality_counter.most_common(40):
            pct_chord = 100 * c / max(chord_total, 1)
            print(f"  {quality:20s} | {c:10,d} | {pct_chord:7.3f}% de Chord")

    print("\n" + "=" * 100)
    print("DIAGNÓSTICO RÁPIDO")
    print("=" * 100)
    chord_pct = 100 * family_counter.get("Chord", 0) / max(total_tokens, 1)
    rest_pct = 100 * family_counter.get("Rest", 0) / max(total_tokens, 1)
    timesig_pct = 100 * family_counter.get("TimeSig", 0) / max(total_tokens, 1)

    print(f"- Chord:   {chord_pct:.4f}%")
    print(f"- Rest:    {rest_pct:.4f}%")
    print(f"- TimeSig: {timesig_pct:.4f}%")

    if chord_pct < 0.05:
        print("- Los Chord siguen siendo muy escasos: el problema probablemente sigue estando en la detección.")
    if rest_pct > 0:
        print("- Rest ya está entrando en la representación.")
    if timesig_pct > 0:
        print("- TimeSig ya está entrando en la representación.")


def main():
    analyze_folder(
        tokenizer_path=TOKENIZER_PATH,
        json_dir=JSON_DIR,
        max_files=MAX_FILES,
    )


if __name__ == "__main__":
    main()