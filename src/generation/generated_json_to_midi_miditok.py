
from pathlib import Path
import json

from miditok import REMI, TokSequence


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

# Tokenizer entrenado/guardado con MidiTok
TOKENIZER_PATH = Path(r"../../tokenizer/tokenizer_REMI_BPE_v3.json")

# JSON generado por evaluation.py
#GENERATED_JSON_PATH = Path(r"../../output/evaluation/best_test/sample_004.json")
GENERATED_JSON_PATH = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\output\evaluation\best_val\sample_008.json")

# Campo del JSON a convertir
# Opciones típicas:
#   - "full_generated_tokens" -> prompt + continuación
#   - "generated_tokens" -> solo lo generado
#   - "prompt_tokens" -> solo el prompt
JSON_TOKEN_FIELD = "full_generated_tokens"

# MIDI de salida
OUTPUT_MIDI_PATH = Path(r"../../output/generation_v3/generated_from_json8.mid")

# Si True, elimina PAD / BOS / MASK y un EOS final si aparece
FILTER_SPECIAL_TOKENS = True

# Si True, además imprime una vista previa de los primeros tokens básicos
PRINT_DECODED_TOKEN_PREVIEW = True
PREVIEW_LEN = 80

# =============================================================================
# UTILIDADES
# =============================================================================

def load_generated_ids(json_path: Path, token_field: str) -> list[int]:
    if not json_path.exists():
        raise FileNotFoundError(f"No existe el JSON generado: {json_path.resolve()}")

    obj = json.loads(json_path.read_text(encoding="utf-8"))

    if token_field not in obj:
        raise KeyError(
            f"El campo '{token_field}' no existe en {json_path.name}. "
            f"Campos disponibles: {list(obj.keys())}"
        )

    ids = obj[token_field]
    if not isinstance(ids, list) or len(ids) == 0:
        raise ValueError(f"El campo '{token_field}' está vacío o no es una lista válida.")

    return [int(x) for x in ids]


def load_tokenizer(tokenizer_path: Path) -> REMI:
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"No existe el tokenizer: {tokenizer_path.resolve()}")

    # MidiTok permite cargar un tokenizer guardado mediante el argumento params
    tokenizer = REMI(params=tokenizer_path)
    return tokenizer


def get_special_token_ids(tokenizer: REMI) -> set[int]:
    """
    Intenta obtener ids de tokens especiales de forma robusta.
    """
    special_ids = set()

    for token_name in ["PAD_None", "BOS_None", "EOS_None", "MASK_None"]:
        try:
            token_id = tokenizer[token_name]
            if isinstance(token_id, int):
                special_ids.add(token_id)
        except Exception:
            pass

    return special_ids


def clean_token_ids(token_ids: list[int], tokenizer: REMI) -> list[int]:
    """
    Elimina tokens especiales que podrían molestar a la decodificación.
    Mantiene el cuerpo musical y, si hay un EOS al final, lo quita.
    """
    if not FILTER_SPECIAL_TOKENS:
        return token_ids

    special_ids = get_special_token_ids(tokenizer)

    # Quitamos PAD / BOS / MASK siempre
    cleaned = []
    eos_id = None
    try:
        eos_id = tokenizer["EOS_None"]
    except Exception:
        eos_id = None

    for tid in token_ids:
        if tid in special_ids and tid != eos_id:
            continue
        cleaned.append(tid)

    # Si acaba en EOS, lo retiramos
    if eos_id is not None and len(cleaned) > 0 and cleaned[-1] == eos_id:
        cleaned = cleaned[:-1]

    return cleaned


def preview_decoded_tokens(tokenizer: REMI, token_ids: list[int], preview_len: int = 80):
    """
    Muestra una vista previa de los tokens básicos tras deshacer BPE.
    Esto ayuda a comprobar visualmente que el JSON/tokenizer encajan.
    """
    seq = TokSequence(ids=token_ids[:preview_len], are_ids_encoded=True)
    tokenizer.decode_token_ids(seq)
    tokenizer.complete_sequence(seq)

    print("\n" + "=" * 90)
    print("VISTA PREVIA DE TOKENS BÁSICOS DECODIFICADOS")
    print("=" * 90)
    print(seq.tokens[:preview_len])


def decode_json_to_midi(
    tokenizer,
    token_ids: list[int],
    output_midi_path,
):
    """
    Convierte ids -> symusic.Score mediante MidiTok.decode(...)
    y guarda el MIDI asegurando que la ruta final sea un archivo .mid.
    """
    output_midi_path = Path(output_midi_path)

    # Si te pasan una carpeta o una ruta sin sufijo, añadimos nombre de archivo
    if output_midi_path.suffix.lower() != ".mid":
        output_midi_path = output_midi_path / "generated_from_json.mid"

    output_midi_path = output_midi_path.resolve()
    output_midi_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[DEBUG] Ruta final de salida MIDI: {output_midi_path}")

    seq = TokSequence(ids=token_ids, are_ids_encoded=True)

    # Decodificar a Score
    score = tokenizer.decode(seq)

    # Guardar manualmente
    score.dump_midi(str(output_midi_path))

    return score


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 90)
    print("JSON -> MIDI CON MIDITOK")
    print("=" * 90)

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    print(f"[OK] Tokenizer cargado: {TOKENIZER_PATH.resolve()}")
    print(f"[INFO] Tokenizer class      : {tokenizer.__class__.__name__}")
    print(f"[INFO] Vocab size          : {len(tokenizer)}")
    print(f"[INFO] one_token_stream    : {tokenizer.one_token_stream}")
    print(f"[INFO] is_multi_voc        : {tokenizer.is_multi_voc}")

    token_ids = load_generated_ids(GENERATED_JSON_PATH, JSON_TOKEN_FIELD)
    print(f"[OK] JSON cargado         : {GENERATED_JSON_PATH.resolve()}")
    print(f"[INFO] Campo usado         : {JSON_TOKEN_FIELD}")
    print(f"[INFO] Nº ids originales   : {len(token_ids)}")

    token_ids = clean_token_ids(token_ids, tokenizer)
    print(f"[INFO] Nº ids tras limpieza: {len(token_ids)}")

    if PRINT_DECODED_TOKEN_PREVIEW:
        try:
            preview_decoded_tokens(tokenizer, token_ids, preview_len=PREVIEW_LEN)
        except Exception as e:
            print(f"[WARN] No se pudo generar la vista previa de tokens: {e}")

    try:
        score = decode_json_to_midi(
            tokenizer=tokenizer,
            token_ids=token_ids,
            output_midi_path=OUTPUT_MIDI_PATH,
        )
    except Exception as e:
        print("\n[ERROR] Falló la conversión JSON -> MIDI.")
        print("Pistas de depuración:")
        print("  - Comprueba que GENERATED_JSON_PATH apunta a un JSON generado con ESTE tokenizer.")
        print("  - Comprueba si debes usar 'full_generated_tokens' o 'generated_tokens'.")
        print("  - Si el JSON ya estuviera en ids básicos y no BPE, cambia are_ids_encoded=True a False.")
        raise

    print("\n" + "=" * 90)
    print("CONVERSIÓN COMPLETADA")
    print("=" * 90)
    print(f"[OK] MIDI guardado en: {OUTPUT_MIDI_PATH.resolve()}")

    # info básica del score
    try:
        num_tracks = len(score.tracks)
    except Exception:
        num_tracks = "N/A"

    print(f"[INFO] Nº tracks en score: {num_tracks}")


if __name__ == "__main__":
    main()
