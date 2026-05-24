"""
Genera muestras musicales y las convierte entre formatos intermedios y MIDI.

El objetivo es transformar la salida autoregresiva del modelo en artefactos audibles y evaluables.
"""

from __future__ import annotations

"""
Generación completa desde el modelo de pretraining: JSON y MIDI.

Este script integra dos pasos que antes se ejecutaban por separado:
1) generación autorregresiva de muestras en JSON;
2) conversión inmediata de cada muestra generada a MIDI.

Se mantienen los mismos criterios de selección de split, prompt, temperatura y
top-k que en `generation_from_pretraining_v2.py`. La conversion MIDI utiliza el
mismo tokenizador REMI+BPE entrenado para el proyecto.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List

import torch

from src.generation.generation_from_pretraining_v2 import (
    ADD_EOS,
    CKPT_DIR,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MIN_NEW_TOKENS,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_PROMPT_LEN,
    DEFAULT_RANDOM_OFFSET,
    DEFAULT_STOP_ON_EOS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEVICE,
    EOS_ID,
    EVAL_BATCHES,
    OUTPUT_DIR,
    SEED,
    choose_prompt_from_song,
    evaluate_split_loss,
    generate_from_prompt,
    get_model_block_size,
    get_split_files,
    has_valid_prompt_start,
    load_checkpoint_and_model,
    load_token_ids_from_json,
    seed_all,
)
from src.generation.generated_json_to_midi_for_improved_pretraining import (
    clean_token_ids,
    decode_json_to_midi,
    load_tokenizer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "tokenizer_REMI_BPE_v5.json"

# =============================================================================
# CONFIGURACION PARA EJECUTAR DESDE PYCHARM
# -----------------------------------------------------------------------------
# Cambia estas variables y pulsa Run en PyCharm. No necesitas anadir argumentos.
# Si TOTAL_TOKENS no es None, manda sobre MAX_NEW_TOKENS:
#   tokens generados = TOTAL_TOKENS - INPUT_TOKENS
# =============================================================================

MODE = "generate"  # "generate", "loss" o "all"
CKPT_NAME = "best"  # "best" o "last"
SPLIT = "train"  # "train", "val" o "test"

INPUT_TOKENS = DEFAULT_PROMPT_LEN
TOTAL_TOKENS: int | None = None
MAX_NEW_TOKENS = DEFAULT_MAX_NEW_TOKENS
MIN_NEW_TOKENS = DEFAULT_MIN_NEW_TOKENS

NUM_SAMPLES = DEFAULT_NUM_SAMPLES
OUTPUT_DIR_OVERRIDE: Path | None = None

TEMPERATURE = DEFAULT_TEMPERATURE
TOP_K: int | None = DEFAULT_TOP_K
GREEDY = False
RANDOM_OFFSET = DEFAULT_RANDOM_OFFSET
STOP_ON_EOS = DEFAULT_STOP_ON_EOS

RUN_DEVICE = DEVICE
MAX_BATCHES = EVAL_BATCHES
TOKENIZER_PATH_OVERRIDE = TOKENIZER_PATH


def save_json_sample(
    out_dir: Path,
    sample_idx: int,
    split: str,
    ckpt_name: str,
    source_path: Path,
    prompt_start: int,
    prompt_tokens: List[int],
    gt_continuation: List[int],
    full_generated_tokens: List[int],
    hit_eos: bool,
    temperature: float,
    top_k: int | None,
    do_sample: bool,
) -> Path:
    """Guarda una muestra generada y devuelve la ruta del JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_only = full_generated_tokens[len(prompt_tokens):]

    payload = {
        "split": split,
        "checkpoint": ckpt_name,
        "source_json": str(source_path),
        "prompt_start": int(prompt_start),
        "prompt_len": int(len(prompt_tokens)),
        "generated_len": int(len(generated_only)),
        "hit_eos": bool(hit_eos),
        "sampling": {
            "temperature": float(temperature),
            "top_k": None if top_k is None else int(top_k),
            "do_sample": bool(do_sample),
        },
        "prompt_tokens": prompt_tokens,
        "ground_truth_continuation": gt_continuation,
        "generated_tokens": generated_only,
        "full_generated_tokens": full_generated_tokens,
    }

    out_path = out_dir / f"sample_{sample_idx:03d}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def convert_tokens_to_midi(
    tokenizer,
    token_ids: List[int],
    midi_path: Path,
) -> Path:
    """Convierte una lista de ids generados en un archivo MIDI."""
    cleaned_ids = clean_token_ids(token_ids, tokenizer)
    decode_json_to_midi(
        tokenizer=tokenizer,
        token_ids=cleaned_ids,
        output_midi_path=midi_path,
    )
    return midi_path


def run_generation_and_midi(
    model: torch.nn.Module,
    tokenizer,
    ckpt_name: str,
    split: str,
    prompt_len: int,
    max_new_tokens: int,
    min_new_tokens: int,
    num_samples: int,
    random_offset: bool,
    temperature: float,
    top_k: int | None,
    do_sample: bool,
    stop_on_eos: bool,
    device: str,
    out_dir: Path,
    tokenizer_path: Path,
) -> None:
    """Genera muestras, guarda JSONs y exporta MIDIs en una sola pasada."""
    files = get_split_files(split)

    valid_files = []
    for path in files:
        try:
            ids = load_token_ids_from_json(path)
            valid = has_valid_prompt_start(
                ids=ids,
                prompt_len=prompt_len,
                require_after_eos=random_offset,
                eos_token_id=EOS_ID if ADD_EOS else None,
            )
            if valid:
                valid_files.append(path)
            else:
                print(f"[WARN] saltando {path.name}: sin prompt válido")
        except Exception as exc:
            print(f"[WARN] saltando {path.name}: {exc}")

    if not valid_files:
        raise RuntimeError(f"No hay ficheros válidos en split={split}.")

    chosen = random.sample(valid_files, k=min(num_samples, len(valid_files)))
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "checkpoint": ckpt_name,
        "split": split,
        "prompt_len": int(prompt_len),
        "max_new_tokens": int(max_new_tokens),
        "min_new_tokens": int(min_new_tokens),
        "num_samples_requested": int(num_samples),
        "num_samples_generated": int(len(chosen)),
        "random_offset": bool(random_offset),
        "tokenizer_path": str(tokenizer_path.resolve()),
        "checkpoint_dir": str(CKPT_DIR.resolve()),
        "sampling": {
            "temperature": float(temperature),
            "top_k": None if top_k is None else int(top_k),
            "do_sample": bool(do_sample),
            "stop_on_eos": bool(stop_on_eos),
        },
        "samples": [],
    }

    block_size = get_model_block_size(model)

    for sample_idx, source_path in enumerate(chosen):
        ids = load_token_ids_from_json(source_path)
        prompt_tokens, prompt_start, gt_continuation = choose_prompt_from_song(
            ids=ids,
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
            random_offset=random_offset,
            require_after_eos=True,
            eos_token_id=EOS_ID if ADD_EOS else None,
        )

        full_generated, hit_eos = generate_from_prompt(
            model=model,
            prompt_tokens=prompt_tokens,
            block_size=block_size,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            stop_on_eos=stop_on_eos,
            eos_token_id=EOS_ID if stop_on_eos else None,
            device=device,
        )

        json_path = save_json_sample(
            out_dir=out_dir,
            sample_idx=sample_idx,
            split=split,
            ckpt_name=ckpt_name,
            source_path=source_path,
            prompt_start=prompt_start,
            prompt_tokens=prompt_tokens,
            gt_continuation=gt_continuation,
            full_generated_tokens=full_generated,
            hit_eos=hit_eos,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
        )

        midi_path = out_dir / f"sample_{sample_idx:03d}.mid"
        convert_tokens_to_midi(tokenizer, full_generated, midi_path)

        summary["samples"].append({
            "idx": int(sample_idx),
            "source_json": str(source_path),
            "prompt_start": int(prompt_start),
            "prompt_len": int(len(prompt_tokens)),
            "generated_len": int(len(full_generated) - len(prompt_tokens)),
            "hit_eos": bool(hit_eos),
            "json_path": str(json_path),
            "midi_path": str(midi_path),
        })
        print(f"[SAVE] {json_path}")
        print(f"[SAVE] {midi_path}")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[SAVE] {summary_path}")


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Genera muestras de pretraining en JSON y MIDI en una sola ejecución."
    )
    parser.add_argument("--mode", choices=["generate", "loss", "all"], default=MODE)
    parser.add_argument("--ckpt", choices=["best", "last"], default=CKPT_NAME)
    parser.add_argument("--split", choices=["train", "val", "test"], default=SPLIT)
    parser.add_argument("--prompt-len", "--input-tokens", dest="prompt_len", type=int, default=INPUT_TOKENS)
    parser.add_argument("--total-tokens", type=int, default=TOTAL_TOKENS)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--min-new-tokens", type=int, default=MIN_NEW_TOKENS)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=GREEDY, help="Activa argmax en lugar de sampling.")
    parser.add_argument("--device", type=str, default=RUN_DEVICE)
    parser.add_argument("--max-batches", type=int, default=MAX_BATCHES)
    parser.add_argument("--tokenizer-path", type=Path, default=TOKENIZER_PATH_OVERRIDE)
    parser.add_argument("--out-dir", "--output-dir", dest="out_dir", type=Path, default=OUTPUT_DIR_OVERRIDE)
    parser.add_argument(
        "--random-offset",
        action=argparse.BooleanOptionalAction,
        default=RANDOM_OFFSET,
        help="Activa un offset aleatorio valido para el prompt.",
    )
    parser.add_argument(
        "--stop-on-eos",
        action=argparse.BooleanOptionalAction,
        default=STOP_ON_EOS,
        help="Detiene la generación al emitir EOS.",
    )

    args = parser.parse_args()
    if args.prompt_len <= 0:
        raise ValueError("INPUT_TOKENS/--prompt-len debe ser mayor que 0.")
    if args.total_tokens is not None:
        if args.total_tokens <= args.prompt_len:
            raise ValueError("TOTAL_TOKENS/--total-tokens debe ser mayor que INPUT_TOKENS.")
        args.max_new_tokens = args.total_tokens - args.prompt_len
    if args.max_new_tokens <= 0:
        raise ValueError("MAX_NEW_TOKENS/--max-new-tokens debe ser mayor que 0.")
    if args.min_new_tokens < 0:
        raise ValueError("--min-new-tokens no puede ser negativo.")
    if args.min_new_tokens > args.max_new_tokens:
        raise ValueError("--min-new-tokens no puede superar --max-new-tokens.")
    return args


def main() -> None:
    """Punto de entrada del script cuando se ejecuta desde consola."""

    args = parse_args()
    seed_all(SEED)

    model, _ = load_checkpoint_and_model(args.ckpt, device=args.device)
    block_size = get_model_block_size(model)
    if args.prompt_len > block_size:
        raise ValueError(
            f"prompt_len={args.prompt_len} no puede ser mayor que block_size={block_size}."
        )

    if args.mode in {"loss", "all"}:
        evaluate_split_loss(
            model=model,
            split=args.split,
            block_size=block_size,
            device=args.device,
            max_batches=args.max_batches,
        )

    if args.mode in {"generate", "all"}:
        tokenizer = load_tokenizer(args.tokenizer_path)
        out_dir = args.out_dir
        if out_dir is None:
            out_dir = OUTPUT_DIR / f"{args.ckpt}_{args.split}_json_midi"

        run_generation_and_midi(
            model=model,
            tokenizer=tokenizer,
            ckpt_name=args.ckpt,
            split=args.split,
            prompt_len=args.prompt_len,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            num_samples=args.num_samples,
            random_offset=args.random_offset,
            temperature=args.temperature,
            top_k=args.top_k,
            do_sample=not args.greedy,
            stop_on_eos=args.stop_on_eos,
            device=args.device,
            out_dir=out_dir,
            tokenizer_path=args.tokenizer_path,
        )


# Ejecucion directa del script.
if __name__ == "__main__":
    main()
