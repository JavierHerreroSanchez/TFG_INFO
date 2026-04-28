from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.training.pretraining import (
    INDEX_CSV,
    TOKENS_DIR,
    ANCHOR,
    TOKEN_FIELD,
    VAL_RATIO,
    TEST_RATIO,
    SEED,
    ADD_EOS,
    EOS_ID,
    MICRO_BATCH,
    NUM_WORKERS,
    PIN_MEMORY,
    DEVICE,
    EVAL_BATCHES,
    CKPT_DIR,
    seed_all,
    split_train_val_test,
    resolve_json_paths,
    prepare_cache_and_splits,
    MemmapRandomCropDataset,
    evaluate,
)
from src.model.model import MusicTransformerGPTlike, MTModelConfig

# =============================================================================
# CONFIGURACIÓN POR DEFECTO
# -----------------------------------------------------------------------------
# Este script está adaptado a la lógica de pretraining.py:
#   - mismo split reproducible train/val/test
#   - misma resolución de JSON tokenizados
#   - mismo caché binario memmap para loss teacher-forced
#   - misma carga de checkpoints best.pt / last.pt
#
# Tiene dos usos principales:
#   1) Evaluar loss en train / val / test con random crops sobre memmap.
#   2) Generar continuaciones autorregresivas a partir de prompts tomados de
#      MIDIs YA TOKENIZADOS en los JSON del split correspondiente.
# =============================================================================

DEFAULT_PROMPT_LEN = 150
DEFAULT_MAX_NEW_TOKENS = 2000
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_K = 160
DEFAULT_NUM_SAMPLES = 5
DEFAULT_RANDOM_OFFSET = True
DEFAULT_STOP_ON_EOS = True

OUTPUT_DIR = Path("../../output/generation_v2/batch_2").resolve()


def get_model_block_size(model: torch.nn.Module) -> int:
    """Obtiene block_size desde model.cfg, que es donde vive en MusicTransformerGPTlike."""
    cfg = getattr(model, "cfg", None)
    if cfg is None or not hasattr(cfg, "block_size"):
        raise AttributeError("El modelo no expone cfg.block_size.")
    return int(cfg.block_size)


# =============================================================================
# Utilidades de checkpoints / modelo
# =============================================================================

def load_checkpoint_and_model(ckpt_name: str, device: str) -> Tuple[MusicTransformerGPTlike, Dict]:
    """
    Carga best.pt o last.pt y reconstruye el modelo desde el cfg guardado
    dentro del checkpoint del pretraining.
    """
    if ckpt_name not in {"best", "last"}:
        raise ValueError("ckpt_name debe ser 'best' o 'last'.")

    ckpt_path = CKPT_DIR / f"{ckpt_name}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No existe checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    if "cfg" not in ckpt:
        raise KeyError(f"El checkpoint {ckpt_path} no contiene 'cfg'.")

    cfg = MTModelConfig(**ckpt["cfg"])
    model = MusicTransformerGPTlike(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"[CKPT] cargado: {ckpt_path.name}")
    print(f"[CKPT] update={ckpt.get('update', 'NA')} val_loss={ckpt.get('val_loss', float('nan'))}")
    print(f"[CKPT] block_size={model.cfg.block_size} vocab_size={model.cfg.vocab_size}")

    return model, ckpt


# =============================================================================
# Split real por ficheros JSON (no memmap)
# -----------------------------------------------------------------------------
# Para generación condicionada queremos seleccionar una pieza/tokenized JSON real
# del split train/val/test, tomar un prompt y dejar que el modelo continúe.
# Esto es diferente del memmap aleatorio del entrenamiento.
# =============================================================================

def get_split_files(split: str) -> List[Path]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split debe ser 'train', 'val' o 'test'.")

    paths = resolve_json_paths(INDEX_CSV, TOKENS_DIR, ANCHOR)
    train_files, val_files, test_files = split_train_val_test(paths, VAL_RATIO, TEST_RATIO, SEED)

    split_map = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }
    files = split_map[split]
    print(f"[DATA] split={split} | ficheros={len(files)}")
    return files


def load_token_ids_from_json(path: Path) -> List[int]:
    """
    Lee TOKEN_FIELD desde el JSON tokenizado y añade EOS si el pretraining lo hizo
    así también al construir el stream 1D.
    """
    obj = json.loads(path.read_text(encoding="utf-8"))
    ids = obj.get(TOKEN_FIELD, None)

    if not ids:
        raise ValueError(f"JSON sin tokens válidos en campo '{TOKEN_FIELD}': {path}")

    ids = [int(x) for x in ids]
    if ADD_EOS:
        ids = ids + [int(EOS_ID)]
    return ids


def choose_prompt_from_song(
        ids: List[int],
        prompt_len: int,
        max_new_tokens: int,
        random_offset: bool,
) -> Tuple[List[int], int, List[int]]:
    """
    Devuelve:
      - prompt_tokens
      - start_idx del prompt dentro de la pieza
      - continuación real (ground truth) hasta max_new_tokens, si existe
    """
    if len(ids) < prompt_len + 1:
        raise ValueError(
            f"Secuencia demasiado corta ({len(ids)} tokens) para prompt_len={prompt_len}"
        )

    if random_offset:
        max_start = max(0, len(ids) - prompt_len - 1)
        start = random.randint(0, max_start)
    else:
        start = 0

    prompt = ids[start:start + prompt_len]
    gt_cont = ids[start + prompt_len:start + prompt_len + max_new_tokens]
    return prompt, start, gt_cont


# =============================================================================
# Loss teacher-forced sobre memmap (igual que en pretraining)
# -----------------------------------------------------------------------------
# Reutilizamos exactamente la misma lógica de evaluate() y del dataset de random
# crops. Así la loss reportada es consistente con train/val/test del pretraining.
# =============================================================================

def build_memmap_loader(split: str, block_size: int):
    cache = prepare_cache_and_splits()
    dtype = cache["dtype"]

    bin_path = {
        "train": cache["train_bin"],
        "val": cache["val_bin"],
        "test": cache["test_bin"],
    }[split]

    ds = MemmapRandomCropDataset(bin_path, block_size, dtype)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=MICRO_BATCH,
        num_workers=NUM_WORKERS,
        pin_memory=(PIN_MEMORY and DEVICE == "cuda"),
    )
    return loader


def evaluate_split_loss(model: torch.nn.Module, split: str, block_size: int, device: str, max_batches: int):
    loader = build_memmap_loader(split, block_size)
    loss = evaluate(model, loader, device=device, max_batches=max_batches)
    print(f"[LOSS] split={split} | loss={loss:.4f}")
    return loss


# =============================================================================
# Generación autoregresiva
# =============================================================================

def sample_next_token(
        logits_last: torch.Tensor,
        temperature: float = 1.0,
        top_k: int | None = None,
        do_sample: bool = True,
) -> torch.Tensor:
    """
    logits_last: (1, V)
    devuelve next_token: (1, 1)
    """
    if temperature <= 0:
        raise ValueError("temperature debe ser > 0.")

    logits = logits_last / temperature

    if top_k is not None:
        k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k)
        logits[logits < v[:, [-1]]] = -float("inf")

    probs = F.softmax(logits, dim=-1)

    if do_sample:
        next_token = torch.multinomial(probs, num_samples=1)
    else:
        next_token = torch.argmax(probs, dim=-1, keepdim=True)

    return next_token


@torch.no_grad()
def generate_from_prompt(
        model: torch.nn.Module,
        prompt_tokens: List[int],
        block_size: int,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        do_sample: bool,
        stop_on_eos: bool,
        eos_token_id: int | None,
        device: str,
) -> Tuple[List[int], bool]:
    """
    Genera de forma autoregresiva a partir de prompt_tokens.
    Devuelve:
      - secuencia completa (prompt + generación)
      - flag indicando si se encontró EOS
    """
    idx = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
    saw_eos = False

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]  # recorte al contexto máximo real del modelo (cfg.block_size)

        out = model(idx_cond)

        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out

        if not torch.is_tensor(logits):
            raise TypeError("El modelo no ha devuelto logits válidos.")

        logits_last = logits[:, -1, :]  # (1, V)

        next_token = sample_next_token(
            logits_last=logits_last,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
        )

        idx = torch.cat([idx, next_token], dim=1)

        if stop_on_eos and eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
            saw_eos = True
            break

    return idx[0].tolist(), saw_eos


# =============================================================================
# Persistencia de resultados
# =============================================================================

def save_generation_result(
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
):
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
    print(f"[SAVE] {out_path}")


# =============================================================================
# Proceso de generación sobre ficheros reales de train/val/test
# =============================================================================

def run_generation(
        model: torch.nn.Module,
        ckpt_name: str,
        split: str,
        prompt_len: int,
        max_new_tokens: int,
        num_samples: int,
        random_offset: bool,
        temperature: float,
        top_k: int | None,
        do_sample: bool,
        stop_on_eos: bool,
        device: str,
):
    files = get_split_files(split)

    valid_files = []
    for p in files:
        try:
            ids = load_token_ids_from_json(p)
            if len(ids) >= prompt_len + 1:
                valid_files.append(p)
        except Exception as e:
            print(f"[WARN] saltando {p.name}: {e}")

    if len(valid_files) == 0:
        raise RuntimeError(f"No hay ficheros válidos en split={split} con prompt_len={prompt_len}")

    chosen = random.sample(valid_files, k=min(num_samples, len(valid_files)))

    out_dir = OUTPUT_DIR / f"{ckpt_name}_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "checkpoint": ckpt_name,
        "split": split,
        "prompt_len": int(prompt_len),
        "max_new_tokens": int(max_new_tokens),
        "num_samples_requested": int(num_samples),
        "num_samples_generated": int(len(chosen)),
        "random_offset": bool(random_offset),
        "sampling": {
            "temperature": float(temperature),
            "top_k": None if top_k is None else int(top_k),
            "do_sample": bool(do_sample),
            "stop_on_eos": bool(stop_on_eos),
        },
        "samples": [],
    }

    for i, path in enumerate(chosen):
        ids = load_token_ids_from_json(path)

        prompt_tokens, prompt_start, gt_continuation = choose_prompt_from_song(
            ids=ids,
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
            random_offset=random_offset,
        )

        full_generated, hit_eos = generate_from_prompt(
            model=model,
            prompt_tokens=prompt_tokens,
            block_size=get_model_block_size(model),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            stop_on_eos=stop_on_eos,
            eos_token_id=EOS_ID if stop_on_eos else None,
            device=device,
        )

        save_generation_result(
            out_dir=out_dir,
            sample_idx=i,
            split=split,
            ckpt_name=ckpt_name,
            source_path=path,
            prompt_start=prompt_start,
            prompt_tokens=prompt_tokens,
            gt_continuation=gt_continuation,
            full_generated_tokens=full_generated,
            hit_eos=hit_eos,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
        )

        summary["samples"].append({
            "idx": i,
            "source_json": str(path),
            "prompt_start": int(prompt_start),
            "prompt_len": int(len(prompt_tokens)),
            "generated_len": int(len(full_generated) - len(prompt_tokens)),
            "hit_eos": bool(hit_eos),
        })

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[SAVE] {summary_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluación y generación para el pretraining del Music Transformer GPT-like."
    )

    parser.add_argument("--mode", choices=["loss", "generate", "all"], default="all")
    parser.add_argument("--ckpt", choices=["best", "last"], default="best")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")

    parser.add_argument("--prompt-len", type=int, default=DEFAULT_PROMPT_LEN)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--greedy", action="store_true", help="Usa argmax en lugar de sampling.")
    parser.add_argument("--random-offset", action="store_true", default=DEFAULT_RANDOM_OFFSET)
    parser.add_argument("--no-stop-on-eos", action="store_true", help="No parar al generar EOS.")
    parser.add_argument("--max-batches", type=int, default=EVAL_BATCHES)
    parser.add_argument("--device", type=str, default=DEVICE)

    return parser.parse_args()


def main():
    args = parse_args()
    seed_all(SEED)

    do_sample = not args.greedy
    stop_on_eos = not args.no_stop_on_eos

    model, ckpt = load_checkpoint_and_model(args.ckpt, device=args.device)
    block_size = get_model_block_size(model)

    if args.prompt_len > block_size:
        raise ValueError(
            f"prompt_len={args.prompt_len} no puede ser mayor que block_size={block_size} del checkpoint."
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
        run_generation(
            model=model,
            ckpt_name=args.ckpt,
            split=args.split,
            prompt_len=args.prompt_len,
            max_new_tokens=args.max_new_tokens,
            num_samples=args.num_samples,
            random_offset=args.random_offset,
            temperature=args.temperature,
            top_k=args.top_k,
            do_sample=do_sample,
            stop_on_eos=stop_on_eos,
            device=args.device,
        )


if __name__ == "__main__":
    main()
