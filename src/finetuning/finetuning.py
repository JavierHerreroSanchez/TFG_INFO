"""
Gestiona la preparacion y el ajuste fino sobre el subconjunto musical objetivo.

Esta fase adapta el modelo preentrenado al dominio final estudiado en el TFG.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.model.model import MusicTransformerGPTlike, MTModelConfig
from src.training.common import (
    CacheConfig,
    configure_amp,
    configure_optimizer,
    evaluate,
    lr_schedule,
    make_loaders,
    prepare_cache_and_splits,
    save_ckpt,
    seed_all,
)

"""
Fine-tuning base del modelo preentrenado.

Esta version conserva los parametros del primer flujo de ajuste fino, pero usa
la infraestructura comun para que el codigo sea comparable con finetuning_v2 y
mas facil de leer, mantener y explicar.
"""


# =============================================================================
# CONFIGURACION DEL EXPERIMENTO
# =============================================================================

INDEX_CSV = Path(r"../../output/generation_finetuning_tfg_first/finetuning_aug_index.csv")
TOKENS_DIR = Path(r"../../data/interim/tokenized_finetuning")
ANCHOR = r"data\interim\tokenized_finetuning"

TOKEN_FIELD = "ids"
VOCAB_SIZE = 30000

VAL_RATIO = 0.10
TEST_RATIO = 0.10
SEED = 1453

CACHE_DIR = Path(r"../../data/bin/bin_for_finetuning").resolve()
ADD_EOS = True
EOS_ID = 2
USE_UINT16 = True

PRETRAINED_CKPT = Path(r"../../output/checkpoints/pretraining/best.pt").resolve()

BLOCK_SIZE = 2048
D_MODEL = 512
N_HEADS = 8
N_LAYER = 8
DROPOUT = 0.1
D_FF = None
TIE_WEIGHTS = True
USE_FINAL_LN = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MICRO_BATCH = 1
GRAD_ACCUM = 16

LR = 3e-4
MIN_LR = 2e-6
WARMUP_UPDATES = 100
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0

MAX_EPOCHS = 100
MAX_UPDATES: Optional[int] = None
EVAL_EVERY = 100
EVAL_BATCHES = 64
PRINT_EVERY = 50
SAVE_EVERY = 500

EARLY_STOP = True
PATIENCE_EVALS = 12
MIN_DELTA = 1e-4
START_EARLY_AFTER = 2000

NUM_WORKERS = 2
PIN_MEMORY = True

USE_AMP = True
AMP_DTYPE = "bf16"

CKPT_DIR = Path(r"../../output/checkpoints/finetuning").resolve()
SAMPLES_DIR = CKPT_DIR / "samples"

N_LISTEN_SAMPLES = 4
LISTEN_PRIMER_TOKENS = 128
LISTEN_GEN_TOKENS = 512
GEN_TEMPERATURE = 1.0
GEN_TOP_K = 140


def cache_config() -> CacheConfig:
    """Define el cache binario para el dataset de finetuning v1."""
    return CacheConfig(
        index_csv=INDEX_CSV,
        tokens_dir=TOKENS_DIR,
        anchor=ANCHOR,
        token_field=TOKEN_FIELD,
        vocab_size=VOCAB_SIZE,
        cache_dir=CACHE_DIR,
        block_size=BLOCK_SIZE,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
        use_uint16=USE_UINT16,
        add_eos=ADD_EOS,
        eos_id=EOS_ID,
        progress_every=500,
    )


def model_config() -> MTModelConfig:
    """Construye una arquitectura compatible con el checkpoint preentrenado v1."""
    return MTModelConfig(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        dropout=DROPOUT,
        d_ff=D_FF,
        bias=True,
        tie_weights=TIE_WEIGHTS,
        use_final_ln=USE_FINAL_LN,
        debug=False,
    )


def load_pretrained_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: str) -> None:
    """Carga pesos de pretraining y muestra diferencias si el checkpoint no encaja."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No existe checkpoint preentrenado: {ckpt_path}")

    obj = torch.load(ckpt_path, map_location=device)
    state = obj["model"] if "model" in obj else obj
    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"[LOAD] checkpoint: {ckpt_path.name}")
    print(f"[LOAD] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
    if missing:
        print("[LOAD][WARN] Missing keys:")
        for key in missing[:20]:
            print("   ", key)
    if unexpected:
        print("[LOAD][WARN] Unexpected keys:")
        for key in unexpected[:20]:
            print("   ", key)


@torch.no_grad()
def export_listen_samples(model: MusicTransformerGPTlike, test_bin: Path, dtype, out_dir: Path, update: int, device: str) -> None:
    """Genera muestras JSON para comparar auditivamente la evolucion del ajuste."""
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.memmap(test_bin, mode="r", dtype=dtype)
    n = int(data.shape[0])
    if n < LISTEN_PRIMER_TOKENS + 1:
        print("[LISTEN][WARN] test.bin demasiado pequeno para generar muestras.")
        return

    rng = random.Random(SEED)
    max_start = n - (LISTEN_PRIMER_TOKENS + 1)
    starts = [rng.randint(0, max_start) for _ in range(N_LISTEN_SAMPLES)]

    model.eval()
    for i, start in enumerate(starts):
        primer = np.asarray(data[start:start + LISTEN_PRIMER_TOKENS], dtype=np.int64)
        x = torch.from_numpy(primer).unsqueeze(0).to(device)
        y = model.generate(
            x,
            max_new_tokens=LISTEN_GEN_TOKENS,
            temperature=GEN_TEMPERATURE,
            top_k=GEN_TOP_K,
        )

        full_ids = y[0].detach().cpu().tolist()
        payload = {
            "update": update,
            "sample_id": i,
            "primer_ids": primer.tolist(),
            "generated_ids": full_ids[LISTEN_PRIMER_TOKENS:],
            "full_ids": full_ids,
            "temperature": GEN_TEMPERATURE,
            "top_k": GEN_TOP_K,
        }

        out_path = out_dir / f"update_{update:06d}_sample_{i:02d}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    model.train()
    print(f"[LISTEN] guardadas {N_LISTEN_SAMPLES} muestras en: {out_dir}")


def train_one_update(model, opt, scaler, autocast_dtype, train_iter, train_loader):
    """Entrena un update completo con acumulacion de gradiente."""
    opt.zero_grad(set_to_none=True)
    accum_loss = 0.0

    for _ in range(GRAD_ACCUM):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        if DEVICE == "cuda" and USE_AMP and autocast_dtype is not None:
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                _, loss = model(x, y)
                loss = loss / GRAD_ACCUM
        else:
            _, loss = model(x, y)
            loss = loss / GRAD_ACCUM

        accum_loss += loss.item()
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    if scaler is not None:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()

    return accum_loss, train_iter


def maybe_log_progress(update: int, total_updates: int, loss: float, lr: float, tokens_per_update: int, start_time: float) -> None:
    """Imprime metricas de progreso sin mezclar logica de entrenamiento."""
    if update % PRINT_EVERY != 0:
        return

    elapsed = time.time() - start_time
    tokens_seen = update * tokens_per_update
    tok_s = tokens_seen / max(elapsed, 1e-9)
    print(
        f"[upd {update:>6}/{total_updates}] loss={loss:.4f} lr={lr:.3e} "
        f"tokens_seen~{tokens_seen:,} tok/s~{tok_s:,.0f} elapsed={elapsed:.1f} sec"
    )


def evaluate_and_checkpoint(update: int, total_updates: int, model, val_loader, opt, scaler, cfg: MTModelConfig, cache, state: dict) -> bool:
    """Evalua, guarda last/best, exporta muestras y decide si activar early stopping."""
    if update % EVAL_EVERY != 0 and update != total_updates:
        return False

    val_loss = evaluate(model, val_loader, DEVICE, EVAL_BATCHES)
    print(f"[VAL] update={update} val_loss={val_loss:.4f}")

    save_ckpt(
        CKPT_DIR / "last.pt",
        model,
        opt,
        scaler,
        cfg,
        update=update,
        val_loss=val_loss,
        extra={"best_val_so_far": state["best_val"], "best_update_so_far": state["best_update"]},
    )

    export_listen_samples(
        model=model,
        test_bin=cache["test_bin"],
        dtype=cache["dtype"],
        out_dir=SAMPLES_DIR,
        update=update,
        device=DEVICE,
    )

    improved = (state["best_val"] - val_loss) > MIN_DELTA
    if improved:
        state["best_val"] = val_loss
        state["best_update"] = update
        state["patience_count"] = 0
        save_ckpt(
            CKPT_DIR / "best.pt",
            model,
            opt,
            scaler,
            cfg,
            update=update,
            val_loss=val_loss,
            extra={"best_update": state["best_update"]},
        )
        print(f"[BEST] nuevo mejor checkpoint en update={update} | val_loss={val_loss:.4f}")
    else:
        state["patience_count"] += 1
        print(f"[EARLY] sin mejora significativa ({state['patience_count']}/{PATIENCE_EVALS})")

    if update % SAVE_EVERY == 0:
        save_ckpt(CKPT_DIR / f"step_{update:06d}.pt", model, opt, scaler, cfg, update=update, val_loss=val_loss)

    should_stop = EARLY_STOP and update >= START_EARLY_AFTER and state["patience_count"] >= PATIENCE_EVALS
    if should_stop:
        print("[EARLY] stopping activado.")
    return should_stop


def evaluate_test(model, test_loader) -> None:
    """Carga best.pt y evalua test si existe un mejor checkpoint."""
    best_ckpt = CKPT_DIR / "best.pt"
    if not best_ckpt.exists():
        return

    print(f"[TEST] cargando mejor checkpoint: {best_ckpt.name}")
    obj = torch.load(best_ckpt, map_location=DEVICE)
    model.load_state_dict(obj["model"], strict=True)
    test_loss = evaluate(model, test_loader, DEVICE, EVAL_BATCHES)
    print(f"[TEST] test_loss={test_loss:.4f}")


def main():
    """Punto de entrada del script cuando se ejecuta desde consola."""

    seed_all(SEED)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[ENV] device={DEVICE} torch={torch.__version__}")
    print(f"[INFO] PRETRAINED_CKPT = {PRETRAINED_CKPT}")
    print(f"[INFO] CACHE_DIR = {CACHE_DIR}")
    print(f"[INFO] CKPT_DIR = {CKPT_DIR}")

    cache = prepare_cache_and_splits(cache_config())
    train_loader, val_loader, test_loader = make_loaders(
        cache,
        block_size=BLOCK_SIZE,
        batch_size=MICRO_BATCH,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        device=DEVICE,
    )

    cfg = model_config()
    model = MusicTransformerGPTlike(cfg).to(DEVICE)
    load_pretrained_checkpoint(model, PRETRAINED_CKPT, DEVICE)

    opt = configure_optimizer(model, LR, WEIGHT_DECAY)
    scaler, autocast_dtype = configure_amp(DEVICE, USE_AMP, AMP_DTYPE)

    train_tokens = int(cache["meta"]["train_tokens"])
    tokens_per_update = MICRO_BATCH * BLOCK_SIZE * GRAD_ACCUM
    updates_per_epoch = max(1, math.ceil(train_tokens / tokens_per_update))
    total_updates = MAX_UPDATES if MAX_UPDATES is not None else updates_per_epoch * MAX_EPOCHS

    print(f"[PLAN] train_tokens={train_tokens:,}")
    print(f"[PLAN] tokens/update={tokens_per_update:,}")
    print(f"[PLAN] updates/epoch~{updates_per_epoch}")
    print(f"[PLAN] total_updates={total_updates}")

    model.train()
    state = {"best_val": float("inf"), "best_update": -1, "patience_count": 0}
    train_iter = iter(train_loader)
    start_time = time.time()
    update = 0

    while update < total_updates:
        lr = lr_schedule(update, total_updates, LR, MIN_LR, WARMUP_UPDATES)
        for pg in opt.param_groups:
            pg["lr"] = lr

        loss, train_iter = train_one_update(model, opt, scaler, autocast_dtype, train_iter, train_loader)
        update += 1

        maybe_log_progress(update, total_updates, loss, lr, tokens_per_update, start_time)
        if evaluate_and_checkpoint(update, total_updates, model, val_loader, opt, scaler, cfg, cache, state):
            break

    print("[DONE] finetuning terminado.")
    print(f"[DONE] best_val={state['best_val']:.4f} @ update={state['best_update']}")
    evaluate_test(model, test_loader)


# Ejecucion directa del script.
if __name__ == "__main__":
    main()
