"""
Gestiona una fase de preentrenamiento del modelo sobre el corpus tokenizado.

Incluye configuracion de datos, modelo, optimizacion y guardado de checkpoints para poder continuar o evaluar los experimentos del TFG.
"""

from __future__ import annotations

import math
import time
from math import ceil
from pathlib import Path

import torch

from src.model.model import MusicTransformerAutoregressive, MTModelConfig
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

# =============================================================================
# CONFIGURACION DEL EXPERIMENTO
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_CSV = PROJECT_ROOT / "data" / "interim" / "indexes" / "index_pretraining.csv"
TOKENS_DIR = PROJECT_ROOT / "data" / "interim" / "tokenized_json_bpe"
ANCHOR = r"data\interim\tokenized_json_bpe"

TOKEN_FIELD = "ids"
VOCAB_SIZE = 30000

VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 1453

CACHE_DIR = (PROJECT_ROOT / "data" / "bin" / "bin_for_pretraining").resolve()
ADD_EOS = True
EOS_ID = 2
USE_UINT16 = True

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
MIN_LR = 5e-6
WARMUP_UPDATES = 1000
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

EPOCHS = 1.2
EVAL_EVERY = 500
EVAL_BATCHES = 500
PRINT_EVERY = 50
SAVE_EVERY = 500

EARLY_STOP = True
PATIENCE_EVALS = 10
MIN_DELTA = 0.005
START_EARLY_AFTER = 5000

CKPT_DIR = (PROJECT_ROOT / "output" / "checkpoints" / "pretraining").resolve()

NUM_WORKERS = 2
PIN_MEMORY = True

USE_AMP = True
AMP_DTYPE = "bf16"


def cache_config() -> CacheConfig:
    """Define como se construyen train.bin, val.bin y test.bin para pretraining v1."""
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
        progress_every=2000,
    )


def model_config() -> MTModelConfig:
    """Construye la arquitectura usada en el primer pretraining."""
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


def train_one_update(model, opt, scaler, autocast_dtype, train_iter, train_loader):
    """Ejecuta un update completo: varios micro-batches, backward, clipping y step."""
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
            with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
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
    """Muestra perdida, learning rate y throughput estimado."""
    if update % PRINT_EVERY != 0:
        return

    elapsed = time.time() - start_time
    tokens_seen = (update + 1) * tokens_per_update
    tok_s = tokens_seen / max(elapsed, 1e-9)
    print(
        f"[upd {update:>6}/{total_updates}] loss={loss:.4f} lr={lr:.3e} "
        f"tokens_seen~{tokens_seen:,} tok/s~{tok_s:,.0f} elapsed={elapsed:.1f} sec"
    )


def maybe_save_last(update: int, model, opt, scaler, cfg: MTModelConfig) -> None:
    """Guarda un checkpoint de recuperacion aunque ese update no tenga validacion."""
    if update > 0 and update % SAVE_EVERY == 0:
        save_ckpt(CKPT_DIR / "last.pt", model, opt, scaler, cfg, update, val_loss=float("nan"))


def evaluate_validation(update: int, model, val_loader, opt, scaler, cfg: MTModelConfig, state: dict) -> bool:
    """Evalua validacion, actualiza el mejor checkpoint y decide early stopping."""
    if update <= 0 or update % EVAL_EVERY != 0:
        return False

    val_loss = evaluate(model, val_loader, DEVICE, max_batches=EVAL_BATCHES)
    print(f"[VAL] update={update} val_loss={val_loss:.4f}")

    if not math.isfinite(val_loss):
        print("[VAL][WARN] val_loss NaN/Inf -> no actualizo best ni early stopping.")
        save_ckpt(CKPT_DIR / "last.pt", model, opt, scaler, cfg, update, val_loss=state["last_val_loss"])
        return False

    state["last_val_loss"] = float(val_loss)
    save_ckpt(CKPT_DIR / "last.pt", model, opt, scaler, cfg, update, val_loss=val_loss)

    improved = val_loss < state["best_val"] - MIN_DELTA
    if improved:
        state["best_val"] = float(val_loss)
        state["no_improve"] = 0
        save_ckpt(CKPT_DIR / "best.pt", model, opt, scaler, cfg, update, val_loss=val_loss)
    else:
        state["no_improve"] += 1
        print(f"[EARLY] no_improve={state['no_improve']}/{PATIENCE_EVALS} | best_val={state['best_val']:.4f}")

    should_stop = EARLY_STOP and update >= START_EARLY_AFTER and state["no_improve"] >= PATIENCE_EVALS
    if should_stop:
        print("[EARLY STOP] val_loss no mejora. Se para entrenamiento.")
    return should_stop


def evaluate_test(model, test_loader) -> None:
    """Evalua test con best.pt si existe; si no, usa el modelo final."""
    best_path = CKPT_DIR / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        test_loss = evaluate(model, test_loader, DEVICE, max_batches=EVAL_BATCHES)
        print(f"[TEST] loss={test_loss:.4f} (evaluado con best.pt)")
    else:
        print("[TEST][WARN] No existe best.pt; evaluo test con el modelo final.")
        test_loss = evaluate(model, test_loader, DEVICE, max_batches=EVAL_BATCHES)
        print(f"[TEST] loss={test_loss:.4f}")


def main():
    """Punto de entrada del script cuando se ejecuta desde consola."""

    seed_all(SEED)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[ENV] device={DEVICE} torch={torch.__version__}")
    print(f"[INFO] CKPT_DIR = {CKPT_DIR}")
    print(f"[INFO] CACHE_DIR = {CACHE_DIR}")

    cache = prepare_cache_and_splits(cache_config())
    train_loader, val_loader, test_loader = make_loaders(
        cache,
        block_size=BLOCK_SIZE,
        batch_size=MICRO_BATCH,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        device=DEVICE,
        virtual_size=10_000_000,
    )

    cfg = model_config()
    model = MusicTransformerAutoregressive(cfg).to(DEVICE)
    opt = configure_optimizer(model, LR, WEIGHT_DECAY)
    scaler, autocast_dtype = configure_amp(DEVICE, USE_AMP, AMP_DTYPE)

    train_tokens = int(cache["meta"]["train_tokens"])
    tokens_per_update = MICRO_BATCH * BLOCK_SIZE * GRAD_ACCUM
    updates_per_epoch = math.ceil(train_tokens / tokens_per_update)
    total_updates = ceil(updates_per_epoch * EPOCHS)

    print(f"[PLAN] train_tokens={train_tokens:,}")
    print(f"[PLAN] tokens/update={tokens_per_update:,} (micro={MICRO_BATCH}, block={BLOCK_SIZE}, accum={GRAD_ACCUM})")
    print(f"[PLAN] updates/epoch~{updates_per_epoch} epochs={EPOCHS} total_updates={total_updates}")

    model.train()
    state = {"best_val": float("inf"), "no_improve": 0, "last_val_loss": float("nan")}
    train_iter = iter(train_loader)
    start_time = time.time()
    update = 0

    while update < total_updates:
        lr = lr_schedule(update, total_updates, LR, MIN_LR, WARMUP_UPDATES)
        for pg in opt.param_groups:
            pg["lr"] = lr

        loss, train_iter = train_one_update(model, opt, scaler, autocast_dtype, train_iter, train_loader)
        maybe_log_progress(update, total_updates, loss, lr, tokens_per_update, start_time)
        maybe_save_last(update, model, opt, scaler, cfg)

        if evaluate_validation(update, model, val_loader, opt, scaler, cfg, state):
            break

        update += 1

    print("[TRAIN DONE]")
    evaluate_test(model, test_loader)
    print("[DONE] pretraining finished.")


# Ejecucion directa del script.
if __name__ == "__main__":
    main()
