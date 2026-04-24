from __future__ import annotations

import json
import math
import random
import time
import sys
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch


class Tee:
    """Duplica stdout/stderr a consola y fichero con flush inmediato."""
    def __init__(self, console_stream, file_stream):
        self.console = console_stream
        self.file = file_stream

    def write(self, data):
        self.console.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        try:
            self.console.flush()
        except Exception:
            pass
        try:
            self.file.flush()
        except Exception:
            pass


from src.model.model import MusicTransformerGPTlike, MTModelConfig

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

# 1) Rutas
INDEX_CSV = Path(r"../../data/interim/debug_dataset/index_pretraining_v2.csv")
TOKENS_DIR = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\data\interim\tokenized_json_bpe_v2")
ANCHOR = r"data\interim\tokenized_json_bpe_v2"
TOKEN_FIELD = "ids"
VOCAB_SIZE = 18000

# 2) Split de dataset
VAL_RATIO = 0.05
TEST_RATIO = 0.05
SEED = 1453

# 3) Caché binario
CACHE_DIR = Path(r"../../data/bin/bin_for_pretraining_v2").resolve()
ADD_BOS = True
BOS_ID = 1
ADD_EOS = True
EOS_ID = 2
USE_UINT16 = True

# 4) Hiperparámetros del modelo
BLOCK_SIZE = 2048
D_MODEL = 512
N_HEADS = 8
N_LAYER = 8
DROPOUT = 0.1
D_FF = None
TIE_WEIGHTS = True
USE_FINAL_LN = True

# 5) Hiperparámetros de entrenamiento
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MICRO_BATCH = 2
GRAD_ACCUM = 8

LR = 3e-4
MIN_LR = 5e-6
WARMUP_UPDATES = 1000
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

# Referencia informativa; la reanudación manda con TARGET_TOTAL_UPDATES.
EPOCHS = 1.2
TARGET_TOTAL_UPDATES = 40000   # <-- se debe cambiar al update final deseado

EVAL_EVERY = 500
EVAL_BATCHES = 200
PRINT_EVERY = 100
SAVE_EVERY = 500

# =============================================================================
# EARLY STOPPING
# -----------------------------------------------------------------------------
# Lo restauramos desde checkpoint si existe `extra["early_stopping"]`.
# Si el checkpoint antiguo no lo tiene, tomamos como mejor valor el de best.pt
# y reiniciamos el contador de paciencia.
# =============================================================================
EARLY_STOPPING = True
ES_PATIENCE_EVALS = 12
ES_MIN_DELTA = 1e-3
ES_WARMUP_EVALS = 2

CKPT_DIR = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\output\checkpoints\pretraining_v2").resolve()
RESUME_CKPT = CKPT_DIR / "last.pt"
BEST_CKPT = CKPT_DIR / "best.pt"

LOG_DIR = CKPT_DIR / "logs"
STDOUT_LOG = LOG_DIR / "stdout_reanudation.log"

NUM_WORKERS = 0
PIN_MEMORY = True

USE_AMP = True
AMP_DTYPE = "bf16"   # "bf16" o "fp16"


# =============================================================================
# Funciones generales
# =============================================================================

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_np_dtype(use_uint16: bool, vocab_size: int):
    if use_uint16:
        if vocab_size >= 65535:
            raise ValueError("VOCAB_SIZE no cabe en uint16; usa uint32 (USE_UINT16=False).")
        return np.uint16
    return np.uint32


def split_train_val_test(paths: List[Path], val_ratio: float, test_ratio: float, seed: int):
    rng = random.Random(seed)
    p = paths[:]
    rng.shuffle(p)
    n = len(p)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    test_paths = p[:n_test]
    val_paths = p[n_test:n_test + n_val]
    train_paths = p[n_test + n_val:]
    return train_paths, val_paths, test_paths


def rebase_path(abs_path: str, tokens_dir: Path, anchor: str) -> Path:
    s = abs_path.replace("\\", "/")
    a = anchor.replace("\\", "/")
    pos = s.find(a)
    if pos == -1:
        return Path(abs_path)
    rel_part = s[pos + len(a):].lstrip("/")
    return tokens_dir / rel_part


def resolve_json_paths(index_csv: Path, tokens_dir: Path, anchor: str) -> List[Path]:
    if not index_csv.exists():
        raise FileNotFoundError(f"INDEX_CSV no existe: {index_csv}")

    df = pd.read_csv(index_csv)
    if "path" not in df.columns:
        raise ValueError("index_pretraining.csv debe tener columna 'path'.")

    raw_paths = df["path"].tolist()

    paths1 = [Path(p) for p in raw_paths]
    exist1 = [p for p in paths1 if p.exists()]
    if len(exist1) > 0:
        print(f"[DATA] paths OK (tal cual): {len(exist1)}")
        return exist1

    print("[DATA][WARN] 0 paths existentes usando rutas absolutas del CSV. Intento rebase...")

    if tokens_dir.exists():
        paths2 = [rebase_path(p, tokens_dir, anchor) for p in raw_paths]
        exist2 = [p for p in paths2 if p.exists()]
        if len(exist2) > 0:
            print(f"[DATA] paths OK (rebase): {len(exist2)}")
            return exist2

    print("[DATA][WARN] 0 paths existentes tras rebase. Fallback: escaneo TOKENS_DIR...")

    if not tokens_dir.exists():
        raise FileNotFoundError(f"TOKENS_DIR no existe: {tokens_dir}")

    scan = sorted([p for p in tokens_dir.rglob("*.json") if p.is_file()])
    print(f"[DATA] paths OK (scan): {len(scan)}")
    return scan


def file_size_multiple_of_dtype(path: Path, dtype) -> bool:
    if not path.exists():
        return False
    size = path.stat().st_size
    return (size % np.dtype(dtype).itemsize) == 0


def safe_remove(path: Path):
    if path.exists():
        path.unlink()


def build_memmap(
    files: List[Path],
    out_bin: Path,
    token_field: str,
    dtype,
    add_bos: bool,
    bos_id: int,
    add_eos: bool,
    eos_id: int,
) -> int:
    if len(files) == 0:
        raise ValueError("No hay ficheros para construir el memmap (lista vacía).")

    total = 0
    for p in files:
        obj = json.loads(p.read_text(encoding="utf-8"))
        ids = obj.get(token_field, None)
        if not ids:
            continue
        ids = list(ids)
        if add_bos and ids and ids[0] == bos_id:
            ids = ids[1:]
        if add_eos and ids and ids[-1] == eos_id:
            ids = ids[:-1]
        total += len(ids) + (1 if add_bos else 0) + (1 if add_eos else 0)

    if total <= 0:
        raise ValueError("Total tokens = 0. ¿TOKEN_FIELD correcto? ¿JSON vacíos?")

    mm = np.memmap(out_bin, mode="w+", dtype=dtype, shape=(total,))
    w = 0

    for i, p in enumerate(files, start=1):
        obj = json.loads(p.read_text(encoding="utf-8"))
        ids = obj.get(token_field, None)
        if not ids:
            continue
        ids = list(ids)

        if add_bos and ids and ids[0] == bos_id:
            ids = ids[1:]
        if add_eos and ids and ids[-1] == eos_id:
            ids = ids[:-1]

        if add_bos:
            mm[w] = np.asarray([bos_id], dtype=dtype)[0]
            w += 1

        arr = np.asarray(ids, dtype=dtype)
        mm[w:w + len(arr)] = arr
        w += len(arr)

        if add_eos:
            mm[w] = np.asarray([eos_id], dtype=dtype)[0]
            w += 1

        if i % 2000 == 0:
            print(f"[cache] {i}/{len(files)} escritos | tokens={w:,}")

    mm.flush()
    assert w == total
    print(f"[cache] OK -> {out_bin.name} | total_tokens={total:,}")
    return total


class MemmapRandomCropDataset(torch.utils.data.Dataset):
    def __init__(self, bin_path: Path, block_size: int, dtype):
        if not bin_path.exists():
            raise FileNotFoundError(f"Bin no existe: {bin_path}")
        if not file_size_multiple_of_dtype(bin_path, dtype):
            raise ValueError(
                f"El bin {bin_path} está corrupto o no coincide con dtype={dtype}. "
                f"Borra CACHE_DIR y reconstruye."
            )
        self.data = np.memmap(bin_path, mode="r", dtype=dtype)
        self.n = int(self.data.shape[0])
        self.block_size = block_size

        if self.n < block_size + 1:
            raise ValueError(f"Stream demasiado corto ({self.n}) para block_size={block_size}")
        self.max_start = self.n - (block_size + 1)

    def __len__(self):
        return 10_000_000

    def __getitem__(self, idx):
        start = random.randint(0, self.max_start)
        chunk = np.asarray(self.data[start:start + self.block_size + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


# =============================================================================
# Optimizador y LR schedule
# =============================================================================

def configure_optimizer(model: torch.nn.Module, lr: float, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = name.lower()
        if n.endswith("bias") or "ln" in n or "layernorm" in n or "embedding" in n or "wte" in n:
            no_decay.append(p)
        else:
            decay.append(p)

    print(
        f"[optim] decay_params={sum(p.numel() for p in decay):,} | "
        f"no_decay_params={sum(p.numel() for p in no_decay):,}"
    )
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.999), eps=1e-8
    )


def lr_schedule(update: int, total_updates: int, base_lr: float, min_lr: float, warmup: int):
    if update < warmup:
        return base_lr * (update + 1) / max(1, warmup)
    t = (update - warmup) / max(1, total_updates - warmup)
    t = min(max(t, 0.0), 1.0)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * t)) * (base_lr - min_lr)


@torch.no_grad()
def evaluate(model, loader, device: str, max_batches: int):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("inf")


@torch.no_grad()
def read_best_checkpoint_info(best_ckpt_path: Path, device: str):
    if not best_ckpt_path.exists():
        print(f"[BEST][WARN] No existe {best_ckpt_path}")
        return None

    obj = torch.load(best_ckpt_path, map_location=device)
    best_update = int(obj.get("update", -1))
    best_val_loss = float(obj.get("val_loss", float("inf")))
    extra = obj.get("extra", {}) if isinstance(obj, dict) else {}
    print(f"[BEST] file={best_ckpt_path.name} | update={best_update} | val_loss={best_val_loss:.6f}")
    if extra:
        print(f"[BEST] extra keys: {list(extra.keys())}")
    return {"update": best_update, "val_loss": best_val_loss, "extra": extra}


def save_ckpt(path: Path, model, opt, scaler, cfg: MTModelConfig, update: int, val_loss: float, extra: Optional[dict] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "update": update,
        "val_loss": val_loss,
        "cfg": cfg.__dict__,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
    }
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, path)
    print(f"[CKPT] saved: {path.name} | update={update} val_loss={val_loss:.4f}")


# =============================================================================
# Cache y splits
# =============================================================================

def prepare_cache_and_splits() -> Dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    dtype = choose_np_dtype(USE_UINT16, VOCAB_SIZE)

    train_bin = CACHE_DIR / "train.bin"
    val_bin = CACHE_DIR / "val.bin"
    test_bin = CACHE_DIR / "test.bin"
    meta_json = CACHE_DIR / "meta.json"

    paths = resolve_json_paths(INDEX_CSV, TOKENS_DIR, ANCHOR)
    print(f"[DATA] json existentes: {len(paths)}")
    if len(paths) == 0:
        raise RuntimeError("No se ha encontrado ningún JSON. Revisa INDEX_CSV/TOKENS_DIR/ANCHOR.")

    train_files, val_files, test_files = split_train_val_test(paths, VAL_RATIO, TEST_RATIO, SEED)
    print(f"[DATA] split: train={len(train_files)} val={len(val_files)} test={len(test_files)}")

    for b in [train_bin, val_bin, test_bin]:
        if b.exists() and not file_size_multiple_of_dtype(b, dtype):
            print(f"[cache][WARN] {b.name} no coincide con dtype -> borrando")
            safe_remove(b)

    if not train_bin.exists() or not val_bin.exists() or not test_bin.exists() or not meta_json.exists():
        print("[cache] Construyendo memmaps...")
        train_tokens = build_memmap(train_files, train_bin, TOKEN_FIELD, dtype, ADD_BOS, BOS_ID, ADD_EOS, EOS_ID)
        val_tokens = build_memmap(val_files, val_bin, TOKEN_FIELD, dtype, ADD_BOS, BOS_ID, ADD_EOS, EOS_ID)
        test_tokens = build_memmap(test_files, test_bin, TOKEN_FIELD, dtype, ADD_BOS, BOS_ID, ADD_EOS, EOS_ID)

        meta = {
            "vocab_size": VOCAB_SIZE,
            "block_size": BLOCK_SIZE,
            "dtype": str(dtype),
            "train_tokens": int(train_tokens),
            "val_tokens": int(val_tokens),
            "test_tokens": int(test_tokens),
            "train_files": len(train_files),
            "val_files": len(val_files),
            "test_files": len(test_files),
            "token_field": TOKEN_FIELD,
            "add_bos": ADD_BOS,
            "bos_id": BOS_ID,
            "add_eos": ADD_EOS,
            "eos_id": EOS_ID,
        }
        meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[cache] meta guardado: {meta_json}")
    else:
        meta = json.loads(meta_json.read_text(encoding="utf-8"))
        print(
            f"[cache] Reusando cache | train_tokens={meta['train_tokens']:,} "
            f"val_tokens={meta['val_tokens']:,} test_tokens={meta['test_tokens']:,}"
        )

    return {
        "dtype": dtype,
        "train_bin": train_bin,
        "val_bin": val_bin,
        "test_bin": test_bin,
        "meta": meta,
    }


# =============================================================================
# Entrenamiento
# =============================================================================

def main():
    seed_all(SEED)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_fh = STDOUT_LOG.open("a", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.stdout, log_fh)
    sys.stderr = Tee(sys.stderr, log_fh)

    print("=" * 90)
    print("[RUN] pretraining_with_reanudation_v2")
    print(f"[ENV] device={DEVICE} torch={torch.__version__}")
    print(f"[INFO] CKPT_DIR = {CKPT_DIR}")
    print(f"[INFO] CACHE_DIR = {CACHE_DIR}")
    print(f"[INFO] RESUME_CKPT = {RESUME_CKPT}")
    print(f"[INFO] BEST_CKPT = {BEST_CKPT}")
    print(f"[INFO] STDOUT_LOG = {STDOUT_LOG}")
    print("=" * 90)

    cache = prepare_cache_and_splits()
    dtype = cache["dtype"]

    train_ds = MemmapRandomCropDataset(cache["train_bin"], BLOCK_SIZE, dtype)
    val_ds = MemmapRandomCropDataset(cache["val_bin"], BLOCK_SIZE, dtype)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=MICRO_BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=MICRO_BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )

    cfg = MTModelConfig(
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
    model = MusicTransformerGPTlike(cfg).to(DEVICE)
    opt = configure_optimizer(model, LR, WEIGHT_DECAY)

    amp_dtype = torch.bfloat16 if AMP_DTYPE == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and AMP_DTYPE == "fp16" and DEVICE.startswith("cuda")))

    # -------------------------------------------------------------------------
    # Leemos best.pt para conocer el mejor val_loss histórico.
    # -------------------------------------------------------------------------
    best_info = read_best_checkpoint_info(BEST_CKPT, DEVICE)
    if best_info is not None:
        best_val = best_info["val_loss"]
        best_ckpt_update = best_info["update"]
    else:
        best_val = float("inf")
        best_ckpt_update = -1

    # Estado de reanudación y early stopping.
    resume_update = 0
    last_val_loss = float("nan")
    no_improve_evals = 0
    n_evals_done = 0

    if RESUME_CKPT.exists():
        ckpt = torch.load(RESUME_CKPT, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])

        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])

        resume_update = int(ckpt.get("update", -1)) + 1
        last_val_loss = float(ckpt.get("val_loss", float("nan")))

        extra = ckpt.get("extra", {}) if isinstance(ckpt, dict) else {}
        es_extra = extra.get("early_stopping", {}) if isinstance(extra, dict) else {}

        if es_extra:
            best_val = float(es_extra.get("best_val", best_val))
            no_improve_evals = int(es_extra.get("no_improve_evals", 0))
            n_evals_done = int(es_extra.get("n_evals_done", 0))
            best_ckpt_update = int(es_extra.get("best_ckpt_update", best_ckpt_update))
            print(
                f"[RESUME] Restored ES state | best_val={best_val:.6f} "
                f"no_improve={no_improve_evals} evals_done={n_evals_done} best_update={best_ckpt_update}"
            )
        else:
            print("[RESUME] Checkpoint antiguo sin estado de early stopping; se usará best.pt como referencia.")

        print(f"[RESUME] Loaded {RESUME_CKPT.name} | ckpt_update={resume_update - 1} | next_update={resume_update}")
        print(f"[RESUME] last val_loss saved in last.pt = {last_val_loss:.6f}")
    else:
        print("[RESUME] No existe last.pt; se entrenará desde cero.")

    train_tokens = int(cache["meta"]["train_tokens"])
    tokens_per_update = MICRO_BATCH * BLOCK_SIZE * GRAD_ACCUM
    updates_per_epoch = math.ceil(train_tokens / tokens_per_update)
    epochs_equiv = TARGET_TOTAL_UPDATES / max(1, updates_per_epoch)

    total_updates = max(TARGET_TOTAL_UPDATES, resume_update + 1)

    print(f"[PLAN] train_tokens={train_tokens:,}")
    print(f"[PLAN] tokens/update={tokens_per_update:,}")
    print(f"[PLAN] updates/epoch≈{updates_per_epoch}")
    print(f"[PLAN] EPOCHS(ref)={EPOCHS}")
    print(f"[PLAN] TARGET_TOTAL_UPDATES={TARGET_TOTAL_UPDATES}")
    print(f"[PLAN] total_updates={total_updates}")
    print(f"[PLAN] epochs_equiv≈{epochs_equiv:.3f}")
    print(f"[PLAN] current best_val={best_val:.6f} (best.pt update={best_ckpt_update})")

    model.train()
    t0 = time.time()
    update = resume_update
    train_iter = iter(train_loader)

    while update < total_updates:
        lr = lr_schedule(update, total_updates, LR, MIN_LR, WARMUP_UPDATES)
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(GRAD_ACCUM):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(USE_AMP and DEVICE.startswith("cuda"))):
                _, loss = model(x, y)
                loss = loss / GRAD_ACCUM

            accum_loss += loss.item()

            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

        if (update % PRINT_EVERY == 0) or (update == resume_update):
            dt = time.time() - t0
            print(
                f"[train] update={update:6d}/{total_updates} | "
                f"loss={accum_loss:.4f} | lr={lr:.6e} | dt={dt:.1f}s"
            )
            t0 = time.time()

        if (update > 0) and (update % EVAL_EVERY == 0):
            val_loss = evaluate(model, val_loader, DEVICE, EVAL_BATCHES)
            n_evals_done += 1
            improved = val_loss < (best_val - ES_MIN_DELTA)

            print(
                f"[eval] update={update:6d} | val_loss={val_loss:.6f} | "
                f"best_val={best_val:.6f} | improved={improved}"
            )

            last_extra = {
                "early_stopping": {
                    "best_val": float(best_val),
                    "no_improve_evals": int(no_improve_evals),
                    "n_evals_done": int(n_evals_done),
                    "best_ckpt_update": int(best_ckpt_update),
                }
            }

            if improved:
                best_val = val_loss
                no_improve_evals = 0
                best_ckpt_update = update
                best_extra = {
                    "early_stopping": {
                        "best_val": float(best_val),
                        "no_improve_evals": int(no_improve_evals),
                        "n_evals_done": int(n_evals_done),
                        "best_ckpt_update": int(best_ckpt_update),
                    }
                }
                save_ckpt(BEST_CKPT, model, opt, scaler, cfg, update, val_loss, extra=best_extra)
            else:
                if n_evals_done > ES_WARMUP_EVALS:
                    no_improve_evals += 1

            last_extra = {
                "early_stopping": {
                    "best_val": float(best_val),
                    "no_improve_evals": int(no_improve_evals),
                    "n_evals_done": int(n_evals_done),
                    "best_ckpt_update": int(best_ckpt_update),
                }
            }
            save_ckpt(RESUME_CKPT, model, opt, scaler, cfg, update, val_loss, extra=last_extra)

            if EARLY_STOPPING and (n_evals_done > ES_WARMUP_EVALS) and (no_improve_evals >= ES_PATIENCE_EVALS):
                print(
                    f"[EARLY_STOP] Stop at update={update} | best_val={best_val:.6f} "
                    f"| patience={ES_PATIENCE_EVALS} reached"
                )
                break

        elif (update > 0) and (update % SAVE_EVERY == 0):
            save_ckpt(
                RESUME_CKPT,
                model,
                opt,
                scaler,
                cfg,
                update,
                last_val_loss if not math.isnan(last_val_loss) else float("inf"),
                extra={
                    "early_stopping": {
                        "best_val": float(best_val),
                        "no_improve_evals": int(no_improve_evals),
                        "n_evals_done": int(n_evals_done),
                        "best_ckpt_update": int(best_ckpt_update),
                    }
                },
            )

        update += 1
    else:
        print(f"[DONE] Reached total_updates={total_updates}")
        return

    print("[DONE] Training finished by early stopping.")


if __name__ == "__main__":
    main()
