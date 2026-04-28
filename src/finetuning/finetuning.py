from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch

from src.model.model import MusicTransformerGPTlike, MTModelConfig

# =============================================================================
# CONFIGURACIÓN DE FINETUNING
# =============================================================================

# ---- Dataset de sonatas para finetuning ----
INDEX_CSV = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\output\finetuning\finetuning_aug_index.csv")
TOKENS_DIR = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\data\interim\tokenized_finetuning")
ANCHOR = r"data\interim\tokenized_finetuning"

TOKEN_FIELD = "ids"
VOCAB_SIZE = 30000

VAL_RATIO = 0.10
TEST_RATIO = 0.10
SEED = 1453

# ---- Cache específico de finetuning ----
CACHE_DIR = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\data\bin\bin_for_finetuning").resolve()
ADD_EOS = True
EOS_ID = 2
USE_UINT16 = True

# ---- Cargar checkpoint preentrenado ----
PRETRAINED_CKPT = Path(r"../../output/checkpoints/pretraining/best.pt").resolve()

# ---- Hiperparámetros del modelo (coincidentes con el pretraining) ----
BLOCK_SIZE = 2048
D_MODEL = 512
N_HEADS = 8
N_LAYER = 8
DROPOUT = 0.1
D_FF = None
TIE_WEIGHTS = True
USE_FINAL_LN = True

# ---- Hiperparámetros de finetuning ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MICRO_BATCH = 1
GRAD_ACCUM = 16

LR = 3e-4
MIN_LR = 2e-6
WARMUP_UPDATES = 100
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0

MAX_EPOCHS = 100
MAX_UPDATES: Optional[int] = None  # si None, se calcula por epochs
EVAL_EVERY = 100
EVAL_BATCHES = 64
SAVE_EVERY = 500

# ---- Early stopping ----
EARLY_STOP = True
PATIENCE_EVALS = 12
MIN_DELTA = 1e-4
START_EARLY_AFTER = 2000

NUM_WORKERS = 2
PIN_MEMORY = True

USE_AMP = True
AMP_DTYPE = "bf16"

CKPT_DIR = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\output\checkpoints\finetuning").resolve()
SAMPLES_DIR = CKPT_DIR / "samples"

# ---- Generación de muestras fijas para escucha ----
N_LISTEN_SAMPLES = 4
LISTEN_PRIMER_TOKENS = 128
LISTEN_GEN_TOKENS = 512
GEN_TEMPERATURE = 1.0
GEN_TOP_K = 140


# =============================================================================
# UTILIDADES GENERALES
# =============================================================================

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_np_dtype(use_uint16: bool, vocab_size: int):
    if use_uint16:
        if vocab_size >= 65535:
            raise ValueError("VOCAB_SIZE no cabe en uint16; usa uint32.")
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

    print("[DATA][WARN] 0 paths existentes usando rutas del CSV. Intento rebase...")

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


def build_memmap(files: List[Path], out_bin: Path, token_field: str, dtype, add_eos: bool, eos_id: int) -> int:
    if len(files) == 0:
        raise ValueError("No hay ficheros para construir el memmap.")

    total = 0
    for p in files:
        obj = json.loads(p.read_text(encoding="utf-8"))
        ids = obj.get(token_field, None)
        if not ids:
            continue
        total += len(ids) + (1 if add_eos else 0)

    if total <= 0:
        raise ValueError("Total tokens = 0. ¿TOKEN_FIELD correcto?")

    mm = np.memmap(out_bin, mode="w+", dtype=dtype, shape=(total,))
    w = 0

    for i, p in enumerate(files, start=1):
        obj = json.loads(p.read_text(encoding="utf-8"))
        ids = obj.get(token_field, None)
        if not ids:
            continue

        arr = np.asarray(ids, dtype=dtype)
        mm[w:w + len(arr)] = arr
        w += len(arr)

        if add_eos:
            mm[w] = np.asarray([eos_id], dtype=dtype)[0]
            w += 1

        if i % 500 == 0:
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
            raise ValueError(f"Bin corrupto o dtype incompatible: {bin_path}")

        self.data = np.memmap(bin_path, mode="r", dtype=dtype)
        self.n = int(self.data.shape[0])
        self.block_size = block_size

        if self.n < block_size + 1:
            raise ValueError(f"Stream demasiado corto ({self.n}) para block_size={block_size}")

        self.max_start = self.n - (block_size + 1)

    def __len__(self):
        return 1_000_000

    def __getitem__(self, idx):
        start = random.randint(0, self.max_start)
        chunk = np.asarray(self.data[start:start + self.block_size + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


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

    print(f"[optim] decay_params={sum(p.numel() for p in decay):,} | no_decay_params={sum(p.numel() for p in no_decay):,}")

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


def load_pretrained_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: str):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No existe checkpoint preentrenado: {ckpt_path}")

    obj = torch.load(ckpt_path, map_location=device)
    state = obj["model"] if "model" in obj else obj
    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"[LOAD] checkpoint: {ckpt_path.name}")
    print(f"[LOAD] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
    if missing:
        print("[LOAD][WARN] Missing keys:")
        for k in missing[:20]:
            print("   ", k)
    if unexpected:
        print("[LOAD][WARN] Unexpected keys:")
        for k in unexpected[:20]:
            print("   ", k)


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
        raise RuntimeError("No se ha encontrado ningún JSON.")

    train_files, val_files, test_files = split_train_val_test(paths, VAL_RATIO, TEST_RATIO, SEED)
    print(f"[DATA] split: train={len(train_files)} val={len(val_files)} test={len(test_files)}")

    for b in [train_bin, val_bin, test_bin]:
        if b.exists() and not file_size_multiple_of_dtype(b, dtype):
            print(f"[cache][WARN] {b.name} no coincide con dtype -> borrando")
            safe_remove(b)

    if not train_bin.exists() or not val_bin.exists() or not test_bin.exists() or not meta_json.exists():
        print("[cache] Construyendo memmaps...")
        train_tokens = build_memmap(train_files, train_bin, TOKEN_FIELD, dtype, ADD_EOS, EOS_ID)
        val_tokens = build_memmap(val_files, val_bin, TOKEN_FIELD, dtype, ADD_EOS, EOS_ID)
        test_tokens = build_memmap(test_files, test_bin, TOKEN_FIELD, dtype, ADD_EOS, EOS_ID)

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


@torch.no_grad()
def export_listen_samples(
    model: MusicTransformerGPTlike,
    test_bin: Path,
    dtype,
    out_dir: Path,
    update: int,
    device: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.memmap(test_bin, mode="r", dtype=dtype)
    n = int(data.shape[0])

    if n < LISTEN_PRIMER_TOKENS + 1:
        print("[LISTEN][WARN] test.bin demasiado pequeño para generar muestras.")
        return

    rng = random.Random(SEED)
    starts = []
    max_start = n - (LISTEN_PRIMER_TOKENS + 1)
    for _ in range(N_LISTEN_SAMPLES):
        starts.append(rng.randint(0, max_start))

    model.eval()
    for i, start in enumerate(starts):
        primer = np.asarray(data[start:start + LISTEN_PRIMER_TOKENS], dtype=np.int64)
        x = torch.from_numpy(primer).unsqueeze(0).to(device)

        y = model.generate(
            x,
            max_new_tokens=LISTEN_GEN_TOKENS,
            temperature=GEN_TEMPERATURE,
            top_k=GEN_TOP_K
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


def main():
    seed_all(SEED)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[ENV] device={DEVICE} torch={torch.__version__}")
    print(f"[INFO] PRETRAINED_CKPT = {PRETRAINED_CKPT}")
    print(f"[INFO] CACHE_DIR = {CACHE_DIR}")
    print(f"[INFO] CKPT_DIR = {CKPT_DIR}")

    cache = prepare_cache_and_splits()
    dtype = cache["dtype"]

    train_ds = MemmapRandomCropDataset(cache["train_bin"], BLOCK_SIZE, dtype)
    val_ds = MemmapRandomCropDataset(cache["val_bin"], BLOCK_SIZE, dtype)
    test_ds = MemmapRandomCropDataset(cache["test_bin"], BLOCK_SIZE, dtype)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=MICRO_BATCH,
        num_workers=NUM_WORKERS,
        pin_memory=(PIN_MEMORY and DEVICE == "cuda"),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=MICRO_BATCH,
        num_workers=NUM_WORKERS,
        pin_memory=(PIN_MEMORY and DEVICE == "cuda"),
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=MICRO_BATCH,
        num_workers=NUM_WORKERS,
        pin_memory=(PIN_MEMORY and DEVICE == "cuda"),
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
    load_pretrained_checkpoint(model, PRETRAINED_CKPT, DEVICE)

    opt = configure_optimizer(model, LR, WEIGHT_DECAY)

    scaler = None
    autocast_dtype = None
    if DEVICE == "cuda" and USE_AMP:
        if AMP_DTYPE == "bf16":
            autocast_dtype = torch.bfloat16
            scaler = None
        else:
            autocast_dtype = torch.float16
            scaler = torch.cuda.amp.GradScaler()

    train_tokens = int(cache["meta"]["train_tokens"])
    tokens_per_update = MICRO_BATCH * BLOCK_SIZE * GRAD_ACCUM
    updates_per_epoch = max(1, math.ceil(train_tokens / tokens_per_update))
    total_updates = MAX_UPDATES if MAX_UPDATES is not None else updates_per_epoch * MAX_EPOCHS

    print(f"[PLAN] train_tokens={train_tokens:,}")
    print(f"[PLAN] tokens/update={tokens_per_update:,}")
    print(f"[PLAN] updates/epoch≈{updates_per_epoch}")
    print(f"[PLAN] total_updates={total_updates}")

    model.train()
    best_val = float("inf")
    best_update = -1
    patience_count = 0
    update = 0
    train_iter = iter(train_loader)
    t0 = time.time()

    while update < total_updates:
        lr = lr_schedule(update, total_updates, LR, MIN_LR, WARMUP_UPDATES)
        for pg in opt.param_groups:
            pg["lr"] = lr

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

        update += 1

        if update % 50 == 0:
            elapsed = time.time() - t0
            tokens_seen = update * tokens_per_update
            tok_s = tokens_seen / max(elapsed, 1e-9)

            print(
                f"[upd {update:>6}/{total_updates}] "
                f"loss={accum_loss:.4f} "
                f"lr={lr:.3e} "
                f"tokens_seen~{tokens_seen:,} "
                f"tok/s~{tok_s:,.0f} "
                f"elapsed={elapsed:.1f} sec"
            )

        if update % EVAL_EVERY == 0 or update == total_updates:
            val_loss = evaluate(model, val_loader, DEVICE, EVAL_BATCHES)
            print(f"[VAL] update={update} val_loss={val_loss:.4f}")

            save_ckpt(
                CKPT_DIR / "last.pt",
                model, opt, scaler, cfg,
                update=update,
                val_loss=val_loss,
                extra={"best_val_so_far": best_val, "best_update_so_far": best_update},
            )

            export_listen_samples(
                model=model,
                test_bin=cache["test_bin"],
                dtype=dtype,
                out_dir=SAMPLES_DIR,
                update=update,
                device=DEVICE,
            )

            improved = (best_val - val_loss) > MIN_DELTA
            if improved:
                best_val = val_loss
                best_update = update
                patience_count = 0

                save_ckpt(
                    CKPT_DIR / "best.pt",
                    model, opt, scaler, cfg,
                    update=update,
                    val_loss=val_loss,
                    extra={"best_update": best_update},
                )
                print(f"[BEST] nuevo mejor checkpoint en update={update} | val_loss={val_loss:.4f}")
            else:
                patience_count += 1
                print(f"[EARLY] sin mejora significativa ({patience_count}/{PATIENCE_EVALS})")

            if update % SAVE_EVERY == 0:
                save_ckpt(
                    CKPT_DIR / f"step_{update:06d}.pt",
                    model, opt, scaler, cfg,
                    update=update,
                    val_loss=val_loss,
                )

            if EARLY_STOP and update >= START_EARLY_AFTER and patience_count >= PATIENCE_EVALS:
                print("[EARLY] stopping activado.")
                break

    print("[DONE] finetuning terminado.")
    print(f"[DONE] best_val={best_val:.4f} @ update={best_update}")

    best_ckpt = CKPT_DIR / "best.pt"
    if best_ckpt.exists():
        print(f"[TEST] cargando mejor checkpoint: {best_ckpt.name}")
        obj = torch.load(best_ckpt, map_location=DEVICE)
        model.load_state_dict(obj["model"], strict=True)
        test_loss = evaluate(model, test_loader, DEVICE, EVAL_BATCHES)
        print(f"[TEST] test_loss={test_loss:.4f}")


if __name__ == "__main__":
    main()