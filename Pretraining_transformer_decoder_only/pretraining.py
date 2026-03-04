# pretrain_mt.py
from __future__ import annotations

import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch

from model import MusicTransformerGPT, MTModelConfig


# =========================
# CONFIG
# =========================
INDEX_CSV = Path(r"C:\Users\hersa\PycharmProjects\TFG_INFO\debug_dataset\index.csv")
TOKEN_FIELD = "ids"
VOCAB_SIZE = 30000

BLOCK_SIZE = 512          # empieza 256/512 en CPU; 1024+ mejor en GPU
BATCH_SIZE = 4

N_LAYER = 6
D_MODEL = 256
N_HEADS = 8
DROPOUT = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1337
VAL_RATIO = 0.02

LR = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

MAX_STEPS = 3000
PRINT_EVERY = 50
EVAL_EVERY = 500
EVAL_BATCHES = 50

CKPT_DIR = Path("../checkpoints").resolve()
CKPT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_MAX_FILES = 64
# =========================


def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class JsonTokenCache:
    def __init__(self, max_files: int):
        self.max_files = max_files
        self.cache: Dict[str, List[int]] = {}
        self.order: List[str] = []

    def get(self, path: Path, field: str) -> List[int]:
        key = str(path)
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]

        obj = json.loads(path.read_text(encoding="utf-8"))
        tokens = obj[field]
        if not isinstance(tokens, list) or (len(tokens) and not isinstance(tokens[0], int)):
            raise TypeError(f"Campo '{field}' no es lista de ints en {path}")

        self.cache[key] = tokens
        self.order.append(key)
        if len(self.order) > self.max_files:
            old = self.order.pop(0)
            self.cache.pop(old, None)
        return tokens


class RandomCropJsonDataset(torch.utils.data.Dataset):
    """
    window = tokens[start : start+block_size+1]
    x = window[:-1], y = window[1:]
    """
    def __init__(self, paths: List[Path], block_size: int, token_field: str, cache: JsonTokenCache):
        self.paths = paths
        self.block_size = block_size
        self.token_field = token_field
        self.cache = cache
        self.virtual_len = 200_000
        print(f"[Dataset] files={len(paths)} block_size={block_size} field={token_field}")

    def __len__(self):
        return self.virtual_len

    def __getitem__(self, idx):
        while True:
            p = random.choice(self.paths)
            tokens = self.cache.get(p, self.token_field)
            if len(tokens) < self.block_size + 1:
                continue
            start = random.randint(0, len(tokens) - (self.block_size + 1))
            window = tokens[start:start + self.block_size + 1]
            x = torch.tensor(window[:-1], dtype=torch.long)
            y = torch.tensor(window[1:], dtype=torch.long)
            return x, y


def split_train_val(paths: List[Path], val_ratio: float, seed: int) -> Tuple[List[Path], List[Path]]:
    rng = random.Random(seed)
    p = paths[:]
    rng.shuffle(p)
    n_val = max(1, int(len(p) * val_ratio))
    return p[n_val:], p[:n_val]


def configure_optimizer(model: torch.nn.Module, lr: float, weight_decay: float):
    # weight decay solo a pesos “matriciales” (práctica habitual)
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n = name.lower()
        if n.endswith("bias") or "ln" in n or "layernorm" in n or "embedding" in n or "wte" in n:
            no_decay.append(param)
        else:
            decay.append(param)

    print(f"[optim] decay_params={sum(p.numel() for p in decay):,} | no_decay_params={sum(p.numel() for p in no_decay):,}")
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr
    )


@torch.no_grad()
def evaluate(model, loader, device: str, max_batches: int) -> float:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def save_ckpt(path: Path, model, opt, step: int, val_loss: float, cfg: MTModelConfig):
    torch.save({
        "step": step,
        "val_loss": val_loss,
        "cfg": cfg.__dict__,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
    }, path)
    print(f"[CKPT] saved: {path} (step={step}, val_loss={val_loss:.4f})")


def main():
    seed_all(SEED)
    print(f"[ENV] device={DEVICE} torch={torch.__version__}")

    df = pd.read_csv(INDEX_CSV)
    if "path" not in df.columns:
        raise ValueError("index.csv debe tener columna 'path' con rutas a JSON.")

    paths = [Path(p) for p in df["path"].tolist()]
    paths = [p for p in paths if p.exists()]
    print(f"[DATA] index paths existentes: {len(paths)}")

    train_paths, val_paths = split_train_val(paths, VAL_RATIO, SEED)
    print(f"[DATA] split train={len(train_paths)} val={len(val_paths)}")

    cache = JsonTokenCache(CACHE_MAX_FILES)
    train_ds = RandomCropJsonDataset(train_paths, BLOCK_SIZE, TOKEN_FIELD, cache)
    val_ds = RandomCropJsonDataset(val_paths, BLOCK_SIZE, TOKEN_FIELD, cache)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)

    cfg = MTModelConfig(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        dropout=DROPOUT,
        d_ff=None,
        bias=True,
        tie_weights=True,
        use_final_ln=True,
        debug=False,
    )
    model = MusicTransformerGPT(cfg).to(DEVICE)
    opt = configure_optimizer(model, lr=LR, weight_decay=WEIGHT_DECAY)

    # debug primer batch
    x0, y0 = next(iter(train_loader))
    print(f"[DEBUG] batch x={tuple(x0.shape)} y={tuple(y0.shape)} | x_min={x0.min().item()} x_max={x0.max().item()}")
    if x0.max().item() >= VOCAB_SIZE:
        raise ValueError(f"Hay token_id >= vocab_size ({x0.max().item()} >= {VOCAB_SIZE}). Ajusta VOCAB_SIZE.")

    best_val = float("inf")
    t0 = time.time()
    model.train()

    for step, (x, y) in enumerate(train_loader):
        if step >= MAX_STEPS:
            break

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()

        if step % PRINT_EVERY == 0:
            elapsed = time.time() - t0
            toks = BATCH_SIZE * BLOCK_SIZE
            print(f"[step {step:>6}] loss={loss.item():.4f} grad_norm={float(grad_norm):.3f} tok/step~{toks} t={elapsed:.1f}s")

        if step > 0 and step % EVAL_EVERY == 0:
            val_loss = evaluate(model, val_loader, DEVICE, max_batches=EVAL_BATCHES)
            print(f"[EVAL] step={step} val_loss={val_loss:.4f}")

            save_ckpt(CKPT_DIR / "last.pt", model, opt, step, val_loss, cfg)
            if val_loss < best_val:
                best_val = val_loss
                save_ckpt(CKPT_DIR / "best.pt", model, opt, step, val_loss, cfg)
                print(f"[EVAL] NEW BEST: {best_val:.4f}")

    print("[DONE] pretraining finished.")


if __name__ == "__main__":
    main()