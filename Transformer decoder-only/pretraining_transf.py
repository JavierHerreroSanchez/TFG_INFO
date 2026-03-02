from __future__ import annotations

import json
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn

from gpt_model import GPT, GPTConfig


# =========================
# CONFIG
# =========================

# Tu index.csv (el que ya tienes)
INDEX_CSV = Path(r"../debug_dataset/index.csv")  # <-- cambia a tu ruta real en tu PC

# Si en el CSV los paths son relativos, define una base:
BASE_DIR = None  # e.g. Path(r"C:\...\tokenizer\tokens_json")  o None si ya son absolutos

TOKEN_FIELD = "ids"      # "ids" o "ids_encoded"
VOCAB_SIZE = 30000       # pon el real (o max_id+1)
BLOCK_SIZE = 1024        # 1024 recomendado para empezar

# Modelo
N_LAYER = 6
D_MODEL = 256
N_HEADS = 8
DROPOUT = 0.1

# Training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1337
VAL_RATIO = 0.02

BATCH_SIZE = 8
MAX_STEPS = 5000
LR = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

PRINT_EVERY = 50
EVAL_EVERY = 500
EVAL_BATCHES = 100

CKPT_DIR = Path("../checkpoints").resolve()
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Cache simple (para no releer siempre el mismo JSON)
CACHE_MAX_FILES = 64

# =========================


def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class JsonTokenCache:
    """Cache LRU muy simple para tokens cargados desde JSON."""
    def __init__(self, max_files: int):
        self.max_files = max_files
        self.cache: Dict[str, List[int]] = {}
        self.order: List[str] = []

    def get(self, path: Path, field: str) -> List[int]:
        key = str(path)
        if key in self.cache:
            # mover al final (más reciente)
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]

        obj = json.loads(path.read_text(encoding="utf-8"))
        tokens = obj[field]
        if not isinstance(tokens, list) or (len(tokens) > 0 and not isinstance(tokens[0], int)):
            raise TypeError(f"Campo '{field}' no es lista de ints en {path}")

        self.cache[key] = tokens
        self.order.append(key)

        # expulsión LRU
        if len(self.order) > self.max_files:
            old = self.order.pop(0)
            self.cache.pop(old, None)

        return tokens


class RandomCropJsonDataset(torch.utils.data.Dataset):
    """
    Dataset autoregresivo:
      - elige un fichero aleatorio
      - toma una ventana de tamaño (block_size+1)
      - x = window[:-1], y = window[1:]
    """
    def __init__(self, paths: List[Path], block_size: int, token_field: str, cache: JsonTokenCache):
        self.paths = paths
        self.block_size = block_size
        self.token_field = token_field
        self.cache = cache

        # “longitud artificial”: puedes ajustarla; aquí usamos muchos crops por epoch
        self.virtual_len = 200_000

        print(f"[Dataset] files={len(paths)} block_size={block_size} field={token_field}")

    def __len__(self):
        return self.virtual_len

    def __getitem__(self, idx):
        # ignoramos idx y sampleamos aleatorio (minGPT style)
        while True:
            p = random.choice(self.paths)
            tokens = self.cache.get(p, self.token_field)

            # necesitamos block_size+1 para construir (x,y)
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


@torch.no_grad()
def evaluate(model: GPT, loader, device: str, max_batches: int) -> float:
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


def save_ckpt(path: Path, model: GPT, opt: torch.optim.Optimizer, step: int, val_loss: float, cfg: GPTConfig):
    payload = {
        "step": step,
        "val_loss": val_loss,
        "cfg": cfg.__dict__,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
    }
    torch.save(payload, path)
    print(f"[CKPT] saved: {path} (step={step}, val_loss={val_loss:.4f})")


def main():
    seed_all(SEED)
    print(f"[ENV] device={DEVICE} torch={torch.__version__}")

    df = pd.read_csv(INDEX_CSV)
    paths = [Path(p) for p in df["path"].tolist()]
    if BASE_DIR is not None:
        paths = [BASE_DIR / p for p in paths]

    # Filtra los que no existan (por si hay paths “viejos”)
    paths = [p for p in paths if p.exists()]
    print(f"[DATA] index paths existentes: {len(paths)}")

    train_paths, val_paths = split_train_val(paths, VAL_RATIO, SEED)
    print(f"[DATA] split train={len(train_paths)} val={len(val_paths)}")

    cache = JsonTokenCache(max_files=CACHE_MAX_FILES)

    train_ds = RandomCropJsonDataset(train_paths, BLOCK_SIZE, TOKEN_FIELD, cache)
    val_ds = RandomCropJsonDataset(val_paths, BLOCK_SIZE, TOKEN_FIELD, cache)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)

    cfg = GPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        dropout=DROPOUT,
        d_ff=None,
        use_sinusoidal_pos=False,
        tie_weights=True,
        use_final_ln=True,
        debug=True,
    )
    model = GPT(cfg).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # debug: primer batch
    x0, y0 = next(iter(train_loader))
    print(f"[DEBUG] batch x={tuple(x0.shape)} y={tuple(y0.shape)}")
    x0 = x0.to(DEVICE)
    y0 = y0.to(DEVICE)
    with torch.no_grad():
        logits0, loss0 = model(x0, y0)
    print(f"[DEBUG] logits={tuple(logits0.shape)} loss={loss0.item():.4f}")

    best_val = float("inf")
    t0 = time.time()
    model.train()

    for step, (x, y) in enumerate(train_loader):
        if step >= MAX_STEPS:
            break

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()

        if step % PRINT_EVERY == 0:
            elapsed = time.time() - t0
            toks = BATCH_SIZE * BLOCK_SIZE
            print(f"[step {step:>6}] loss={loss.item():.4f} | grad_norm={float(grad_norm):.3f} | "
                  f"tok/step~{toks} | t={elapsed:.1f}s")

        if step > 0 and step % EVAL_EVERY == 0:
            val_loss = evaluate(model, val_loader, DEVICE, max_batches=EVAL_BATCHES)
            print(f"[EVAL] step={step} val_loss={val_loss:.4f}")

            # guarda siempre last
            save_ckpt(CKPT_DIR / "last.pt", model, opt, step, val_loss, cfg)

            if val_loss < best_val:
                best_val = val_loss
                save_ckpt(CKPT_DIR / "best.pt", model, opt, step, val_loss, cfg)
                print(f"[EVAL] NEW BEST: {best_val:.4f}")

    print("[DONE] pretraining loop finished.")


if __name__ == "__main__":
    main()