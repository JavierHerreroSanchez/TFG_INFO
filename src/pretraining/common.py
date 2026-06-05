"""
Gestiona una fase de preentrenamiento del modelo sobre el corpus tokenizado.

Incluye configuracion de datos, modelo, optimizacion y guardado de checkpoints para poder continuar o evaluar los experimentos del TFG.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.model.model import MTModelConfig


@dataclass(frozen=True)
class CacheConfig:
    """
    Parametros necesarios para convertir JSON tokenizados en bins reutilizables.

    La misma estructura sirve para pretraining y finetuning. Las diferencias entre
    ambos flujos (por ejemplo, anadir BOS en pretraining v2 pero no en finetuning)
    se expresan con flags para no duplicar todo el codigo de cache.
    """

    index_csv: Path
    tokens_dir: Path
    anchor: str
    token_field: str
    vocab_size: int
    cache_dir: Path
    block_size: int
    val_ratio: float
    test_ratio: float
    seed: int
    use_uint16: bool = True
    add_bos: bool = False
    bos_id: int = 1
    add_eos: bool = True
    eos_id: int = 2
    progress_every: int = 1000


def seed_all(seed: int) -> None:
    """Fija semillas de Python, NumPy y PyTorch para hacer reproducibles los splits y crops."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_np_dtype(use_uint16: bool, vocab_size: int):
    """Elige el dtype del memmap según el tamaño del vocabulario."""
    if use_uint16:
        if vocab_size >= 65535:
            raise ValueError("VOCAB_SIZE no cabe en uint16; usa uint32.")
        return np.uint16
    return np.uint32


def split_train_val_test(paths: List[Path], val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[Path], List[Path], List[Path]]:
    """Baraja una lista de ficheros y devuelve un split reproducible train/val/test."""
    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))

    test_paths = shuffled[:n_test]
    val_paths = shuffled[n_test:n_test + n_val]
    train_paths = shuffled[n_test + n_val:]
    return train_paths, val_paths, test_paths


def rebase_path(abs_path: str, tokens_dir: Path, anchor: str) -> Path:
    """Reconstruye rutas del CSV cuando el proyecto se ha movido de máquina o carpeta."""
    source = abs_path.replace("\\", "/")
    marker = anchor.replace("\\", "/")
    pos = source.find(marker)
    if pos == -1:
        return Path(abs_path)

    rel_part = source[pos + len(marker):].lstrip("/")
    return tokens_dir / rel_part


def resolve_json_paths(index_csv: Path, tokens_dir: Path, anchor: str, path_column: str = "path") -> List[Path]:
    """Resuelve los JSON tokenizados con tres estrategias: ruta directa, rebase y escaneo."""
    if not index_csv.exists():
        raise FileNotFoundError(f"INDEX_CSV no existe: {index_csv}")

    df = pd.read_csv(index_csv)
    if path_column not in df.columns:
        raise ValueError(f"{index_csv.name} debe tener columna '{path_column}'.")

    raw_paths = df[path_column].tolist()

    direct_paths = [Path(p) for p in raw_paths]
    existing_direct = [p for p in direct_paths if p.exists()]
    if existing_direct:
        print(f"[DATA] paths OK (tal cual): {len(existing_direct)}")
        return existing_direct

    print("[DATA][WARN] 0 paths existentes usando rutas del CSV. Intento rebase...")
    if tokens_dir.exists():
        rebased_paths = [rebase_path(p, tokens_dir, anchor) for p in raw_paths]
        existing_rebased = [p for p in rebased_paths if p.exists()]
        if existing_rebased:
            print(f"[DATA] paths OK (rebase): {len(existing_rebased)}")
            return existing_rebased

    print("[DATA][WARN] 0 paths existentes tras rebase. Fallback: escaneo TOKENS_DIR...")
    if not tokens_dir.exists():
        raise FileNotFoundError(f"TOKENS_DIR no existe: {tokens_dir}")

    scanned = sorted(p for p in tokens_dir.rglob("*.json") if p.is_file())
    print(f"[DATA] paths OK (scan): {len(scanned)}")
    return scanned


def file_size_multiple_of_dtype(path: Path, dtype) -> bool:
    """Verifica que un binario puede leerse con el dtype esperado."""
    if not path.exists():
        return False
    return (path.stat().st_size % np.dtype(dtype).itemsize) == 0


def safe_remove(path: Path) -> None:

    if path.exists():
        path.unlink()


def normalize_token_ids(ids, add_bos: bool, bos_id: int, add_eos: bool, eos_id: int) -> List[int]:
    """Aplica BOS/EOS sin duplicarlos si ya venian en el JSON."""
    tokens = list(ids)
    if add_bos and tokens and tokens[0] == bos_id:
        tokens = tokens[1:]
    if add_eos and tokens and tokens[-1] == eos_id:
        tokens = tokens[:-1]
    if add_bos:
        tokens.insert(0, bos_id)
    if add_eos:
        tokens.append(eos_id)
    return tokens


def build_memmap(
    files: List[Path],
    out_bin: Path,
    token_field: str,
    dtype,
    add_bos: bool = False,
    bos_id: int = 1,
    add_eos: bool = True,
    eos_id: int = 2,
    progress_every: int = 1000,
) -> int:
    """Concatena muchos JSON tokenizados en un unico stream memmap.

    El entrenamiento toma crops aleatorios del stream, asi que este paso evita
    abrir miles de JSON en cada epoch y mantiene el acceso a datos como IO secuencial.
    """

    if not files:
        raise ValueError("No hay ficheros para construir el memmap.")

    total = 0
    for path in files:
        obj = json.loads(path.read_text(encoding="utf-8"))
        ids = obj.get(token_field)
        if ids:
            total += len(normalize_token_ids(ids, add_bos, bos_id, add_eos, eos_id))

    if total <= 0:
        raise ValueError("Total tokens = 0. Comprobar TOKEN_FIELD y JSON vacios.")

    mm = np.memmap(out_bin, mode="w+", dtype=dtype, shape=(total,))
    written = 0

    for i, path in enumerate(files, start=1):
        obj = json.loads(path.read_text(encoding="utf-8"))
        ids = obj.get(token_field)
        if not ids:
            continue

        arr = np.asarray(normalize_token_ids(ids, add_bos, bos_id, add_eos, eos_id), dtype=dtype)
        mm[written:written + len(arr)] = arr
        written += len(arr)

        if progress_every and i % progress_every == 0:
            print(f"[cache] {i}/{len(files)} escritos | tokens={written:,}")

    mm.flush()
    assert written == total
    print(f"[cache] OK -> {out_bin.name} | total_tokens={total:,}")
    return total


class MemmapRandomCropDataset(torch.utils.data.Dataset):
    """Dataset virtual que samplea ventanas aleatorias de longitud block_size + 1."""

    def __init__(self, bin_path: Path, block_size: int, dtype, virtual_size: int = 1_000_000):

        if not bin_path.exists():
            raise FileNotFoundError(f"Bin no existe: {bin_path}")
        if not file_size_multiple_of_dtype(bin_path, dtype):
            raise ValueError(f"Bin corrupto o dtype incompatible: {bin_path}")

        self.data = np.memmap(bin_path, mode="r", dtype=dtype)
        self.n = int(self.data.shape[0])
        self.block_size = block_size
        self.virtual_size = virtual_size

        if self.n < block_size + 1:
            raise ValueError(f"Stream demasiado corto ({self.n}) para block_size={block_size}")
        self.max_start = self.n - (block_size + 1)

    def __len__(self) -> int:

        return self.virtual_size

    def __getitem__(self, idx):

        del idx
        start = random.randint(0, self.max_start)
        chunk = np.asarray(self.data[start:start + self.block_size + 1], dtype=np.int64)
        return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])


def prepare_cache_and_splits(config: CacheConfig) -> Dict:
    """Prepara train.bin, val.bin, test.bin y meta.json para un experimento."""
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    dtype = choose_np_dtype(config.use_uint16, config.vocab_size)

    train_bin = config.cache_dir / "train.bin"
    val_bin = config.cache_dir / "val.bin"
    test_bin = config.cache_dir / "test.bin"
    meta_json = config.cache_dir / "meta.json"

    paths = resolve_json_paths(config.index_csv, config.tokens_dir, config.anchor)
    print(f"[DATA] json existentes: {len(paths)}")
    if not paths:
        raise RuntimeError("No se ha encontrado ningun JSON. Comprobar INDEX_CSV/TOKENS_DIR/ANCHOR.")

    train_files, val_files, test_files = split_train_val_test(paths, config.val_ratio, config.test_ratio, config.seed)
    print(f"[DATA] split: train={len(train_files)} val={len(val_files)} test={len(test_files)}")

    for bin_path in [train_bin, val_bin, test_bin]:
        if bin_path.exists() and not file_size_multiple_of_dtype(bin_path, dtype):
            print(f"[cache][WARN] {bin_path.name} no coincide con dtype -> borrando para reconstruir")
            safe_remove(bin_path)

    if not train_bin.exists() or not val_bin.exists() or not test_bin.exists() or not meta_json.exists():
        print("[cache] Construyendo memmaps...")
        train_tokens = build_memmap_for_config(train_files, train_bin, config, dtype)
        val_tokens = build_memmap_for_config(val_files, val_bin, config, dtype)
        test_tokens = build_memmap_for_config(test_files, test_bin, config, dtype)

        meta = {
            "vocab_size": config.vocab_size,
            "block_size": config.block_size,
            "dtype": str(dtype),
            "train_tokens": int(train_tokens),
            "val_tokens": int(val_tokens),
            "test_tokens": int(test_tokens),
            "train_files": len(train_files),
            "val_files": len(val_files),
            "test_files": len(test_files),
            "token_field": config.token_field,
            "add_bos": config.add_bos,
            "bos_id": config.bos_id,
            "add_eos": config.add_eos,
            "eos_id": config.eos_id,
        }
        meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[cache] meta guardado: {meta_json}")
    else:
        meta = json.loads(meta_json.read_text(encoding="utf-8"))
        print(
            f"[cache] Reusando cache | train_tokens={meta['train_tokens']:,} "
            f"val_tokens={meta['val_tokens']:,} test_tokens={meta['test_tokens']:,}"
        )

    return {"dtype": dtype, "train_bin": train_bin, "val_bin": val_bin, "test_bin": test_bin, "meta": meta}


def build_memmap_for_config(files: List[Path], out_bin: Path, config: CacheConfig, dtype) -> int:
    """
    Construye una estructura auxiliar usada por el resto del flujo.

    """

    return build_memmap(
        files,
        out_bin,
        config.token_field,
        dtype,
        add_bos=config.add_bos,
        bos_id=config.bos_id,
        add_eos=config.add_eos,
        eos_id=config.eos_id,
        progress_every=config.progress_every,
    )


def make_loaders(cache: Dict, block_size: int, batch_size: int, num_workers: int, pin_memory: bool, device: str, virtual_size: int = 1_000_000):
    """Construye DataLoaders homogeneos para train, val y test."""
    dtype = cache["dtype"]
    datasets = {
        split: MemmapRandomCropDataset(cache[f"{split}_bin"], block_size, dtype, virtual_size=virtual_size)
        for split in ("train", "val", "test")
    }

    return tuple(
        torch.utils.data.DataLoader(
            datasets[split],
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(pin_memory and device == "cuda"),
        )
        for split in ("train", "val", "test")
    )


def configure_optimizer(model: torch.nn.Module, lr: float, weight_decay: float):
    """Crea AdamW separando parametros con weight decay y parametros exentos."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lower_name = name.lower()
        if lower_name.endswith("bias") or "ln" in lower_name or "layernorm" in lower_name or "embedding" in lower_name or "wte" in lower_name:
            no_decay.append(param)
        else:
            decay.append(param)

    print(f"[optim] decay_params={sum(p.numel() for p in decay):,} | no_decay_params={sum(p.numel() for p in no_decay):,}")
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def lr_schedule(update: int, total_updates: int, base_lr: float, min_lr: float, warmup: int) -> float:
    """Warmup lineal seguido de decaimiento cosenoidal."""
    if update < warmup:
        return base_lr * (update + 1) / max(1, warmup)
    t = (update - warmup) / max(1, total_updates - warmup)
    t = min(max(t, 0.0), 1.0)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * t)) * (base_lr - min_lr)


@torch.no_grad()
def evaluate(model, loader, device: str, max_batches: int) -> float:
    """Evalua una loss media aproximada usando un numero acotado de batches."""
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


def configure_amp(device: str, use_amp: bool, amp_dtype: str):
    """Configura autocast y GradScaler. En bf16 no hace falta scaler."""
    scaler = None
    autocast_dtype = None
    if device == "cuda" and use_amp:
        if amp_dtype == "bf16":
            autocast_dtype = torch.bfloat16
        else:
            autocast_dtype = torch.float16
            scaler = torch.cuda.amp.GradScaler()
    return scaler, autocast_dtype


def save_ckpt(
    path: Path,
    model,
    opt,
    scaler,
    cfg: MTModelConfig,
    update: int,
    val_loss: float,
    extra: Optional[dict] = None,
) -> None:
    """Guarda un checkpoint autocontenido para reanudar o evaluar despues."""
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
