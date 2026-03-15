from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch

from src.model.model import MusicTransformerGPTlike, MTModelConfig

# =============================================================================
# CONFIGURACIÓN (ejecución local / PyCharm)
# -----------------------------------------------------------------------------
# En esta sección definimos rutas, ratios de split, caché binario y los
# hiperparámetros principales del modelo y del entrenamiento. La idea es
# mantener todas las constantes en un único lugar para poder reproducir
# experimentos y minimizar cambios dispersos en el código.
# =============================================================================

# 1) Rutas
INDEX_CSV = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\data\interim\debug_dataset\index.csv")
TOKENS_DIR = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\data\interim\tokenized_json_bpe")

# ANCHOR: fragmento de ruta usado para “rebasar” paths del CSV y reconstruirlos
# dentro de TOKENS_DIR. Esto es útil cuando el CSV fue generado en otra máquina
# o con un root distinto, pero mantiene un subpath reconocible.
ANCHOR = r"data\interim\tokenized_json_bpe"

# Campo del JSON donde se encuentran los ids tokenizados.
TOKEN_FIELD = "ids"  # o "ids_encoded"
VOCAB_SIZE = 30000

# 2) Split de dataset
# Reservamos una fracción pequeña para validación y test, manteniendo el grueso
# para entrenamiento. El seed permite reproducibilidad.
VAL_RATIO = 0.01
TEST_RATIO = 0.01
SEED = 100454434

# 3) Caché binario (memmap)
# Esto permite convertir muchos JSON en un stream 1D concatenado (train/val/test) para
# entrenar eficientemente en memoria con recortes aleatorios (random crops).
CACHE_DIR = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\data\processed\pretraining").resolve()
ADD_EOS = True
EOS_ID = 2
USE_UINT16 = True  # vocab 30k cabe en uint16

# 4) Hiperparámetros del modelo
BLOCK_SIZE = 1024
D_MODEL = 512
N_HEADS = 8
N_LAYER = 8
DROPOUT = 0.1
D_FF = None
TIE_WEIGHTS = True
USE_FINAL_LN = True

# 5) Hiperparámetros de entrenamiento
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MICRO_BATCH = 4
GRAD_ACCUM = 8

LR = 3e-4   # learning rate, elegido para AdamW de acuerdo con la constante de Karpathy
MIN_LR = 3e-5
WARMUP_UPDATES = 1000   # permite no "arrancar demasiado fuerte" los pesos de AdamW
WEIGHT_DECAY = 0.1      # regularización sobre los pesos
GRAD_CLIP = 1.0

EPOCHS = 3  # “epochs de tokens” sobre train.bin

EVAL_EVERY = 1000
EVAL_BATCHES = 200

SAVE_EVERY = 500
CKPT_DIR = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\output\checkpoints").resolve()

NUM_WORKERS = 2
PIN_MEMORY = True

USE_AMP = True      # Automatic Mixed Precision: para acelerar el entrenamiento sin sacrificar mucha precisión
AMP_DTYPE = "bf16"  # "bf16" o "fp16"


# =============================================================================
# Funciones generales
# =============================================================================

def seed_all(seed: int):
    """
    En esta función fijamos las semillas de Python, NumPy y PyTorch para
    mejorar la reproducibilidad de los experimentos (shuffle, crops aleatorios,
    inicializaciones, etc.).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def choose_np_dtype(use_uint16: bool, vocab_size: int):
    """
    En esta función seleccionamos el dtype del memmap:
    - uint16 si el vocabulario cabe (ahorra disco/IO),
    - uint32 en caso contrario (evita overflow).
    """
    if use_uint16:
        if vocab_size >= 65535:
            raise ValueError("VOCAB_SIZE no cabe en uint16; usa uint32 (USE_UINT16=False).")
        return np.uint16
    return np.uint32

def split_train_val_test(paths: List[Path], val_ratio: float, test_ratio: float, seed: int):
    """
    En esta función barajamos la lista de ficheros y generamos un split reproducible
    en train/val/test. Forzamos un mínimo de 1 fichero en val y test para evitar
    conjuntos vacíos en datasets pequeños.
    """
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


def  rebase_path(abs_path: str, tokens_dir: Path, anchor: str) -> Path:
    """
    En esta función transformamos un path absoluto (típicamente guardado en index.csv)
    en un path equivalente dentro de TOKENS_DIR.

    La estrategia es:
      1) buscar el substring `anchor` dentro del path,
      2) tomar la parte posterior a `anchor` como ruta relativa,
      3) “repegarla” a `tokens_dir`.

    Si no se encuentra el anchor, devolvemos el path original tal cual.
    """
    s = abs_path.replace("\\", "/")
    a = anchor.replace("\\", "/")
    pos = s.find(a)
    if pos == -1:
        return Path(abs_path)

    rel_part = s[pos + len(a):].lstrip("/")  # parte dentro de tokens_json_bpe
    return tokens_dir / rel_part


def resolve_json_paths(index_csv: Path, tokens_dir: Path, anchor: str) -> List[Path]:
    """
    En esta función resolvemos la lista real de JSON tokenizados, contemplando
    escenarios típicos de “paths rotos” (p. ej., al mover el proyecto).

    Probamos tres estrategias, por orden:
      1) usar los paths del CSV tal cual,
      2) rebasar con TOKENS_DIR + ANCHOR,
      3) si lo anterior falla, escanear TOKENS_DIR recursivamente.

    Esta lógica evita bloqueos por rutas absolutas obsoletas y permite continuar
    el entrenamiento sin generar de nuevo el index.
    """
    if not index_csv.exists():
        raise FileNotFoundError(f"INDEX_CSV no existe: {index_csv}")

    df = pd.read_csv(index_csv)
    if "path" not in df.columns:
        raise ValueError("index.csv debe tener columna 'path'.")

    raw_paths = df["path"].tolist()

    # 1) Uso directo
    paths1 = [Path(p) for p in raw_paths]
    exist1 = [p for p in paths1 if p.exists()]
    if len(exist1) > 0:
        print(f"[DATA] paths OK (tal cual): {len(exist1)}")
        return exist1

    print("[DATA][WARN] 0 paths existentes usando rutas absolutas del CSV. Intento rebase...")

    # 2) Rebase
    if tokens_dir.exists():
        paths2 = [rebase_path(p, tokens_dir, anchor) for p in raw_paths]
        exist2 = [p for p in paths2 if p.exists()]
        if len(exist2) > 0:
            print(f"[DATA] paths OK (rebase): {len(exist2)}")
            return exist2

    print("[DATA][WARN] 0 paths existentes tras rebase. Fallback: escaneo TOKENS_DIR...")

    # 3) Scan
    if not tokens_dir.exists():
        raise FileNotFoundError(f"TOKENS_DIR no existe: {tokens_dir}")
    scan = sorted([p for p in tokens_dir.rglob("*.json") if p.is_file()])
    print(f"[DATA] paths OK (scan): {len(scan)}")
    return scan


def file_size_multiple_of_dtype(path: Path, dtype) -> bool:
    """
    En esta función comprobamos que el tamaño del binario sea múltiplo del tamaño
    del dtype usado en el memmap. Si no lo es, asumimos que el archivo está corrupto
    o fue construido con un dtype distinto.
    """
    if not path.exists():
        return False
    size = path.stat().st_size
    return (size % np.dtype(dtype).itemsize) == 0


def safe_remove(path: Path):
    """
    En esta función eliminamos un archivo si existe, evitando excepciones por
    inexistencia. Se usa principalmente para reconstruir bins corruptos.
    """
    if path.exists():
        path.unlink()


def build_memmap(files: List[Path], out_bin: Path, token_field: str, dtype, add_eos: bool, eos_id: int) -> int:
    """
    En esta función construimos un stream 1D concatenando tokens de muchos JSON.

    La motivación es doble:
      - Reducimos el coste de abrir/parsear JSON constantemente durante entrenamiento.
      - Convertimos el corpus en un array contiguo (memmap) para muestrear recortes
        aleatorios de longitud fija (block_size).

    El resultado es un archivo binario `out_bin` que se accede con np.memmap.
    Devolvemos el total de tokens escritos (incluyendo EOS si se añade).
    """
    if len(files) == 0:
        raise ValueError("No hay ficheros para construir el memmap (lista vacía).")

    # 1) Pasada de conteo: calculamos el tamaño final del stream (para crear memmap).
    total = 0
    for p in files:
        obj = json.loads(p.read_text(encoding="utf-8"))
        ids = obj.get(token_field, None)
        if not ids:
            continue
        total += len(ids) + (1 if add_eos else 0)

    if total <= 0:
        raise ValueError("Total tokens = 0. ¿TOKEN_FIELD correcto? ¿JSON vacíos?")

    # 2) Creamos el memmap con el tamaño final y vamos escribiendo secuencialmente.
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

        if i % 2000 == 0:
            print(f"[cache] {i}/{len(files)} escritos | tokens={w:,}")

    mm.flush()
    assert w == total
    print(f"[cache] OK -> {out_bin.name} | total_tokens={total:,}")
    return total

# =============================================================================
# Dataset: random crops sobre un stream memmap
# -----------------------------------------------------------------------------
# En lugar de iterar “canción a canción”, entrenamos como un LM clásico sobre un
# stream continuo de tokens. Cada sample toma un recorte aleatorio de longitud
# block_size+1, y produce:
#   x = chunk[:-1]
#   y = chunk[1:]
# para next-token prediction.
# =============================================================================
class MemmapRandomCropDataset(torch.utils.data.Dataset):
    def __init__(self, bin_path: Path, block_size: int, dtype):
        if not bin_path.exists():
            raise FileNotFoundError(f"Bin no existe: {bin_path}")
        if not file_size_multiple_of_dtype(bin_path, dtype):
            raise ValueError(f"El bin {bin_path} está corrupto o no coincide con dtype={dtype}. "
                             f"Borra CACHE_DIR y reconstruye.")
        self.data = np.memmap(bin_path, mode="r", dtype=dtype)
        self.n = int(self.data.shape[0])
        self.block_size = block_size

        # Necesitamos al menos block_size+1 tokens para construir (x,y).
        if self.n < block_size + 1:
            raise ValueError(f"Stream demasiado corto ({self.n}) para block_size={block_size}")
        self.max_start = self.n - (block_size + 1)

    def __len__(self):
        # El dataset es “virtual”: no está indexado por canciones sino por samples
        # aleatorios. Definimos un tamaño grande para que el DataLoader pueda iterar
        # indefinidamente sin agotar el dataset.
        return 10_000_000  # virtual

    def __getitem__(self, idx):
        start = random.randint(0, self.max_start)
        chunk = np.asarray(self.data[start:start + self.block_size + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y

# =============================================================================
# Optimizador y schedule de learning rate
# =============================================================================
def configure_optimizer(model: torch.nn.Module, lr: float, weight_decay: float):
    """
    En esta función construimos AdamW separando parámetros con y sin weight decay.

    Es una práctica común excluir de la regularización:
      - biases
      - LayerNorm (y afines)
      - embeddings
    porque el weight decay en estos parámetros suele ser contraproducente.
    """
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
        lr=lr, betas=(0.9, 0.95), eps=1e-8
    )


def lr_schedule(update: int, total_updates: int, base_lr: float, min_lr: float, warmup: int):
    """
    En esta función aplicamos un schedule típico en Transformers:
      - warmup lineal durante `warmup` updates,
      - luego decaimiento cosenoidal hasta `min_lr`.

    Este patrón (warmup + cosine) es frecuente en entrenamiento de modelos grandes
    porque reduce inestabilidades iniciales y mejora convergencia.
    """
    if update < warmup:
        return base_lr * (update + 1) / max(1, warmup)
    t = (update - warmup) / max(1, total_updates - warmup)
    t = min(max(t, 0.0), 1.0)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * t)) * (base_lr - min_lr)


@torch.no_grad()
def evaluate(model, loader, device: str, max_batches: int):
    """
    En esta función evaluamos el modelo en un número acotado de batches para
    obtener una estimación rápida de la loss media.

    Se hace en no_grad() para ahorrar memoria y tiempo, alternando model.eval()
    y model.train() para respetar dropout y LayerNorm.
    """
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


def save_ckpt(path: Path, model, opt, scaler, cfg: MTModelConfig, update: int, val_loss: float):
    """
    En esta función serializamos un checkpoint autocontenido:
      - estado del modelo
      - estado del optimizador
      - estado del scaler (si se usa fp16)
      - configuración usada
      - update actual y métrica de validación

    Esto permite reanudar entrenamiento y además conservar el “best.pt” para test.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "update": update,
        "val_loss": val_loss,
        "cfg": cfg.__dict__,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
    }, path)
    print(f"[CKPT] saved: {path.name} | update={update} val_loss={val_loss:.4f}")

# =============================================================================
# Preparación de caché + splits
# -----------------------------------------------------------------------------
# Esta parte crea (si hace falta) los bins train/val/test y un meta.json con
# estadísticas. La construcción se hace una sola vez y luego se reutiliza.
# =============================================================================
def prepare_cache_and_splits() -> Dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    dtype = choose_np_dtype(USE_UINT16, VOCAB_SIZE)

    train_bin = CACHE_DIR / "train.bin"
    val_bin = CACHE_DIR / "val.bin"
    test_bin = CACHE_DIR / "test.bin"
    meta_json = CACHE_DIR / "meta.json"

    # Resolvemos paths reales
    paths = resolve_json_paths(INDEX_CSV, TOKENS_DIR, ANCHOR)
    print(f"[DATA] json existentes: {len(paths)}")
    if len(paths) == 0:
        raise RuntimeError("No se ha encontrado ningún JSON. Revisa INDEX_CSV/TOKENS_DIR/ANCHOR.")

    train_files, val_files, test_files = split_train_val_test(paths, VAL_RATIO, TEST_RATIO, SEED)
    print(f"[DATA] split: train={len(train_files)} val={len(val_files)} test={len(test_files)}")

    # Si detectamos bins incompatibles con el dtype esperado, los eliminamos para
    # reconstruirlos de forma limpia (evita errores de lectura por memmap).
    for b in [train_bin, val_bin, test_bin]:
        if b.exists() and not file_size_multiple_of_dtype(b, dtype):
            print(f"[cache][WARN] {b.name} no coincide con dtype -> borrando para reconstruir")
            safe_remove(b)

    # Construimos el caché si falta cualquiera de los bins o el meta.json.
    if not train_bin.exists() or not val_bin.exists() or not test_bin.exists() or not meta_json.exists():
        print("[cache] Construyendo memmaps (esto se hace 1 vez)...")
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
        print(f"[cache] Reusando cache | train_tokens={meta['train_tokens']:,} val_tokens={meta['val_tokens']:,} test_tokens={meta['test_tokens']:,}")

    return {
        "dtype": dtype,
        "train_bin": train_bin,
        "val_bin": val_bin,
        "test_bin": test_bin,
        "meta": meta,
    }

# =============================================================================
# Main: entrenamiento completo
# -----------------------------------------------------------------------------
# En main se orquesta:
#  - seeding
#  - preparación de cache + loaders
#  - construcción del modelo + optimizador
#  - bucle de entrenamiento con grad accumulation, clipping, AMP
#  - evaluaciones periódicas y checkpoints (last / best)
#  - evaluación final en test usando best.pt
# =============================================================================
def main():
    seed_all(SEED)
    print(f"[ENV] device={DEVICE} torch={torch.__version__}")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] CKPT_DIR = {CKPT_DIR}")
    print(f"[INFO] CACHE_DIR = {CACHE_DIR}")

    cache = prepare_cache_and_splits()
    dtype = cache["dtype"]

    # Datasets/Loaders
    train_ds = MemmapRandomCropDataset(cache["train_bin"], BLOCK_SIZE, dtype)
    val_ds = MemmapRandomCropDataset(cache["val_bin"], BLOCK_SIZE, dtype)
    test_ds = MemmapRandomCropDataset(cache["test_bin"], BLOCK_SIZE, dtype)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=MICRO_BATCH, num_workers=NUM_WORKERS,
        pin_memory=(PIN_MEMORY and DEVICE == "cuda")
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=MICRO_BATCH, num_workers=NUM_WORKERS,
        pin_memory=(PIN_MEMORY and DEVICE == "cuda")
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=MICRO_BATCH, num_workers=NUM_WORKERS,
        pin_memory=(PIN_MEMORY and DEVICE == "cuda")
    )

    # Modelo
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

    # AMP: configuramos autocast dtype y scaler (si fp16).
    scaler = None
    autocast_dtype = None
    if DEVICE == "cuda" and USE_AMP:
        if AMP_DTYPE == "bf16":
            autocast_dtype = torch.bfloat16
            scaler = None
        else:
            autocast_dtype = torch.float16
            scaler = torch.cuda.amp.GradScaler()

    # Plan por tokens (usa el corpus de train.bin entero)
    train_tokens = int(cache["meta"]["train_tokens"])
    tokens_per_update = MICRO_BATCH * BLOCK_SIZE * GRAD_ACCUM
    updates_per_epoch = math.ceil(train_tokens / tokens_per_update)
    total_updates = updates_per_epoch * EPOCHS

    print(f"[PLAN] train_tokens={train_tokens:,}")
    print(f"[PLAN] tokens/update={tokens_per_update:,} (micro={MICRO_BATCH}, block={BLOCK_SIZE}, accum={GRAD_ACCUM})")
    print(f"[PLAN] updates/epoch≈{updates_per_epoch} epochs={EPOCHS} total_updates={total_updates}")

    # Bucle de entrenamiento.
    model.train()
    best_val = float("inf")
    t0 = time.time()
    update = 0
    train_iter = iter(train_loader)

    # Aqui comienza el train + val
    while update < total_updates:
        # Ajustamos LR por schedule (warmup + cosine) en cada update.
        lr = lr_schedule(update, total_updates, LR, MIN_LR, WARMUP_UPDATES)
        for pg in opt.param_groups:
            pg["lr"] = lr

        # Grad accumulation: acumulamos GRAD_ACCUM micro-batches antes de step().
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

            # Forward en autocast (automatic mixed precision) si procede; normalizamos loss por acumulación
            # para que el gradiente sea equivalente a un batch grande.
            if DEVICE == "cuda" and USE_AMP and autocast_dtype is not None:
                with torch.amp.autocast("cuda"):
                    _, loss = model(x, y)
                    loss = loss / GRAD_ACCUM
            else:
                _, loss = model(x, y)
                loss = loss / GRAD_ACCUM    # esto se usa para la loss media más adelante

            # Backprop con scaler si fp16; si bf16 o CPU, backward estándar.
            # Esto permite sumar las losses
            if scaler is not None:  # Con el scaler, se multiplica temporalmente la loss por un factor grande antes del backward para que los gradientes no desaparezcan al representarse en float16
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += float(loss.item())

        # Antes del step aplicamos clipping para estabilizar el entrenamiento. Sirve para escalar el gradiente y que
        # no tenga un valor demasiado grande que afecte negativamente al entrenamiento
        if scaler is not None:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

        # Log periódico de progreso y throughput aproximado.
        if update % 50 == 0:
            elapsed = time.time() - t0
            tokens_seen = (update + 1) * tokens_per_update
            tok_s = tokens_seen / max(elapsed, 1e-9)
            print(f"[upd {update:>6}/{total_updates}] loss={accum_loss:.4f} lr={lr:.3e} tokens_seen~{tokens_seen:,} tok/s~{tok_s:,.0f}")

        # Guardado periódico del estado “last” para tolerancia a fallos.
        if update > 0 and update % SAVE_EVERY == 0:
            save_ckpt(CKPT_DIR / "last.pt", model, opt, scaler, cfg, update, val_loss=float("nan"))

        # Evaluación periódica en validación: aquí decidimos “best”.
        if update > 0 and update % EVAL_EVERY == 0:
            val_loss = evaluate(model, val_loader, DEVICE, max_batches=EVAL_BATCHES)
            print(f"[VAL] update={update} val_loss={val_loss:.4f}")
            save_ckpt(CKPT_DIR / "last.pt", model, opt, scaler, cfg, update, val_loss=val_loss)
            if val_loss < best_val:
                best_val = val_loss
                save_ckpt(CKPT_DIR / "best.pt", model, opt, scaler, cfg, update, val_loss=val_loss)

        update += 1

    print("[TRAIN DONE]")

    # Evaluación final en test:
    # Usamos best.pt (seleccionado por validación) y evaluamos una sola vez.
    best_path = CKPT_DIR / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        test_loss = evaluate(model, test_loader, DEVICE, max_batches=EVAL_BATCHES)
        print(f"[TEST] loss={test_loss:.4f} (evaluado con best.pt)")
    else:
        print("[TEST][WARN] No existe best.pt; evalúo test con el modelo final.")
        test_loss = evaluate(model, test_loader, DEVICE, max_batches=EVAL_BATCHES)
        print(f"[TEST] loss={test_loss:.4f}")

    print("[DONE] pretraining finished.")

if __name__ == "__main__":
    main()