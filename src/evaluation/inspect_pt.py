"""
Utilidad de inspección de checkpoints.

Reconstruye el modelo desde la configuración guardada en el checkpoint y muestra
metadatos básicos, número de parámetros y estado serializado.
"""

from pathlib import Path

import torch

from src.model.model import MusicTransformerAutoregressive, MTModelConfig

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CKPT_PATH = PROJECT_ROOT / "output" / "checkpoints" / "pretraining_v2" / "best.pt"

DEVICE = "cpu"

SHOW_STATE_KEYS = True


# =============================================================================
# FUNCIONES
# =============================================================================

def format_num_params(model: torch.nn.Module) -> str:

    n = sum(p.numel() for p in model.parameters())
    return f"{n:,}"


def main():
    """Punto de entrada del script cuando se ejecuta desde consola."""

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"No existe el checkpoint: {CKPT_PATH.resolve()}")

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    print("=" * 90)
    print("CHECKPOINT CARGADO")
    print("=" * 90)
    print(f"Archivo cargado: {CKPT_PATH.resolve()}")
    print(f"Tipo de objeto: {type(ckpt)}")

    if not isinstance(ckpt, dict):
        print("El checkpoint no es un diccionario. No puedo inspeccionarlo con esta lógica.")
        return

    print("\nCLAVES PRINCIPALES")
    print("-" * 90)
    for k in ckpt.keys():
        print(f"  {k}")

    print("\nMETADATOS")
    print("-" * 90)
    print(f"update   : {ckpt.get('update', 'N/A')}")
    print(f"val_loss : {ckpt.get('val_loss', 'N/A')}")

    cfg_dict = ckpt.get("cfg", None)
    if cfg_dict is None:
        print("\nNo existe la clave 'cfg'.")
        return

    print("\nCONFIGURACIÓN DEL MODELO")
    print("-" * 90)
    for k, v in cfg_dict.items():
        print(f"{k}: {v}")

    # Reconstrucción del modelo desde la configuración guardada.
    cfg = MTModelConfig(**cfg_dict)
    model = MusicTransformerAutoregressive(cfg)

    state_dict = ckpt.get("model", None)
    if state_dict is None:
        print("\nNo existe la clave 'model' en el checkpoint.")
        return

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("\nRECONSTRUCCIÓN DEL MODELO")
    print("-" * 90)
    print(f"Clase           : {model.__class__.__name__}")
    print(f"Número parámetros: {format_num_params(model)}")
    print(f"block_size      : {model.cfg.block_size}")
    print(f"vocab_size      : {model.cfg.vocab_size}")
    print(f"missing_keys    : {len(missing)}")
    print(f"unexpected_keys : {len(unexpected)}")

    if missing:
        print("\nMissing keys:")
        for k in missing[:20]:
            print(f"  {k}")
        if len(missing) > 20:
            print("  ...")

    if unexpected:
        print("\nUnexpected keys:")
        for k in unexpected[:20]:
            print(f"  {k}")
        if len(unexpected) > 20:
            print("  ...")

    print("\nESTADO SERIALIZADO")
    print("-" * 90)
    print(f"Nº tensores en ckpt['model'] : {len(state_dict)}")
    print(f"Contiene optimizer           : {'optimizer' in ckpt}")
    print(f"Contiene scaler              : {'scaler' in ckpt}")

    if SHOW_STATE_KEYS:
        print("\nPRIMERAS CLAVES DEL STATE_DICT")
        print("-" * 90)
        for i, (k, v) in enumerate(state_dict.items()):
            shape = tuple(v.shape) if hasattr(v, "shape") else type(v)
            print(f"{i:03d} | {k} | {shape}")
            if i >= 29:
                print("...")
                break

    print("\n" + "=" * 90)
    print("FIN DE LA INSPECCIÓN")
    print("=" * 90)


# Ejecución directa del script.
if __name__ == "__main__":
    main()
