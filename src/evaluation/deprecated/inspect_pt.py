from pathlib import Path
import torch

from src.model.model import MusicTransformerGPTlike, MTModelConfig


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

# Ruta del modelo .pt
CKPT_PATH = Path(r"../../../output/checkpoints/pretraining/best.pt")

# "cpu" suele ser suficiente para inspeccionar
DEVICE = "cpu"

# True si quieres ver también las primeras claves del state_dict
SHOW_STATE_KEYS = True


# =============================================================================
# CÓDIGO
# =============================================================================

def format_num_params(model: torch.nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters())
    return f"{n:,}"


def main():
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

    # Reconstrucción del modelo desde la cfg del checkpoint
    cfg = MTModelConfig(**cfg_dict)
    model = MusicTransformerGPTlike(cfg)

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


if __name__ == "__main__":
    main()