from pathlib import Path
import torch

# Ruta fija a tu checkpoint
CKPT_PATH = Path(r"/output/checkpoints/best.pt")


def format_num(n: int) -> str:
    return f"{n:,}"


def main():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"No existe el checkpoint: {CKPT_PATH}")

    print(f"[INFO] Cargando checkpoint:\n{CKPT_PATH}\n")

    # IMPORTANTE: cargar en CPU para evitar problemas
    ckpt = torch.load(CKPT_PATH, map_location="cpu")

    print("=" * 80)
    print("CLAVES DEL CHECKPOINT")
    print("=" * 80)
    print(list(ckpt.keys()))

    print("\n" + "=" * 80)
    print("METADATOS")
    print("=" * 80)
    print(f"update: {ckpt.get('update', 'N/A')}")
    print(f"val_loss: {ckpt.get('val_loss', 'N/A')}")

    # =========================
    # CONFIG
    # =========================
    cfg = ckpt.get("cfg", {})
    print("\n" + "=" * 80)
    print("CONFIG (cfg)")
    print("=" * 80)

    if cfg:
        for k in sorted(cfg.keys()):
            print(f"{k}: {cfg[k]}")
    else:
        print("No hay cfg")

    # =========================
    # MODELO
    # =========================
    model_state = ckpt.get("model", {})

    print("\n" + "=" * 80)
    print("MODELO (state_dict)")
    print("=" * 80)

    total_params = 0

    for name, tensor in model_state.items():
        numel = tensor.numel()
        total_params += numel
        print(f"{name:<60} {str(tuple(tensor.shape)):<20} {format_num(numel)}")

    print("\n" + "=" * 80)
    print("RESUMEN MODELO")
    print("=" * 80)
    print(f"Tensores: {len(model_state)}")
    print(f"Parámetros totales: {format_num(total_params)}")

    # =========================
    # OPTIMIZADOR
    # =========================
    opt = ckpt.get("optimizer", {})

    print("\n" + "=" * 80)
    print("OPTIMIZADOR")
    print("=" * 80)

    if opt:
        print("param_groups:", len(opt.get("param_groups", [])))
        print("state entries:", len(opt.get("state", {})))
    else:
        print("No hay optimizer")

    # =========================
    # SCALER (AMP)
    # =========================
    scaler = ckpt.get("scaler", None)

    print("\n" + "=" * 80)
    print("SCALER AMP")
    print("=" * 80)

    if scaler is None:
        print("No usado")
    else:
        print("Presente")


if __name__ == "__main__":
    main()