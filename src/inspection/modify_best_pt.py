"""
Ofrece utilidades de inspeccion para revisar tokens, checkpoints o salidas intermedias.

Estos scripts ayudan a auditar el comportamiento del pipeline durante el desarrollo.
"""

import torch
from pathlib import Path

src = Path(r"../../output/checkpoints/pretraining_v2/last.pt")
dst = src.with_name("last.pt")

ckpt = torch.load(src, map_location="cpu")
print("Valores previos:", ckpt.get("update"), ckpt.get("val_loss"))

ckpt["update"] = 44500   # update que debe quedar reflejado en el checkpoint

torch.save(ckpt, dst)
print("Guardado:", dst)
