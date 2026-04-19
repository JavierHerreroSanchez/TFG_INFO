import torch
from pathlib import Path

src = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\output\checkpoints\pretraining_v2\last.pt")
dst = src.with_name("last.pt")

ckpt = torch.load(src, map_location="cpu")
print("Antes:", ckpt.get("update"), ckpt.get("val_loss"))

ckpt["update"] = 44500   # o el número que quieras reflejar

torch.save(ckpt, dst)
print("Guardado:", dst)