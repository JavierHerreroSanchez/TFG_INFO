from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

# ===== RUTAS (cámbialo si tu carpeta es otra) =====
TOKENS_DIR = Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\data\interim\tokenized_finetuning_v2").resolve()
OUT_CSV = Path(r"/output/generation_finetuning_tfg_second/audit_tokens_report.csv").resolve()
OUT_BAD = OUT_CSV.with_suffix(".bad.txt")

TOKEN_FIELD = "ids"      # si tu v5 guarda ids_encoded, pon "ids_encoded"
VOCAB_SIZE = 18000

def head_hex(b: bytes, n: int = 32) -> str:
    return b[:n].hex(" ")

def audit_one(p: Path):
    b = p.read_bytes()
    row = {
        "path": str(p),
        "size": len(b),
        "head_hex": head_hex(b),
        "status": "OK",
        "detail": "",
        "n_tokens": None,
        "min_id": None,
        "max_id": None,
        "out_of_range": None,
    }

    # Firma binaria típica (pickle/torch)
    if len(b) and b[0] == 0x80:
        row["status"] = "BIN_SIGNATURE"
        row["detail"] = "pickle/torch (0x80...)"
        return row

    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError as e:
        row["status"] = "DECODE_FAIL"
        row["detail"] = f"{e}"
        return row

    try:
        obj = json.loads(s)
    except Exception as e:
        row["status"] = "JSON_PARSE_FAIL"
        row["detail"] = f"{type(e).__name__}: {e}"
        return row

    if not isinstance(obj, dict):
        row["status"] = "NOT_DICT"
        row["detail"] = f"type={type(obj).__name__}"
        return row

    ids = obj.get(TOKEN_FIELD)
    if not isinstance(ids, list) or not ids:
        row["status"] = "BAD_TOKEN_FIELD"
        row["detail"] = f"campo '{TOKEN_FIELD}' ausente/vacío"
        return row

    mn, mx = None, None
    out_rng = False
    for x in ids:
        try:
            xi = int(x)
        except Exception:
            row["status"] = "NON_INT_TOKEN"
            row["detail"] = f"token no int: {repr(x)[:60]}"
            return row
        mn = xi if mn is None else min(mn, xi)
        mx = xi if mx is None else max(mx, xi)
        if xi < 0 or xi >= VOCAB_SIZE:
            out_rng = True

    row["n_tokens"] = len(ids)
    row["min_id"] = mn
    row["max_id"] = mx
    row["out_of_range"] = out_rng
    return row

def main():
    paths = sorted(TOKENS_DIR.rglob("*.json"))
    print("[AUDIT] TOKENS_DIR =", TOKENS_DIR)
    print("[AUDIT] json files =", len(paths))
    if not paths:
        raise RuntimeError("No hay JSON en TOKENS_DIR")

    rows = []
    bad = []
    for i,p in enumerate(paths, 1):
        r = audit_one(p)
        rows.append(r)
        if r["status"] != "OK":
            bad.append(r["path"])
        if i % 200 == 0 or i == len(paths):
            print(f"[AUDIT] {i}/{len(paths)} bad={len(bad)}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print("[AUDIT] reporte ->", OUT_CSV)

    if bad:
        OUT_BAD.write_text("\n".join(bad), encoding="utf-8")
        print("[AUDIT] bad ->", OUT_BAD)

if __name__ == "__main__":
    main()