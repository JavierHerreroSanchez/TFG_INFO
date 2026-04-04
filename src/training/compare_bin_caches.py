from pathlib import Path
import json

CANDIDATE_DIRS = [
    Path(r"C:\data\bin_for_pretraining\pretraining"),
    Path(r"C:\data\bin_for_pretraining"),
    Path(r"C:\Users\herre\PycharmProjects\TFG_INFO\data\bin_for_pretraining"),
]

BIN_NAMES = ["train.bin", "val.bin", "test.bin", "meta.json"]


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024


def inspect_cache_dir(cache_dir: Path) -> None:
    print("\n" + "=" * 80)
    print(f"CACHE_DIR: {cache_dir}")
    print("=" * 80)

    if not cache_dir.exists():
        print("[INFO] No existe")
        return

    for name in BIN_NAMES:
        p = cache_dir / name
        if p.exists():
            print(f"{name:<10} EXISTS   {human_size(p.stat().st_size)}")
        else:
            print(f"{name:<10} MISSING")

    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        print("\n[INFO] No hay meta.json -> no se puede identificar bien la procedencia.")
        return

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"\n[ERROR] No se pudo leer meta.json: {e}")
        return

    print("\n[meta.json]")
    for key in [
        "vocab_size",
        "block_size",
        "dtype",
        "train_tokens",
        "val_tokens",
        "test_tokens",
        "train_files",
        "val_files",
        "test_files",
        "token_field",
        "add_eos",
        "eos_id",
    ]:
        if key in meta:
            print(f"{key}: {meta[key]}")

    total_files = (
        meta.get("train_files", 0)
        + meta.get("val_files", 0)
        + meta.get("test_files", 0)
    )
    total_tokens = (
        meta.get("train_tokens", 0)
        + meta.get("val_tokens", 0)
        + meta.get("test_tokens", 0)
    )

    print(f"\n[RESUMEN]")
    print(f"total_files aprox: {total_files}")
    print(f"total_tokens aprox: {total_tokens:,}")

    if total_files < 5000:
        print("[PISTA] Esto parece un corpus pequeño. Podría ser MAESTRO solo.")
    else:
        print("[PISTA] Esto parece un corpus grande. Podría ser MAESTRO + ARIA.")


def main():
    for cache_dir in CANDIDATE_DIRS:
        inspect_cache_dir(cache_dir)


if __name__ == "__main__":
    main()