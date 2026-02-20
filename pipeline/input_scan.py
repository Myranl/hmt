from __future__ import annotations
from pathlib import Path
import csv
import os

def load_processed(csv_path: Path) -> set[str]:
    processed: set[str] = set()
    if not csv_path.exists():
        return processed
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                if not row:
                    continue
                p = row.get("image_path")
                st = (row.get("status") or "").strip().lower()
                if p and st == "ok":
                    processed.add(str(Path(p).expanduser().resolve()))
    except Exception:
        return processed
    return processed


def iter_images(root_dir: Path, *, subfolders: bool) -> list[Path]:
    # os.walk is usually faster than Path.rglob for big folders
    out: list[Path] = []
    if not subfolders:
        for p in root_dir.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                out.append(p)
    else:
        for dp, _dns, fns in os.walk(root_dir):
            for fn in fns:
                p = Path(dp) / fn
                if p.suffix.lower() in exts:
                    out.append(p)
    out.sort(key=lambda p: str(p).lower())
    return out