"""Scan image paths under an input root (same extensions as main launcher)."""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def iter_images_under(root: Path, *, recursive: bool) -> list[Path]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        return []
    if recursive:
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    else:
        files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort(key=lambda p: str(p).lower())
    return files


def folder_contains_image_directly(folder: Path) -> bool:
    try:
        for p in folder.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                return True
    except OSError:
        pass
    return False
