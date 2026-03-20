from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import os


@dataclass(frozen=True)
class ProcessedIndex:
    """Maps CSV rows to input files: exact paths (new schema) + basename fallback (legacy compact CSV)."""

    paths: frozenset[str] = frozenset()
    legacy_basenames: frozenset[str] = frozenset()

    def matches(self, p: Path | str) -> bool:
        path = Path(p)
        key = str(path.expanduser().resolve())
        if key in self.paths:
            return True
        return path.name.lower() in self.legacy_basenames


def _row_accepted_ok(row: dict[str, str | None]) -> bool:
    """True if this row counts as a successful run (for 'processed' badges).

    New schema: ``accepted == ok``. Legacy: ``status == ok`` when ``accepted`` absent.
    """
    acc = (row.get("accepted") or "").strip().lower()
    if acc:
        return acc == "ok"
    return (row.get("status") or "").strip().lower() == "ok"


def _row_source_image_path(row: dict[str, str | None]) -> str:
    """Absolute or relative path string to the source image, if present in the row."""
    for key in ("image_path", "source_path", "input_path"):
        p = (row.get(key) or "").strip()
        if p:
            return p
    return ""


def _legacy_basename_from_row(row: dict[str, str | None]) -> str:
    """Filename for matching when CSV has no full path (compact ``results.csv``)."""
    raw = (row.get("img_name") or "").strip()
    if not raw:
        return ""
    return Path(raw).name.lower()


def load_processed(csv_path: Path) -> ProcessedIndex:
    """Build index from ``results.csv``.

    - Rows with ``image_path`` (or legacy aliases): match by resolved path.
    - Successful rows without path but with ``img_name``: match any input file with the same basename
      (for older CSV written before ``image_path`` existed).
    """
    paths: set[str] = set()
    legacy: set[str] = set()
    if not csv_path.exists():
        return ProcessedIndex()
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                if not row:
                    continue
                if not _row_accepted_ok(row):
                    continue
                p = _row_source_image_path(row)
                if p:
                    paths.add(str(Path(p).expanduser().resolve()))
                    continue
                base = _legacy_basename_from_row(row)
                if base:
                    legacy.add(base)
    except Exception:
        return ProcessedIndex()
    return ProcessedIndex(frozenset(paths), frozenset(legacy))


exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


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
