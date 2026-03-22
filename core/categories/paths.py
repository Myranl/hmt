from __future__ import annotations

from pathlib import Path


def posix_rel(path: Path, root: Path) -> str | None:
    """Return ``path`` relative to ``root`` as POSIX, or None if not under root."""
    try:
        a = path.expanduser().resolve()
        b = root.expanduser().resolve()
        rel = a.relative_to(b)
        return rel.as_posix()
    except (ValueError, OSError):
        return None


def ancestor_folder_chain_relpaths(file_path: Path, root: Path) -> list[str]:
    """Deepest folder first (file parent, then its parent, …) until ``root``."""
    out: list[str] = []
    try:
        r = root.expanduser().resolve()
        d = file_path.expanduser().resolve().parent
        while True:
            rel = posix_rel(d, r)
            if rel is None:
                break
            out.append(rel)
            if d == r:
                break
            parent = d.parent
            if parent == d:
                break
            d = parent
    except OSError:
        pass
    return out
