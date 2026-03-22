from __future__ import annotations

import re
import sys
from pathlib import Path, PureWindowsPath
from typing import Any

from core.categories.paths import ancestor_folder_chain_relpaths, posix_rel
from core.categories.schema import CategoryStore


def _is_absolute_path_str(s: str) -> bool:
    """True for POSIX abs paths and Windows ``D:\\...`` / ``D:/...`` (even on non-Windows)."""
    s = str(s).strip()
    if not s:
        return False
    try:
        if Path(s).is_absolute():
            return True
    except OSError:
        pass
    try:
        return bool(PureWindowsPath(s).drive)
    except Exception:
        return False


def normalize_posix_relpath(s: str) -> str:
    """Normalize a relative path string to POSIX (no leading slash, no ./)."""
    t = str(s).replace("\\", "/").strip()
    while t.startswith("./"):
        t = t[2:]
    return t.lstrip("/")


def _folder_key_variants(k: str) -> tuple[str, ...]:
    """Match JSON keys that may use '' or '.' for the input root folder."""
    n = normalize_posix_relpath(k)
    if n in ("", "."):
        return ("", ".")
    return (n,)


def _get_assignment_row(assignments: dict[str, dict[str, str]], folder_rel: str) -> dict[str, str] | None:
    for key in _folder_key_variants(folder_rel):
        row = assignments.get(key)
        if row:
            return row
    return None


def folder_chain_from_file_relpath(rel_file: str) -> list[str]:
    """Folder chain deepest-first, same order as ancestor_folder_chain for a file under root.

    Example: ``a/b/c.png`` → ``['a/b', 'a', '.']``
    """
    rel_file = rel_file.replace("\\", "/").strip("/")
    if not rel_file:
        return ["."]
    chain: list[str] = []
    cur = Path(rel_file).parent
    while True:
        chain.append(cur.as_posix())
        if cur.parent == cur:
            break
        cur = cur.parent
    return chain


def _leaf_folder_chain(store: CategoryStore, rel_file: str) -> list[str]:
    """Deepest-first folder chain for ``rel_file``, including ``rel_file`` if it is an assignment key."""
    rel_file = normalize_posix_relpath(rel_file)
    chain = folder_chain_from_file_relpath(rel_file)
    # CSV / JSON may use a folder path as the key (e.g. ``batch1``); ensure that folder is visited.
    if rel_file in store.assignments and rel_file not in chain:
        return [rel_file] + chain
    return chain


def resolve_labels_from_rel_path(store: CategoryStore, rel_file: str) -> dict[str, str]:
    """Resolve labels using a POSIX path relative to ``input_root`` (keys in ``assignments``).

    Works without the file existing on disk (string-based chain for ``leaf_folder``).
    """
    rel_file = normalize_posix_relpath(rel_file)
    if not rel_file:
        return {}

    if store.granularity == "file":
        return dict(store.assignments.get(rel_file, {}))

    chain = _leaf_folder_chain(store, rel_file)
    resolved: dict[str, str] = {}
    for col in store.column_ids():
        for folder_rel in chain:
            row = _get_assignment_row(store.assignments, folder_rel)
            if row and col in row and str(row[col]).strip() != "":
                resolved[col] = str(row[col]).strip()
                break
    file_row = store.assignments.get(rel_file)
    if file_row:
        for col_id, val in file_row.items():
            if col_id in store.column_ids() and str(val).strip() != "":
                resolved[col_id] = str(val).strip()
    return resolved


def _rel_safe_under_root(rel: str) -> bool:
    n = normalize_posix_relpath(rel)
    if ".." in n.split("/"):
        return False
    # Reject mistaken "relative" paths that still contain a Windows drive letter.
    if re.match(r"^[A-Za-z]:", n):
        return False
    return True


def path_contains_folder_segment(full_path: str, folder_key: str) -> bool:
    """True if ``folder_key`` appears as a path segment in ``full_path`` (not a substring of a name)."""
    fk = normalize_posix_relpath(folder_key)
    if not fk:
        return False
    fp = str(full_path).replace("\\", "/")
    flags = re.IGNORECASE if sys.platform == "win32" else 0
    return (
        re.search(r"(^|/)" + re.escape(fk) + r"(/|$)", fp, flags=flags) is not None
    )


def _collect_path_strings_for_segment(
    row: dict[str, Any], extra_paths: list[str] | None
) -> list[str]:
    out: list[str] = []
    for fld in ("image_path", "overlay_path"):
        s = str(row.get(fld, "")).strip()
        if s:
            out.append(s)
    if extra_paths:
        for p in extra_paths:
            ps = str(p).strip()
            if ps:
                out.append(ps)
    return out


def resolve_labels_by_path_segment_match(
    store: CategoryStore,
    row: dict[str, Any],
    extra_paths: list[str] | None = None,
) -> dict[str, str]:
    """When ``input_root`` / ``posix_rel`` disagree with CSV drive layout, match assignment *folder* keys
    as path segments inside ``image_path`` / ``overlay_path`` (e.g. ``TB_P28/Females`` inside a full path).

    For ``leaf_folder``, each column uses the **longest** matching key (deepest folder wins).
    """
    if store.granularity != "leaf_folder":
        return {}

    candidates = _collect_path_strings_for_segment(row, extra_paths)
    if not candidates:
        return {}

    cols = store.column_ids()
    resolved: dict[str, str] = {}
    # Longer keys first = prefer deeper folders
    keys_sorted = sorted(store.assignments.keys(), key=lambda k: len(normalize_posix_relpath(k)), reverse=True)

    for col in cols:
        best_len = -1
        best_val: str | None = None
        for k in keys_sorted:
            row_a = store.assignments.get(k) or {}
            if col not in row_a or not str(row_a[col]).strip():
                continue
            for path in candidates:
                if path_contains_folder_segment(path, k):
                    ln = len(normalize_posix_relpath(k))
                    if ln > best_len:
                        best_len = ln
                        best_val = str(row_a[col]).strip()
                    break
        if best_val is not None:
            resolved[col] = best_val
    return resolved


def _resolve_file_mode_by_path_suffix(store: CategoryStore, path_str: str) -> dict[str, str]:
    """If the image is not under ``input_root``, match a file-mode assignment key as a path suffix."""
    fp = str(path_str).replace("\\", "/")
    flags = re.IGNORECASE if sys.platform == "win32" else 0
    best: dict[str, str] = {}
    best_len = -1
    for k, row_a in store.assignments.items():
        nk = normalize_posix_relpath(k)
        if not nk:
            continue
        if re.search(r"(^|/)" + re.escape(nk) + r"$", fp, flags=flags):
            if len(nk) > best_len:
                best_len = len(nk)
                best = dict(row_a)
    return best


def resolve_rel_file_under_root(store: CategoryStore, row: dict[str, Any]) -> str | None:
    """Pick a POSIX path *under input_root* that matches CSV fields and/or keys in ``assignments``.

    Priority: exact string match to a JSON key → absolute path on disk → unique basename in keys
    → normalized relative path from CSV (no ``..``).
    """
    root = Path(store.input_root).expanduser().resolve()
    keys = set(store.assignments.keys())
    img_name = str(row.get("img_name", "")).strip()

    def try_key(s: str) -> str | None:
        n = normalize_posix_relpath(s)
        if n and n in keys:
            return n
        return None

    # 1) Relative image_path / overlay_path — exact key in saved JSON
    for fld in ("image_path", "overlay_path"):
        s = str(row.get(fld, "")).strip()
        if not s or _is_absolute_path_str(s):
            continue
        hit = try_key(s)
        if hit:
            return hit

    # 2) Absolute paths → relative under root
    for fld in ("image_path", "overlay_path"):
        s = str(row.get(fld, "")).strip()
        if not s:
            continue
        if not _is_absolute_path_str(s):
            continue
        try:
            rp = Path(s).expanduser().resolve()
            rel = posix_rel(rp, root)
            if rel:
                return rel
        except OSError:
            pass

    # 3) Relative path + file exists on disk under root
    for fld in ("image_path", "overlay_path"):
        s = str(row.get(fld, "")).strip()
        if not s or _is_absolute_path_str(s):
            continue
        n = normalize_posix_relpath(s)
        if not n:
            continue
        try:
            cand = (root / n).resolve()
            if cand.is_file():
                rel = posix_rel(cand, root)
                if rel:
                    return rel
        except OSError:
            pass

    # 4) img_name at root of input
    if img_name:
        try:
            cand = (root / img_name).resolve()
            if cand.is_file():
                rel = posix_rel(cand, root)
                if rel:
                    return rel
        except OSError:
            pass

    # 5) Unique basename among assignment keys
    if img_name:
        matches = [k for k in keys if Path(k).name == img_name]
        if len(matches) == 1:
            return matches[0]

    # 6) Fallback: normalized relative path from CSV (under root, no traversal)
    for fld in ("image_path", "overlay_path"):
        s = str(row.get(fld, "")).strip()
        if not s or _is_absolute_path_str(s):
            continue
        n = normalize_posix_relpath(s)
        if n and _rel_safe_under_root(n):
            return n

    return None


def resolve_labels_for_result_row(
    store: CategoryStore,
    row: dict[str, Any],
    image_path: Path | None,
) -> dict[str, str]:
    """Resolve category labels from CSV row + optional filesystem path to the image.

    Merges (lowest → highest priority): segment match in path strings → rel path from CSV/disk
    → ``resolve_labels_for_image`` (filesystem under ``input_root``).
    """
    extra = [str(image_path)] if image_path is not None else None
    seg = resolve_labels_by_path_segment_match(store, row, extra)
    rel = resolve_rel_file_under_root(store, row)
    from_rel = resolve_labels_from_rel_path(store, rel) if rel else {}
    from_img = resolve_labels_for_image(store, image_path) if image_path is not None else {}

    merged: dict[str, str] = {}
    merged.update(seg)
    merged.update(from_rel)
    merged.update(from_img)
    return merged


def resolve_labels_for_image(store: CategoryStore, image_path: Path) -> dict[str, str]:
    """Resolve category values for one image path (absolute or relative)."""
    root = Path(store.input_root)
    ip = image_path.expanduser().resolve()
    rel_file = posix_rel(ip, root)
    if rel_file is None:
        # e.g. CSV on another drive than ``input_root`` — match keys inside the path string
        if store.granularity == "file":
            return _resolve_file_mode_by_path_suffix(store, str(ip))
        return resolve_labels_by_path_segment_match(store, {}, [str(ip)])

    if store.granularity == "file":
        return dict(store.assignments.get(rel_file, {}))

    # leaf_folder: per column, deepest folder in ancestor chain that defines it wins
    chain = ancestor_folder_chain_relpaths(ip, root)
    resolved: dict[str, str] = {}
    for col in store.column_ids():
        for folder_rel in chain:
            row = _get_assignment_row(store.assignments, folder_rel)
            if row and col in row and str(row[col]).strip() != "":
                resolved[col] = str(row[col]).strip()
                break
    # Explicit per-file overrides (same keys as in file mode) — e.g. move one image in Overview
    file_row = store.assignments.get(rel_file)
    if file_row:
        for col_id, val in file_row.items():
            if col_id in store.column_ids() and str(val).strip() != "":
                resolved[col_id] = str(val).strip()
    return resolved


def merge_row_with_categories(
    row: dict[str, Any],
    image_path: Path | None,
    store: CategoryStore | None,
) -> dict[str, Any]:
    if store is None:
        return row
    out = dict(row)
    labels = resolve_labels_for_result_row(store, row, image_path)
    for k, v in labels.items():
        out[k] = v
    return out
