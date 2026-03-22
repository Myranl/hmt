"""Rename columns/values and migrate assignment dicts."""

from __future__ import annotations

from pathlib import Path

from core.categories.schema import CategoryStore


def rename_column_title(store: CategoryStore, column_id: str, new_title: str) -> bool:
    t = str(new_title).strip()
    if not t:
        return False
    for c in store.columns:
        if c.id == column_id:
            c.title = t
            return True
    return False


def rename_column_id(store: CategoryStore, old_id: str, new_id: str) -> tuple[bool, str]:
    """Change CSV / internal column id; migrates assignment keys. Returns (ok, message)."""
    new_id = str(new_id).strip()
    old_id = str(old_id).strip()
    if not new_id or not old_id:
        return False, "Empty id."
    if new_id == old_id:
        return True, ""
    if any(c.id == new_id for c in store.columns if c.id != old_id):
        return False, f"Column id «{new_id}» already exists."
    col = next((c for c in store.columns if c.id == old_id), None)
    if col is None:
        return False, "Column not found."
    col.id = new_id
    for _path, row in list(store.assignments.items()):
        if not isinstance(row, dict):
            continue
        if old_id in row:
            row[new_id] = row.pop(old_id)
    return True, ""


def rename_value_in_column(store: CategoryStore, column_id: str, old_value: str, new_value: str) -> tuple[bool, str]:
    """Rename an allowed value and update all assignments using it."""
    old_value = str(old_value).strip()
    new_value = str(new_value).strip()
    if not new_value:
        return False, "New value cannot be empty."
    if old_value == new_value:
        return True, ""
    col = next((c for c in store.columns if c.id == column_id), None)
    if col is None:
        return False, "Column not found."
    if old_value not in col.values:
        return False, "Old value is not in the allowed list."
    if new_value in col.values:
        return False, "New value already exists in this column."
    col.values = [new_value if v == old_value else v for v in col.values]
    for _path, row in store.assignments.items():
        if not isinstance(row, dict):
            continue
        if row.get(column_id) == old_value:
            row[column_id] = new_value
    return True, ""


def all_folder_prefixes_from_files(file_rels: set[str]) -> set[str]:
    """All parent folder paths (posix) for file relative paths."""
    out: set[str] = set()
    for f in file_rels:
        p = Path(f)
        if len(p.parts) <= 1:
            continue
        for i in range(len(p.parts) - 1):
            out.add(str(Path(*p.parts[: i + 1]).as_posix()))
    return out


def normalize_rel(raw: str) -> str:
    return str(raw).strip().replace("\\", "/").lstrip("/")


def set_paths_to_value(
    store: CategoryStore,
    rel_paths: list[str],
    column_id: str,
    value: str,
    *,
    rel_files_set: set[str],
) -> int:
    """Assign column value for each relative path (file or folder). Returns count updated."""
    col = next((c for c in store.columns if c.id == column_id), None)
    if col is None or value not in col.values:
        return 0
    folder_keys = all_folder_prefixes_from_files(rel_files_set)
    n = 0
    for raw in rel_paths:
        rel = normalize_rel(raw)
        if not rel:
            continue
        ok = False
        if store.granularity == "file":
            ok = rel in rel_files_set
        else:
            ok = rel in rel_files_set or rel in folder_keys
        if not ok:
            continue
        if rel not in store.assignments:
            store.assignments[rel] = {}
        store.assignments[rel][column_id] = value
        n += 1
    return n


def clear_paths_column(
    store: CategoryStore,
    rel_paths: list[str],
    column_id: str,
) -> int:
    n = 0
    for raw in rel_paths:
        rel = normalize_rel(raw)
        if not rel:
            continue
        if rel in store.assignments and column_id in store.assignments[rel]:
            del store.assignments[rel][column_id]
            if not store.assignments[rel]:
                del store.assignments[rel]
            n += 1
    return n


def bucket_resolved_paths(store: CategoryStore, column_id: str, rel_files: list[str]) -> dict[str, list[str]]:
    """Group relative *file* paths by resolved value for this column."""
    from core.categories.resolve import resolve_labels_for_image

    root = Path(store.input_root)
    buckets: dict[str, list[str]] = {}
    for rf in rel_files:
        abs_p = root / rf
        lab = resolve_labels_for_image(store, abs_p).get(column_id, "").strip()
        key = lab if lab else "(unassigned)"
        buckets.setdefault(key, []).append(rf)
    for k in list(buckets.keys()):
        buckets[k].sort(key=str.lower)
    return buckets
