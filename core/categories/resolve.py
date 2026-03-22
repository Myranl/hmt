from __future__ import annotations

from pathlib import Path
from typing import Any

from core.categories.paths import ancestor_folder_chain_relpaths, posix_rel
from core.categories.schema import CategoryStore


def resolve_labels_for_image(store: CategoryStore, image_path: Path) -> dict[str, str]:
    """Resolve category values for one image path (absolute or relative)."""
    root = Path(store.input_root)
    ip = image_path.expanduser().resolve()
    rel_file = posix_rel(ip, root)
    if rel_file is None:
        return {}

    if store.granularity == "file":
        return dict(store.assignments.get(rel_file, {}))

    # leaf_folder: per column, deepest folder in ancestor chain that defines it wins
    chain = ancestor_folder_chain_relpaths(ip, root)
    resolved: dict[str, str] = {}
    for col in store.column_ids():
        for folder_rel in chain:
            row = store.assignments.get(folder_rel)
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


def merge_row_with_categories(row: dict[str, Any], image_path: Path, store: CategoryStore | None) -> dict[str, Any]:
    if store is None:
        return row
    out = dict(row)
    labels = resolve_labels_for_image(store, Path(image_path))
    for k, v in labels.items():
        out[k] = v
    return out
