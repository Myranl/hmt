from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from config import CATEGORIES_STORE_NAME
from core.categories.schema import CATEGORY_STORE_VERSION, CategoryColumn, CategoryStore


def default_store_path(output_dir: str | Path) -> Path:
    return Path(output_dir).expanduser().resolve() / CATEGORIES_STORE_NAME


def _slug_title(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "column"


def make_unique_column_id(title: str, existing: set[str]) -> str:
    base = _slug_title(title)
    if base not in existing:
        existing.add(base)
        return base
    n = 2
    while f"{base}_{n}" in existing:
        n += 1
    cid = f"{base}_{n}"
    existing.add(cid)
    return cid


def load_store(path: Path) -> CategoryStore | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return CategoryStore.from_json(data)


def save_store(path: Path, store: CategoryStore) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    store.version = CATEGORY_STORE_VERSION
    path.write_text(json.dumps(store.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_store_on_disk(path: Path, *, input_root: Path) -> CategoryStore:
    st = load_store(path)
    if st is None:
        st = CategoryStore(
            input_root=str(input_root.expanduser().resolve()),
            granularity="file",
            columns=[],
            assignments={},
        )
    return st
