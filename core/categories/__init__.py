"""User-defined categorical columns (per-file or per-folder) merged into results.csv."""

from core.categories.schema import CategoryColumn, CategoryStore, Granularity
from core.categories.store_io import load_store, save_store, default_store_path
from core.categories.resolve import merge_row_with_categories, resolve_labels_for_image

__all__ = [
    "CategoryColumn",
    "CategoryStore",
    "Granularity",
    "load_store",
    "save_store",
    "default_store_path",
    "resolve_labels_for_image",
    "merge_row_with_categories",
]
