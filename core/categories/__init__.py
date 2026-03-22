"""User-defined categorical columns (per-file or per-folder) merged into results.csv."""

from core.categories.schema import CategoryColumn, CategoryStore, Granularity
from core.categories.store_io import load_store, save_store, default_store_path
from core.categories.resolve import (
    merge_row_with_categories,
    normalize_posix_relpath,
    path_contains_folder_segment,
    resolve_labels_by_path_segment_match,
    resolve_labels_for_image,
    resolve_labels_for_result_row,
    resolve_rel_file_under_root,
    resolve_labels_from_rel_path,
)
from core.categories.debug_resolve import format_category_resolution_report

__all__ = [
    "CategoryColumn",
    "CategoryStore",
    "Granularity",
    "load_store",
    "save_store",
    "default_store_path",
    "resolve_labels_for_image",
    "merge_row_with_categories",
    "normalize_posix_relpath",
    "path_contains_folder_segment",
    "resolve_labels_by_path_segment_match",
    "resolve_labels_for_result_row",
    "resolve_rel_file_under_root",
    "resolve_labels_from_rel_path",
    "format_category_resolution_report",
]
