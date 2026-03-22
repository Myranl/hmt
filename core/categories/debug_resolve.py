"""Human-readable explanation of why category labels apply or not (for troubleshooting)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.categories.paths import posix_rel
from core.categories.resolve import (
    folder_chain_from_file_relpath,
    merge_row_with_categories,
    normalize_posix_relpath,
    resolve_labels_by_path_segment_match,
    resolve_labels_for_image,
    resolve_labels_for_result_row,
    resolve_labels_from_rel_path,
    resolve_rel_file_under_root,
    path_contains_folder_segment,
)
from core.categories.schema import CategoryStore


def format_category_resolution_report(
    store: CategoryStore,
    row: dict[str, Any],
    image_path: Path | None = None,
) -> str:
    """Build a multi-line report for debugging (paste into a ticket or read in the UI)."""
    lines: list[str] = []
    lines.append("=== Category resolution debug ===")
    lines.append(f"granularity: {store.granularity}")
    lines.append(f"input_root: {store.input_root}")
    lines.append(f"columns (ids): {store.column_ids()}")
    lines.append("")
    lines.append("--- CSV / row fields ---")
    for k in ("image_path", "overlay_path", "img_name"):
        lines.append(f"  {k}: {row.get(k, '')!r}")
    lines.append("")

    root = Path(store.input_root).expanduser().resolve()
    ip_s = str(row.get("image_path", "")).strip()

    lines.append("--- Under input_root? ---")
    if ip_s:
        try:
            ip = Path(ip_s).expanduser().resolve()
            pr = posix_rel(ip, root)
            lines.append(f"posix_rel(image_path, input_root) = {pr!r}")
            if pr is None:
                lines.append(
                    "  → Path is NOT under input_root (different drive or folder prefix).\n"
                    "  → Matching uses full path strings (segment rules) and/or unique basename."
                )
            else:
                lines.append(f"  → Relative file path under root: {pr!r}")
                if store.granularity == "leaf_folder":
                    chain = folder_chain_from_file_relpath(pr)
                    lines.append(f"  → leaf_folder chain (deepest-first): {chain}")
        except OSError as e:
            lines.append(f"  (path error: {e})")
    else:
        lines.append("(no image_path in row)")
    lines.append("")

    rel = resolve_rel_file_under_root(store, row)
    lines.append(f"resolve_rel_file_under_root → {rel!r}")
    if rel and store.granularity == "leaf_folder":
        from_rel = resolve_labels_from_rel_path(store, rel)
        lines.append(f"  resolve_labels_from_rel_path (string chain) → {from_rel!r}")
    lines.append("")

    seg = resolve_labels_by_path_segment_match(store, row, [str(image_path)] if image_path else None)
    lines.append(f"resolve_labels_by_path_segment_match → {seg!r}")
    lines.append("")

    if image_path is not None:
        img_labs = resolve_labels_for_image(store, image_path)
        lines.append(f"resolve_labels_for_image(Path) → {img_labs!r}")
    else:
        lines.append("resolve_labels_for_image skipped (no Path passed)")
    lines.append("")

    final = resolve_labels_for_result_row(store, row, image_path)
    merged_row = merge_row_with_categories(dict(row), image_path, store)
    cat_only = {k: merged_row[k] for k in store.column_ids() if k in merged_row}
    lines.append(f"resolve_labels_for_result_row (merged logic) → {final!r}")
    lines.append(f"merge_row_with_categories (category columns only) → {cat_only!r}")
    lines.append("")

    lines.append("--- Segment match matrix (each JSON key vs each path string) ---")
    lines.append("Rule: the key must appear as a full path segment, e.g. .../TB_P28/Females/K13/... contains TB_P28/Females but NOT TB_P28/Females/M13 if the folder is K13.")
    cand: list[str] = []
    for fld in ("image_path", "overlay_path"):
        s = str(row.get(fld, "")).strip()
        if s:
            cand.append(s)
    if image_path is not None:
        cand.append(str(image_path))
    if not cand:
        lines.append("(no path strings to test)")
    else:
        keys_sorted = sorted(store.assignments.keys(), key=lambda k: len(normalize_posix_relpath(k)), reverse=True)
        for path in cand:
            lines.append(f"path: {path[:120]}{'…' if len(path) > 120 else ''}")
            for k in keys_sorted[:40]:  # cap
                ok = path_contains_folder_segment(path, k)
                vals = store.assignments.get(k) or {}
                mark = "YES" if ok else "no"
                lines.append(f"  [{mark}] key {k!r} → {vals!r}")
            if len(keys_sorted) > 40:
                lines.append(f"  … ({len(keys_sorted) - 40} more keys omitted)")
            lines.append("")

    lines.append("--- Hints ---")
    if store.granularity == "leaf_folder" and ip_s:
        for k in sorted(store.assignments.keys(), key=len, reverse=True):
            if path_contains_folder_segment(ip_s, k):
                continue
            nk = normalize_posix_relpath(k)
            if not nk or "/" not in nk:
                continue
            parent = str(Path(nk).parent.as_posix())
            if parent in ("", ".", ""):
                continue
            if path_contains_folder_segment(ip_s, parent):
                lines.append(
                    f"• Key {k!r} does NOT match (wrong subfolder?). "
                    f"Parent {parent!r} IS in the path — assign the column on that parent folder "
                    f"to apply to all subfolders (K13, M13, …), or use the exact folder key that appears in the path."
                )
                break
    if not final and store.assignments:
        lines.append("• No labels resolved: check that a folder key in JSON is a segment of the path (same folder names), or assign on a parent folder.")

    return "\n".join(lines)


def write_category_debug_report(
    store: CategoryStore,
    row: dict[str, Any],
    output_dir: str | Path,
    *,
    image_path: Path | None = None,
    filename: str = "category_resolve_debug.txt",
) -> Path:
    """Write ``format_category_resolution_report`` next to the output CSV (overwrite)."""
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    p = out / filename
    p.write_text(format_category_resolution_report(store, row, image_path=image_path), encoding="utf-8")
    return p
