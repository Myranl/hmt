from __future__ import annotations
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable
import json

def _iter_image_paths(root: Path, *, recursive: bool = True) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    if recursive:
        it: Iterable[Path] = root.rglob("*")
    else:
        it = root.glob("*")

    paths = [p for p in it if p.is_file() and p.suffix.lower() in exts]
    paths.sort(key=lambda p: str(p).lower())
    return paths


def _to_rows(result: Any, *, image_path: Path) -> list[dict[str, Any]]:
    # Accept dict, dataclass, or any object with __dict__.
    if result is None:
        row0: dict[str, Any] = {"image_path": str(image_path), "status": "skip"}
    elif isinstance(result, dict):
        row0 = dict(result)
    elif is_dataclass(result):
        row0 = asdict(result)
    else:
        row0 = dict(getattr(result, "__dict__", {}))

    row0.setdefault("image_path", str(image_path))
    row0.setdefault("img_name", image_path.name)
    st = str(row0.get("status", "ok")).strip().lower()
    accepted = "skip" if st in ("skip", "skipped", "cancelled", "canceled") else ("error" if st == "error" else "ok")

    # Canonical compact schema requested by user.
    non_complete = str(row0.get("non_complete_contour", "")).strip().lower() in ("1", "true", "yes")
    corrected = str(row0.get("contour_corrected", "")).strip().lower() in ("1", "true", "yes")
    contour_version = "corrected" if non_complete else "single"
    if non_complete and not corrected:
        contour_version = "raw"

    row: dict[str, Any] = {
        "overlay_path": row0.get("overlay_path", ""),
        "img_name": row0.get("img_name", image_path.name),
        "accepted": accepted,
        "contour_version": contour_version,
        "brain_area_px": row0.get("brain_area_px", ""),
        "brain_perim_px": row0.get("brain_perim_px", ""),
        "midline_area_left_px": row0.get("midline_area_left_px", ""),
        "midline_area_right_px": row0.get("midline_area_right_px", ""),
        "midline_perimeter_left_px": row0.get("midline_perimeter_left_px", ""),
        "midline_perimeter_right_px": row0.get("midline_perimeter_right_px", ""),
        "non_complete_contour": row0.get("non_complete_contour", ""),
        "hipp_area_left_px": row0.get("hipp_area_left_px", row0.get("hip_left_area_px", "")),
        "hipp_area_right_px": row0.get("hipp_area_right_px", row0.get("hip_right_area_px", "")),
        "hipp_perimeter_left_px": row0.get("hipp_perimeter_left_px", row0.get("hip_left_perim_px", "")),
        "hipp_perimeter_right_px": row0.get("hipp_perimeter_right_px", row0.get("hip_right_perim_px", "")),
    }
    if non_complete and corrected:
        row_raw = dict(row)
        row_raw["contour_version"] = "raw"
        return [row_raw, row]
    return [row]

def process_paths(
    img_paths: list[Path],
    out_dir: str | Path,
) -> "list[dict[str, Any]]":
    """Process a given list of image paths.

    Writes results incrementally to out_dir/results.csv to avoid losing progress.
    """
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    try:
        from pipeline.single_image import process_one_image  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Missing pipeline.single_image.process_one_image. "
            "Create pipeline/single_image.py with a function process_one_image(path, out_dir)."
        ) from e

    rows: list[dict[str, Any]] = []

    csv_path = out / "results.csv"
    csv_fieldnames: list[str] = [
        "overlay_path",
        "img_name",
        "accepted",
        "contour_version",
        "brain_area_px",
        "brain_perim_px",
        "midline_area_left_px",
        "midline_area_right_px",
        "midline_perimeter_left_px",
        "midline_perimeter_right_px",
        "non_complete_contour",
        "hipp_area_left_px",
        "hipp_area_right_px",
        "hipp_perimeter_left_px",
        "hipp_perimeter_right_px",
    ]

    import csv

    def _coerce_scalar(v: Any) -> Any:
        if v is None:
            return ""
        if isinstance(v, (dict, list, tuple)):
            try:
                return json.dumps(v, ensure_ascii=False)
            except Exception:
                return str(v)
        return v

    def _normalize_to_schema(row: dict[str, Any]) -> dict[str, Any]:
        return {k: _coerce_scalar(row.get(k, "")) for k in csv_fieldnames}

    def _read_existing_rows() -> list[dict[str, Any]]:
        if not csv_path.exists():
            return []
        rows_existing: list[dict[str, Any]] = []
        try:
            with csv_path.open("r", newline="", encoding="utf-8") as rf:
                rdr = csv.reader(rf)
                hdr = next(rdr, None)
                if not hdr:
                    return []
                for raw_fields in rdr:
                    if not raw_fields:
                        continue
                    fields = list(raw_fields)
                    # Recover legacy malformed row written as one quoted CSV line.
                    if len(fields) == 1 and "," in fields[0]:
                        try:
                            fields = next(csv.reader([fields[0]]))
                        except Exception:
                            pass
                    rec = {k: (fields[i] if i < len(fields) else "") for i, k in enumerate(hdr)}
                    rows_existing.append(_normalize_to_schema(rec))
        except Exception:
            return []
        return rows_existing

    def _append_row(row: dict[str, Any]) -> None:
        """Upsert one row by overlay_path/img_name/contour_version and rewrite CSV."""
        norm = _normalize_to_schema(dict(row))
        base_key = str(norm.get("overlay_path", "")).strip() or str(norm.get("img_name", "")).strip()
        key = f"{base_key}|{str(norm.get('contour_version', '')).strip()}"

        rows_existing = _read_existing_rows()
        replaced = False
        if key:
            for i, rec in enumerate(rows_existing):
                rec_base = str(rec.get("overlay_path", "")).strip() or str(rec.get("img_name", "")).strip()
                rec_key = f"{rec_base}|{str(rec.get('contour_version', '')).strip()}"
                if rec_key == key:
                    rows_existing[i] = norm
                    replaced = True
                    break
        if not replaced:
            rows_existing.append(norm)

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=csv_fieldnames)
            w.writeheader()
            for rec in rows_existing:
                w.writerow({k: rec.get(k, "") for k in csv_fieldnames})
            f.flush()

    processed_images = 0
    try:
        for p in img_paths:
            try:
                res = process_one_image(p, out_dir=out)
                rows_for_image = _to_rows(res, image_path=p)
            except Exception:
                rows_for_image = [{
                    "img_name": p.name,
                    "accepted": "error",
                    "contour_version": "single",
                    "overlay_path": "",
                    "brain_area_px": "",
                    "brain_perim_px": "",
                    "midline_area_left_px": "",
                    "midline_area_right_px": "",
                    "midline_perimeter_left_px": "",
                    "midline_perimeter_right_px": "",
                    "non_complete_contour": "",
                    "hipp_area_left_px": "",
                    "hipp_area_right_px": "",
                    "hipp_perimeter_left_px": "",
                    "hipp_perimeter_right_px": "",
                }]

            processed_images += 1
            for row in rows_for_image:
                rows.append(row)
                try:
                    _append_row(row)
                except Exception:
                    pass
    finally:
        # If we exited mid-loop, append "skipped" for images we didn't process.
        for p in img_paths[processed_images:]:
            try:
                for row in _to_rows(None, image_path=p):
                    _append_row(row)
            except Exception:
                pass

    return rows


def process_folder(
    input_dir: str | Path,
    out_dir: str | Path,
    *,
    recursive: bool = True,
    glob_limit: int | None = None,
) -> "list[dict[str, Any]]":
    """Process all images in a folder.

    Parameters
    - input_dir: folder with images (optionally with subfolders)
    - out_dir: output folder for overlays/masks/metrics
    - recursive: search subfolders
    - glob_limit: optionally cap number of images (debug)

    Returns
    - list of per-image result rows (dict)
    """
    in_dir = Path(input_dir).expanduser().resolve()
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    img_paths = _iter_image_paths(in_dir, recursive=recursive)
    if glob_limit is not None:
        img_paths = img_paths[: int(glob_limit)]

    rows = process_paths(img_paths, out)
    return rows
