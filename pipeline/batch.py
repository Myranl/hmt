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


def _to_row(result: Any, *, image_path: Path) -> dict[str, Any]:
    # Accept dict, dataclass, or any object with __dict__.
    if result is None:
        return {"image_path": str(image_path), "status": "skipped"}

    if isinstance(result, dict):
        row = dict(result)
    elif is_dataclass(result):
        row = asdict(result)
    else:
        row = dict(getattr(result, "__dict__", {}))

    row.setdefault("image_path", str(image_path))
    row.setdefault("status", "ok")

    # --- Flatten selected nested structures for CSV readability ---
    def _jsonify(v: Any) -> str:
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return json.dumps(str(v), ensure_ascii=False)

    # Brain outline params
    bop = row.get("brain_outline_params")
    if isinstance(bop, dict):
        row.setdefault("brain_thr", bop.get("thr"))
        row.setdefault("brain_close", bop.get("close"))
        row.setdefault("brain_open", bop.get("open"))
        row.setdefault("brain_smooth", bop.get("smooth"))
        row.setdefault("brain_scale", bop.get("scale"))
        row.setdefault("brain_area_px", bop.get("area_px"))
        row.setdefault("brain_perim_px", bop.get("perim_px"))
        row.setdefault("brain_accepted", bop.get("accepted"))
        row["brain_outline_params_json"] = _jsonify(bop)

    # Midline params (hemispheres)
    mlp = row.get("midline_params")
    if isinstance(mlp, dict):
        row.setdefault("midline_xy", mlp.get("midline_xy"))
        row.setdefault("hemi_area_left_px", mlp.get("area_left_px"))
        row.setdefault("hemi_area_right_px", mlp.get("area_right_px"))
        row.setdefault("hemi_perim_left_px", mlp.get("perimeter_left_px"))
        row.setdefault("hemi_perim_right_px", mlp.get("perimeter_right_px"))
        row.setdefault("midline_crop_x0", mlp.get("crop_x0"))
        row.setdefault("midline_crop_y0", mlp.get("crop_y0"))
        row.setdefault("midline_crop_x1", mlp.get("crop_x1"))
        row.setdefault("midline_crop_y1", mlp.get("crop_y1"))
        row.setdefault("midline_pad", mlp.get("pad"))
        row["midline_params_json"] = _jsonify(mlp)

    # Hippocampus summary metrics
    row.setdefault("hip_left_area_px", row.get("hip_left_area_px"))
    row.setdefault("hip_left_perim_px", row.get("hip_left_perim_px"))
    row.setdefault("hip_right_area_px", row.get("hip_right_area_px"))
    row.setdefault("hip_right_perim_px", row.get("hip_right_perim_px"))

    # ROI + UI parameters
    roi = row.get("roi")
    if roi is not None:
        row["roi_json"] = _jsonify(roi)

    ui_params = row.get("params")
    if isinstance(ui_params, dict):
        # promote the most useful ones
        row.setdefault("ui_t1", ui_params.get("t1"))
        row.setdefault("ui_t2", ui_params.get("t2"))
        row.setdefault("ui_small_to_gray", ui_params.get("small_to_gray"))
        row.setdefault("ui_small_N", ui_params.get("small_N"))
        row.setdefault("ui_grid_on", ui_params.get("grid_on"))
        row.setdefault("ui_grid_step", ui_params.get("grid_step"))
        row.setdefault("ui_roi_x0", ui_params.get("x0"))
        row.setdefault("ui_roi_y0", ui_params.get("y0"))
        row.setdefault("ui_roi_x1", ui_params.get("x1"))
        row.setdefault("ui_roi_y1", ui_params.get("y1"))
        row["ui_params_json"] = _jsonify(ui_params)

    # Keep the original nested dicts (optional), but make sure CSV doesn't explode
    # by also providing stable JSON columns above.

    return row

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
    csv_fieldnames: list[str] = []
    extra_col = "extra_json"

    baseline = ["image_path", "status", "error", "error_type", "traceback"]

    # Stable, readable schema (promoted fields from _to_row).
    promoted = [
        # brain outline
        "brain_thr",
        "brain_close",
        "brain_open",
        "brain_smooth",
        "brain_scale",
        "brain_area_px",
        "brain_perim_px",
        "brain_accepted",
        "brain_outline_params_json",

        # hemispheres / midline
        "midline_xy",
        "hemi_area_left_px",
        "hemi_area_right_px",
        "hemi_perim_left_px",
        "hemi_perim_right_px",
        "midline_crop_x0",
        "midline_crop_y0",
        "midline_crop_x1",
        "midline_crop_y1",
        "midline_pad",
        "midline_params_json",

        # hippocampus metrics
        "hip_left_area_px",
        "hip_left_perim_px",
        "hip_right_area_px",
        "hip_right_perim_px",

        # ROI + UI params
        "roi_json",
        "ui_t1",
        "ui_t2",
        "ui_small_to_gray",
        "ui_small_N",
        "ui_grid_on",
        "ui_grid_step",
        "ui_roi_x0",
        "ui_roi_y0",
        "ui_roi_x1",
        "ui_roi_y1",
        "ui_params_json",
    ]

    # Keep extra_json last as a safety net for future/unknown keys.
    csv_fieldnames = baseline + promoted + [extra_col]

    def _ensure_header_and_writer():
        """Open CSV in append mode and ensure header is present.

        If results.csv already exists, reuse its header (append-only, no rewrites).
        """
        import csv

        nonlocal csv_fieldnames

        write_header = not csv_path.exists()

        if not write_header and not csv_fieldnames:
            try:
                with csv_path.open("r", newline="", encoding="utf-8") as rf:
                    r = csv.reader(rf)
                    hdr = next(r, None)
                if hdr:
                    csv_fieldnames = list(hdr)
            except Exception:
                pass

        f = csv_path.open("a", newline="", encoding="utf-8")
        w = csv.DictWriter(f, fieldnames=csv_fieldnames)
        if write_header:
            w.writeheader()
        return f, w

    for p in img_paths:
        try:
            res = process_one_image(p, out_dir=out)
            row = _to_row(res, image_path=p)
        except Exception as e:
            import traceback

            row = {
                "image_path": str(p),
                "status": "error",
                "error": repr(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
            }

        # Keep in-memory copy
        rows.append(row)

        # Write incrementally so we don't lose progress if the run stops mid-way.
        try:
            f, w = _ensure_header_and_writer()
            try:
                # Protect CSV from rewrites: unknown keys go into extra_json.
                extra = {k: row[k] for k in row.keys() if k not in csv_fieldnames}
                if extra:
                    # keep existing extra_json if present
                    prev = row.get(extra_col)
                    if prev:
                        try:
                            prev_obj = json.loads(prev) if isinstance(prev, str) else prev
                        except Exception:
                            prev_obj = {"prev": str(prev)}
                        if isinstance(prev_obj, dict):
                            prev_obj.update(extra)
                            extra = prev_obj
                    row[extra_col] = json.dumps(extra, ensure_ascii=False)

                safe_row = {k: row.get(k, "") for k in csv_fieldnames}
                w.writerow(safe_row)
            finally:
                f.close()
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
