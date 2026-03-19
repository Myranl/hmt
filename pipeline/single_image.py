from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import cv2
from PIL import Image

from ui.pick_components import pick_hippocampus_and_split_by_midline
from ui.review_ui import review_and_maybe_edit  # type: ignore
from segmentation.postprocess import smooth_fill_mask  # type: ignore
from preproc.retina import downsample_rgb_cv2, enhance_contrast_and_smooth, retina_subtract_local_mean
from ui.roi.run_ui_and_get_params import run_ui_and_get_params
from preproc.quantize import sketch_three_bins, small_components_to_gray, apply_midline_cut_to_sketch

from viz.overlay import _overlay_masks_on_original
from ui.brain.threshold_ui import brain_mask_auto, brain_mask_threshold_ui
from ui.brain.brain_outline_UI import brain_outline_ui, overlay_mask_outline_rgb
from ui.brain.fill_voids_ui import fill_voids_ui
from ui.brain.hemisphere import midline_ui
from ui.brain.contour_editor_ui import edit_contour_ui

from preproc.resize import midline_params_to_orig, ds_scale, roi_ds_to_orig

def process_one_image(
    image_path: str | Path,
    *,
    out_dir: str | Path,
    downsample_factor: float = 2.0,
    mean_sigma: float = 8.0,
    gain: float = 3.0,
    debug_show_overlay: bool = False,
) -> dict[str, Any]:
    """Process a single image and return a result row.

    Notes
    - If interactive=False, `default_params` must be provided.
    - Writes overlay PNG into out_dir.
    """
    image_path = Path(image_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    img = np.array(Image.open(image_path).convert("RGB"))
    img2 = downsample_rgb_cv2(img, factor=float(downsample_factor))

    H_ds, W_ds = img2.shape[:2]
    orig_h, orig_w = img.shape[:2]
    sx, sy = ds_scale((orig_h, orig_w), (H_ds, W_ds))

    # --- Step 0: fast brain mask (auto Otsu + morphology, no UI) ---
    bm_res = brain_mask_auto(img2, pad=50)
    brain_mask_ds = bm_res.mask.astype(bool)  # bool mask on img2 (downsample)
    brain_mask_params = bm_res.params  # dict

    # Visualization image for all downstream UIs: gray-out everything outside the brain.
    img2_vis = img2.copy()
    img2_vis[~brain_mask_ds] = (230, 230, 230)

    # Processing image: neutralize outside-brain pixels so they don't affect contrast normalization.
    img2_proc = img2.copy()
    img2_proc[~brain_mask_ds] = (255, 255, 255)

    # Step 1: refine / override brain mask with outline UI.
    # If user clicks "Re-run threshold", Brain outline closes; we open threshold here, then re-open Brain outline.
    while True:
        brain_mask_outline, brain_outline_params = brain_outline_ui(img2_vis, init_mask=brain_mask_ds)
        if brain_outline_params.get("rerun_threshold"):
            gray0 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            bm_res = brain_mask_threshold_ui(gray0, img2, pad=50)
            if bm_res is None:
                continue
            brain_mask_ds = bm_res.mask.astype(bool)
            img2_vis = img2.copy()
            img2_vis[~brain_mask_ds] = (230, 230, 230)
            continue
        break
    brain_mask_step1 = brain_mask_outline.astype(bool)

    # If user checked "Non-complete contour", run contour editor as the next step (fix breaks/gaps).
    if brain_outline_params.get("non_complete_contour", False):
        edited_mask, edit_params = edit_contour_ui(brain_mask_step1.astype(np.uint8) * 255, img2_vis)
        if edit_params.get("accepted", False):
            brain_mask_step1 = edited_mask.astype(bool)
        brain_outline_params["corrected"] = edit_params.get("edited", False)
        brain_outline_params["correction_type"] = edit_params.get("correction_type", "none")

    # Step 1b: fill internal voids inside the fixed contour.
    # `fill_voids_ui` already constrains operations to the interior of the contour.
    brain_mask_filled_u8, fill_params = fill_voids_ui(img2_vis, brain_mask_step1.astype(np.uint8) * 255)
    brain_mask_final = brain_mask_filled_u8.astype(bool)

    midline_params = midline_ui(img2_vis, brain_mask_final, pad=50)
    midline_params_orig = midline_params_to_orig(midline_params, sx, sy)

    stem = image_path.stem
    brain_outline_preview = overlay_mask_outline_rgb(img2_vis, brain_mask_final.astype(np.uint8), color=(0, 255, 0), thickness=2)
    brain_outline_path = out_dir / f"{stem}__brain_outline.png"
    Image.fromarray(brain_outline_preview).save(brain_outline_path)

    # --- Step 2: hippocampus-oriented preprocessing restricted by the final brain mask ---
    gray_base = enhance_contrast_and_smooth(img2_proc, clahe_clip=0.10, clahe_kernel=128, smooth_sigma=8.0)
    gray_used = retina_subtract_local_mean(gray_base, mean_sigma=float(mean_sigma), gain=float(gain),  p_lo=1.0, p_hi=99.0)
    # Make outside-brain pixels neutral in the working grayscale too.
    gray_used[~brain_mask_final] = 0.5

    params = run_ui_and_get_params(gray_used, img2_vis, t1_init=0.33, t2_init=0.66)
    if params is None:
        return {"image_path": str(image_path), "status": "skipped"}

    # clamp ROI in downsampled coords
    W = int(img2.shape[1])
    H = int(img2.shape[0])
    x0 = max(0, min(W - 1, int(params["x0"])))
    y0 = max(0, min(H - 1, int(params["y0"])))
    x1 = max(x0 + 1, min(W, int(params["x1"])))
    y1 = max(y0 + 1, min(H, int(params["y1"])))

    roi_ds = (x0, y0, x1, y1)
    roi_orig = roi_ds_to_orig(roi_ds, sx, sy)

    # recompute ROI sketch
    gray_roi = gray_used[y0:y1, x0:x1]

    _, sketch_u8 = sketch_three_bins(gray_roi, t1=float(params["t1"]), t2=float(params["t2"]))
    brain_roi = brain_mask_final[y0:y1, x0:x1]

    sketch_u8 = apply_midline_cut_to_sketch(sketch_u8,brain_roi=brain_roi,midline_params=midline_params,roi_x0=int(x0), roi_y0=int(y0), thickness=9)
    sketch_u8[~brain_roi] = 127
    # stem = image_path.stem  # removed duplicate stem assignment

    if bool(params.get("small_to_gray", False)):
        sketch_u8 = small_components_to_gray(sketch_u8, min_area=int(params.get("small_N", 0)))

    bg_roi = img2_vis[y0:y1, x0:x1]

    left_roi_sel, right_roi_sel, sketch_after = pick_hippocampus_and_split_by_midline(
        sketch_u8_roi=sketch_u8, bg_roi_rgb=bg_roi,
        midline_params=midline_params, roi_x0=int(x0), roi_y0=int(y0),)

    left_roi_sel, right_roi_sel = review_and_maybe_edit(
        img2_rgb=img2_vis, sketch_u8_roi=sketch_after, bg_roi=bg_roi,
        roi=(x0, y0, x1, y1), left_roi_sel=left_roi_sel, right_roi_sel=right_roi_sel,
    brain_mask_full=brain_mask_final,)

    # Use same smooth_fill params as in review_ui so saved overlay matches what user saw
    left_roi_sel = smooth_fill_mask(left_roi_sel, close_ksize=25, open_ksize=7, blur_sigma=2.0)
    right_roi_sel = smooth_fill_mask(right_roi_sel, close_ksize=25, open_ksize=7, blur_sigma=2.0)

    # build full downsampled masks
    left_ds = np.zeros((H, W), dtype=np.uint8)
    right_ds = np.zeros((H, W), dtype=np.uint8)
    left_ds[y0:y1, x0:x1] = left_roi_sel
    right_ds[y0:y1, x0:x1] = right_roi_sel

    # Constrain hippocampus masks to the final brain mask
    left_ds[~brain_mask_final] = 0
    right_ds[~brain_mask_final] = 0

    # map to original image size
    left_orig = cv2.resize(left_ds, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    right_orig = cv2.resize(right_ds, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    def _mask_area_perim(mask_u8: np.ndarray) -> tuple[int, float]:
        """Return (area_px, perim_px) for a binary mask (0/1 or 0/255)."""
        m = (mask_u8 > 0).astype(np.uint8)
        area = int(m.sum())
        if area == 0:
            return 0, 0.0
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            return area, 0.0
        cnt = max(cnts, key=cv2.contourArea)
        perim = float(cv2.arcLength(cnt, True))
        return area, perim

    hip_left_area_px, hip_left_perim_px = _mask_area_perim(left_orig)
    hip_right_area_px, hip_right_perim_px = _mask_area_perim(right_orig)
    brain_orig_u8 = cv2.resize(
        (brain_mask_final.astype(np.uint8) * 255),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    )
    brain_area_px, brain_perim_px = _mask_area_perim(brain_orig_u8)

    def _f(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return float(default)

    area_scale = float(sx) * float(sy)
    perim_scale = (float(sx) + float(sy)) * 0.5
    midline_area_left_px = int(round(_f(midline_params.get("area_left_px")) * area_scale))
    midline_area_right_px = int(round(_f(midline_params.get("area_right_px")) * area_scale))
    midline_perimeter_left_px = _f(midline_params.get("perimeter_left_px")) * perim_scale
    midline_perimeter_right_px = _f(midline_params.get("perimeter_right_px")) * perim_scale
    non_complete_contour = bool(brain_outline_params.get("non_complete_contour", False))
    contour_corrected = bool(brain_outline_params.get("corrected", False))

    overlay = _overlay_masks_on_original(img, left_orig, right_orig, alpha=0.30)

    overlay_path = out_dir / f"{stem}__hippocampus_overlay.png"
    Image.fromarray(overlay).save(overlay_path)

    if debug_show_overlay:
        ov_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.namedWindow("Overlay (red=LEFT, blue=RIGHT)", cv2.WINDOW_NORMAL)
        cv2.imshow("Overlay (red=LEFT, blue=RIGHT)", ov_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "image_path": str(image_path),
        "img_name": image_path.name,
        "status": "ok",
        "accepted": "ok",
        "overlay_path": str(overlay_path),
        # Canonical metrics in ORIGINAL-image pixels
        "brain_area_px": brain_area_px,
        "brain_perim_px": brain_perim_px,
        "midline_area_left_px": midline_area_left_px,
        "midline_area_right_px": midline_area_right_px,
        "midline_perimeter_left_px": midline_perimeter_left_px,
        "midline_perimeter_right_px": midline_perimeter_right_px,
        "non_complete_contour": non_complete_contour,
        "contour_corrected": contour_corrected,
        "hipp_area_left_px": hip_left_area_px,
        "hipp_area_right_px": hip_right_area_px,
        "hipp_perimeter_left_px": hip_left_perim_px,
        "hipp_perimeter_right_px": hip_right_perim_px,
        "brain_outline_path": str(brain_outline_path),
        "brain_outline_params": brain_outline_params,
        "brain_mask_params": brain_mask_params,
        "geometry": {
            "orig_shape_hw": (orig_h, orig_w),
            "ds_shape_hw": (H_ds, W_ds),
            "sx": sx,
            "sy": sy,
            "roi_ds": roi_ds,
            "roi_orig": roi_orig,
            "downsample_factor": float(downsample_factor),
        },
        "midline_params_ds": midline_params,
        "midline_params_orig": midline_params_orig,
        "hip_left_area_px": hip_left_area_px,
        "hip_left_perim_px": hip_left_perim_px,
        "hip_right_area_px": hip_right_area_px,
        "hip_right_perim_px": hip_right_perim_px,
        "roi_ds": roi_ds,
        "roi_orig": roi_orig,
        "params": params,
    }