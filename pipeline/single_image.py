from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import cv2
from PIL import Image

# interactive UIs / helpers
from ui.pick_components import select_components_on_background
from ui.review_ui import review_and_maybe_edit  # type: ignore
from segmentation.postprocess import smooth_fill_mask  # type: ignore
from preproc.retina import downsample_rgb_cv2, enhance_contrast_and_smooth, retina_subtract_local_mean
from ui.test_ui import run_ui_and_get_params
from preproc.quantize import sketch_three_bins, small_components_to_gray
from analysis.overlay import _overlay_masks_on_original
from segmentation.brain_outline import brain_outline_ui, overlay_mask_outline_rgb
from segmentation.hemisphere import midline_ui

def process_one_image(
    image_path: str | Path,
    *,
    out_dir: str | Path,
    interactive: bool = True,
    default_params: dict[str, Any] | None = None,
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

    gray_base = enhance_contrast_and_smooth(img2, clahe_clip=0.10, clahe_kernel=128, smooth_sigma=8.0)
    gray_used = retina_subtract_local_mean(gray_base, mean_sigma=float(mean_sigma), gain=float(gain), p_lo=1.0, p_hi=99.0)

    if interactive:
        params = run_ui_and_get_params(gray_used, img2, t1_init=0.33, t2_init=0.66)
        if params is None:
            return {"image_path": str(image_path), "status": "skipped"}
    else:
        if default_params is None:
            raise ValueError("interactive=False requires default_params")
        params = dict(default_params)

    # OpenCV outline UI (run after Tk UI to avoid macOS Tk/Cocoa crashes)
    brain_mask, brain_outline_params = brain_outline_ui(img2)
    midline_params = midline_ui(img2, brain_mask, pad=50)
    stem = image_path.stem
    brain_outline_preview = overlay_mask_outline_rgb(img2, brain_mask, color=(0, 255, 0), thickness=2)
    brain_outline_path = out_dir / f"{stem}__brain_outline.png"
    Image.fromarray(brain_outline_preview).save(brain_outline_path)

    # clamp ROI in downsampled coords
    W = int(img2.shape[1])
    H = int(img2.shape[0])
    x0 = max(0, min(W - 1, int(params["x0"])))
    y0 = max(0, min(H - 1, int(params["y0"])))
    x1 = max(x0 + 1, min(W, int(params["x1"])))
    y1 = max(y0 + 1, min(H, int(params["y1"])))

    # recompute ROI sketch
    gray_roi = gray_used[y0:y1, x0:x1]
    _, sketch_u8 = sketch_three_bins(gray_roi, t1=float(params["t1"]), t2=float(params["t2"]))


    # save brain outline preview (downsampled for quick inspection)
    # stem = image_path.stem  # removed duplicate stem assignment

    if bool(params.get("small_to_gray", False)):
        sketch_u8 = small_components_to_gray(sketch_u8, min_area=int(params.get("small_N", 0)))

    if interactive:
        bg_roi = img2[y0:y1, x0:x1]
        left_roi_sel, sketch_after_left = select_components_on_background(sketch_u8, bg_roi, window="Pick LEFT hippocampus (green)")
        right_roi_sel, sketch_after_both = select_components_on_background(sketch_after_left, bg_roi, window="Pick RIGHT hippocampus (green)")

        roi = (x0, y0, x1, y1)
        left_roi_sel, right_roi_sel = review_and_maybe_edit(
            img2_rgb=img2,
            sketch_u8_roi=sketch_after_both,
            bg_roi=bg_roi,
            roi=roi,
            left_roi_sel=left_roi_sel,
            right_roi_sel=right_roi_sel,
        )
    else:
        # non-interactive: expect selections in params (as masks in ROI coords)
        left_roi_sel = np.asarray(params["left_roi_sel"], dtype=np.uint8)
        right_roi_sel = np.asarray(params["right_roi_sel"], dtype=np.uint8)

    # after ALL OK: make each selection a single smooth filled object
    left_roi_sel = smooth_fill_mask(left_roi_sel, close_ksize=25, open_ksize=7, blur_sigma=2.0)
    right_roi_sel = smooth_fill_mask(right_roi_sel, close_ksize=25, open_ksize=7, blur_sigma=2.0)

    # build full downsampled masks
    left_ds = np.zeros((H, W), dtype=np.uint8)
    right_ds = np.zeros((H, W), dtype=np.uint8)
    left_ds[y0:y1, x0:x1] = left_roi_sel
    right_ds[y0:y1, x0:x1] = right_roi_sel

    # map to original image size
    orig_h, orig_w = img.shape[:2]
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
        "status": "ok",
        "overlay_path": str(overlay_path),
        "brain_outline_path": str(brain_outline_path),
        "brain_outline_params": brain_outline_params,
        "midline_params": midline_params,
        "hip_left_area_px": hip_left_area_px,
        "hip_left_perim_px": hip_left_perim_px,
        "hip_right_area_px": hip_right_area_px,
        "hip_right_perim_px": hip_right_perim_px,
        "roi": (x0, y0, x1, y1),
        "params": params,
    }