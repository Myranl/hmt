import numpy as np
from ui.review_ui import review_and_maybe_edit  # type: ignore
from segmentation.postprocess import smooth_fill_mask  # type: ignore
from ui.roi.roi_picker_ui import run_roi_ui
from ui.roi.bins_sketch_ui import run_bins_ui

def run_ui_and_get_params(gray_used: np.ndarray, img2: np.ndarray, *, t1_init: float, t2_init: float) -> dict | None:
    """Two-step UI.

    Step 1: pick ROI (drag on original) + grid settings.
    Step 2: tune 3-bin sketch params on the fixed ROI.

    Returns a dict with keys:
      t1,t2,x0,y0,x1,y1,small_to_gray,small_N,grid_on,grid_step
    or None if cancelled.
    """

    # ---- run step 1 ----
    roi_res = run_roi_ui(img_rgb=img2)
    if roi_res is None:
        return None

    roi = (int(roi_res["x0"]), int(roi_res["y0"]), int(roi_res["x1"]), int(roi_res["y1"]))
    grid_on = bool(roi_res["grid_on"])
    grid_step = int(roi_res["grid_step"])

    # ---- run step 2 ----
    bins_res = run_bins_ui(  gray=gray_used, img_rgb=img2, roi=roi, grid_on=grid_on,  grid_step=grid_step,  t1_init=t1_init, t2_init=t2_init  )
    if bins_res is None:
        return None

    out = {
        "t1": float(bins_res["t1"]),
        "t2": float(bins_res["t2"]),
        "x0": int(roi[0]),
        "y0": int(roi[1]),
        "x1": int(roi[2]),
        "y1": int(roi[3]),
        "small_to_gray": bool(bins_res["small_to_gray"]),
        "small_N": int(bins_res["small_N"]),
        "grid_on": bool(grid_on),
        "grid_step": int(grid_step),
    }
    return out