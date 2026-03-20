import numpy as np
from ui.roi.roi_picker_ui import run_roi_ui

# Defaults aligned with `bins_sketch_ui` (3-bin tuning is opened from pick-hippocampus when needed).
_DEFAULT_SMALL_N = 900


def run_ui_and_get_params(_gray_used: np.ndarray, img2: np.ndarray, *, t1_init: float, t2_init: float) -> dict | None:
    """Pick ROI + grid settings only.

    Initial 3-bin thresholds (t1/t2) and small-component settings use defaults; the user can open
    the full 3-bin sketch UI from the pick-hippocampus step (closes pick → bins → pick reopens).

    Returns a dict with keys:
      t1,t2,x0,y0,x1,y1,small_to_gray,small_N,grid_on,grid_step
    or None if cancelled.

    ``_gray_used`` is kept for a stable pipeline call signature (contrast is applied before ROI UI).
    """
    roi_res = run_roi_ui(img_rgb=img2)
    if roi_res is None:
        return None

    roi = (int(roi_res["x0"]), int(roi_res["y0"]), int(roi_res["x1"]), int(roi_res["y1"]))
    grid_on = bool(roi_res["grid_on"])
    grid_step = int(roi_res["grid_step"])

    return {
        "t1": float(t1_init),
        "t2": float(t2_init),
        "x0": int(roi[0]),
        "y0": int(roi[1]),
        "x1": int(roi[2]),
        "y1": int(roi[3]),
        "small_to_gray": True,
        "small_N": int(_DEFAULT_SMALL_N),
        "grid_on": bool(grid_on),
        "grid_step": int(grid_step),
    }