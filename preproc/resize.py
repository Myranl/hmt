import numpy as np
from typing import Any
def ds_scale(orig_shape: tuple[int, int], ds_shape: tuple[int, int]) -> tuple[float, float]:
    oh, ow = orig_shape
    dh, dw = ds_shape
    return (ow / float(dw), oh / float(dh))

def roi_ds_to_orig(roi_ds: tuple[int, int, int, int], sx: float, sy: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = roi_ds
    return (
        int(round(x0 * sx)),
        int(round(y0 * sy)),
        int(round(x1 * sx)),
        int(round(y1 * sy)),
    )

def points_ds_to_orig(pts_xy: np.ndarray, sx: float, sy: float) -> np.ndarray:
    if pts_xy.size == 0:
        return pts_xy
    out = pts_xy.astype(np.float32).copy()
    out[:, 0] *= float(sx)
    out[:, 1] *= float(sy)
    return out


def midline_params_to_orig(mp: dict[str, Any], sx: float, sy: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in mp.items():
        out[k] = v
    # preferred key going forward: midline_pts (list of [x,y])
    pts = None
    if "midline_pts" in mp and mp["midline_pts"]:
        try:
            pts = np.array(mp["midline_pts"], dtype=np.float32)
        except Exception:
            pts = None
    # legacy: midline_xy as string "[x0, y0, x1, y1]" or list
    if pts is None and "midline_xy" in mp and mp["midline_xy"]:
        try:
            if isinstance(mp["midline_xy"], str):
                arr = np.array(eval(mp["midline_xy"]), dtype=np.float32).reshape(-1)
            else:
                arr = np.array(mp["midline_xy"], dtype=np.float32).reshape(-1)
            if arr.size == 4:
                pts = np.array([[arr[0], arr[1]], [arr[2], arr[3]]], dtype=np.float32)
        except Exception:
            pts = None

    if pts is not None and pts.ndim == 2 and pts.shape[1] == 2:
        out["midline_pts_orig"] = points_ds_to_orig(pts, sx, sy).tolist()

    # ROI/crop fields, if present, convert as well (ds -> orig)
    for key in ("crop_x0", "crop_y0", "crop_x1", "crop_y1", "roi_x0", "roi_y0", "roi_x1", "roi_y1"):
        if key in mp:
            try:
                if key.endswith(("x0", "x1")):
                    out[key + "_orig"] = int(round(float(mp[key]) * sx))
                else:
                    out[key + "_orig"] = int(round(float(mp[key]) * sy))
            except Exception:
                pass
    return out