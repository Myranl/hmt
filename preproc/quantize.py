import numpy as np
from PIL import Image, ImageDraw
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk
import json

from skimage.color import rgb2gray
from skimage import exposure
from skimage.filters import gaussian


def sketch_three_bins(gray01: np.ndarray, *, t1: float = 0.33, t2: float = 0.66) -> tuple[np.ndarray, np.ndarray]:
    """Return (sketch01, sketch_u8).

    sketch01 levels: 0.0 (white), 0.5 (mid), 1.0 (black)
    sketch_u8 levels: 0, 127, 255
    """
    g = np.clip(np.asarray(gray01, dtype=np.float32), 0.0, 1.0)
    if not (0.0 < t1 < t2 < 1.0):
        raise ValueError("Require 0 < t1 < t2 < 1")

    # bins: 0 -> [0,t1), 1 -> [t1,t2), 2 -> [t2,1]
    b = np.digitize(g, [float(t1), float(t2)], right=False).astype(np.int32)
    levels01 = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    levels_u8 = np.array([0, 127, 255], dtype=np.uint8)

    sketch01 = levels01[b]
    sketch_u8 = levels_u8[b]
    return sketch01, sketch_u8



def small_components_to_gray(sketch_u8: np.ndarray, *, min_area: int) -> np.ndarray:
    """Turn small white/black pixel islands into mid-gray (127).

    Operates on a uint8 image with values {0,127,255}.
    Any connected component in the 0-mask or 255-mask with area < min_area becomes 127.
    """
    if min_area <= 0:
        return sketch_u8

    out = sketch_u8.copy()

    for val in (0, 255):
        m = (out == val).astype(np.uint8)  # 0/1
        num, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < int(min_area):
                out[lab == i] = 127

    return out


def apply_midline_cut_to_sketch(
    sketch_u8: np.ndarray,
    *,
    brain_roi: np.ndarray | None = None,
    midline_params: dict | None = None,
    roi_x0: int = 0,
    roi_y0: int = 0,
    thickness: int = 9,
    cut_value: int = 127,
) -> np.ndarray:
    """Paint a "barrier" along the midline polyline to force a split.

    The barrier is drawn into the ROI coordinate system, but the saved midline points are typically
    in full-image XY coordinates. We convert to ROI coords via (roi_x0, roi_y0) offset.

    Expected: midline_params contains key `midline_pts` as a JSON string: [[x,y], [x,y], ...]

    Pixels on the barrier are set to `cut_value` (default mid-gray=127), optionally only inside brain_roi.
    """
    if midline_params is None:
        return sketch_u8

    s = midline_params.get("midline_pts")
    if not s:
        return sketch_u8

    try:
        pts = json.loads(s)
    except Exception:
        return sketch_u8

    if not isinstance(pts, list) or len(pts) < 2:
        return sketch_u8

    h, w = sketch_u8.shape[:2]

    poly: list[list[int]] = []
    for p in pts:
        if not (isinstance(p, (list, tuple)) and len(p) == 2):
            continue
        x = float(p[0]) - float(roi_x0)
        y = float(p[1]) - float(roi_y0)
        poly.append([int(round(x)), int(round(y))])

    if len(poly) < 2:
        return sketch_u8

    poly_np = np.asarray(poly, dtype=np.int32)
    poly_np[:, 0] = np.clip(poly_np[:, 0], 0, w - 1)
    poly_np[:, 1] = np.clip(poly_np[:, 1], 0, h - 1)

    barrier = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(
        barrier,
        [poly_np],
        isClosed=False,
        color=1,
        thickness=int(max(1, thickness)),
        lineType=cv2.LINE_8,
    )

    out = sketch_u8.copy()

    if brain_roi is None:
        out[barrier > 0] = np.uint8(cut_value)
        return out

    inside = brain_roi.astype(bool)
    out[(barrier > 0) & inside] = np.uint8(cut_value)
    return out
