import numpy as np
from PIL import Image, ImageDraw
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk
import numpy as np
from PIL import Image, ImageDraw
import cv2

import tkinter as tk
from tkinter import ttk
from PIL import ImageTk

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
