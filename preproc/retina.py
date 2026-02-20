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


def downsample_rgb_cv2(img_rgb: np.ndarray, *, factor: float = 2.0) -> np.ndarray:
    if factor <= 0:
        raise ValueError("factor must be > 0")
    h, w = img_rgb.shape[:2]
    new_w = max(1, int(round(w / factor)))
    new_h = max(1, int(round(h / factor)))
    return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def enhance_contrast_and_smooth(
    img_rgb: np.ndarray,
    *,
    clahe_clip: float = 0.10,
    clahe_kernel: int = 128,
    smooth_sigma: float = 8.0,
) -> np.ndarray:
    g = rgb2gray(img_rgb).astype(np.float32)
    g = exposure.equalize_adapthist(g, clip_limit=float(clahe_clip), kernel_size=int(clahe_kernel)).astype(np.float32)
    g = gaussian(g, sigma=float(smooth_sigma), preserve_range=True).astype(np.float32)
    return np.clip(g, 0.0, 1.0)


def retina_subtract_local_mean(
    gray01: np.ndarray,
    *,
    mean_sigma: float = 15.0,
    gain: float = 3.0,
    p_lo: float = 1.0,
    p_hi: float = 99.0,
) -> np.ndarray:
    """gray01 in [0..1] -> (gray-mean)*gain -> robust rescale to [0..1]."""
    g = np.asarray(gray01, dtype=np.float32)
    mean = gaussian(g, sigma=float(mean_sigma), preserve_range=True).astype(np.float32)
    diff = (g - mean) * float(gain)

    m = np.isfinite(diff)
    if not np.any(m):
        return np.zeros_like(diff, dtype=np.float32)

    lo, hi = np.percentile(diff[m], [float(p_lo), float(p_hi)])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.clip(diff, 0.0, 1.0).astype(np.float32)

    out = exposure.rescale_intensity(diff, in_range=(lo, hi), out_range=(0.0, 1.0)).astype(np.float32)
    return out


