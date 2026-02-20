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


def _overlay_masks_on_original(orig_rgb: np.ndarray, left_mask: np.ndarray, right_mask: np.ndarray, *, alpha: float = 0.35) -> np.ndarray:
    """Return RGB image with left(red) and right(blue) masks alpha-blended."""
    out = orig_rgb.astype(np.float32).copy()

    if left_mask is not None:
        m = left_mask.astype(bool)
        out[m, 0] = (1 - alpha) * out[m, 0] + alpha * 255
        out[m, 1] = (1 - alpha) * out[m, 1]
        out[m, 2] = (1 - alpha) * out[m, 2]

    if right_mask is not None:
        m = right_mask.astype(bool)
        out[m, 2] = (1 - alpha) * out[m, 2] + alpha * 255
        out[m, 0] = (1 - alpha) * out[m, 0]
        out[m, 1] = (1 - alpha) * out[m, 1]

    return np.clip(out, 0, 255).astype(np.uint8)

