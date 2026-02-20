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


def to_photo_u8(gray: np.ndarray, *, max_side: int = 700) -> ImageTk.PhotoImage:
    """Convert a 2D image to a Tk PhotoImage (grayscale), with optional downscaling for display."""
    if gray.dtype != np.uint8:
        g = np.clip(gray, 0.0, 1.0)
        g = (g * 255).astype(np.uint8)
    else:
        g = gray

    im = Image.fromarray(g)
    w, h = im.size
    s = max(w, h)
    if s > max_side:
        scale = max_side / float(s)
        im = im.resize((int(round(w * scale)), int(round(h * scale))), resample=Image.Resampling.NEAREST)
    return ImageTk.PhotoImage(im)


# Overlay grid and ROI rectangle
def overlay_grid_and_roi(gray: np.ndarray, *, step: int = 200, roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Return a uint8 image with a light grid drawn on top (for coordinate reading) and optional ROI rectangle."""
    if gray.dtype != np.uint8:
        base = (np.clip(gray, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        base = gray.copy()

    im = Image.fromarray(base)
    dr = ImageDraw.Draw(im)
    w, h = im.size

    step = int(step)
    if step > 0:
        for x in range(0, w, step):
            dr.line([(x, 0), (x, h)], fill=220, width=1)
        for y in range(0, h, step):
            dr.line([(0, y), (w, y)], fill=220, width=1)

        # labels every 2 steps to reduce clutter
        for x in range(0, w, step * 2):
            dr.text((x + 2, 2), str(x), fill=235)
        for y in range(0, h, step * 2):
            dr.text((2, y + 2), str(y), fill=235)

    if roi is not None:
        x0, y0, x1, y1 = map(int, roi)
        dr.rectangle([x0, y0, x1, y1], outline=255, width=2)

    return np.array(im)




