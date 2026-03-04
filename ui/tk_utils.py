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

def left_panel_photo(
    img_rgb: np.ndarray,
    *,
    max_side: int,
    grid_on: bool,
    step: int,
    roi: tuple[int, int, int, int] | None,
) -> ImageTk.PhotoImage:
    """Render the LEFT panel as a PhotoImage using the ORIGINAL RGB image.

    Grid/ROI are drawn AFTER resizing for display, but grid positions/labels are computed
    in ORIGINAL coordinates so labels remain clean multiples (200, 400, ...).
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("_left_panel_photo expects RGB image")

    im0 = Image.fromarray(img_rgb)
    w0, h0 = im0.size
    s0 = max(w0, h0)

    if s0 > max_side:
        scale = max_side / float(s0)
        w = int(round(w0 * scale))
        h = int(round(h0 * scale))
        im = im0.resize((w, h), resample=Image.Resampling.NEAREST)
    else:
        scale = 1.0
        w, h = w0, h0
        im = im0.copy()

    if grid_on or (roi is not None):
        dr = ImageDraw.Draw(im)

        # helper: draw text with black background box for readability
        def _text_box(x: int, y: int, text: str) -> None:
            # small padding box
            tw = int(dr.textlength(text))
            th = 12
            pad = 3
            x0b = max(0, x - pad)
            y0b = max(0, y - pad)
            x1b = min(w - 1, int(x + tw) + pad)
            y1b = min(h - 1, y + th + pad)
            dr.rectangle([x0b, y0b, x1b, y1b], fill=0)
            dr.text((x, y), text, fill=(220, 220, 220))

        if grid_on:
            step0 = max(10, int(step))  # ORIGINAL-coordinate step

            # draw grid lines at ORIGINAL multiples, mapped to display
            for x0 in range(0, w0, step0):
                x = int(round(x0 * scale))
                dr.line([(x, 0), (x, h)], fill=(220, 220, 220), width=2)
            for y0 in range(0, h0, step0):
                y = int(round(y0 * scale))
                dr.line([(0, y), (w, y)], fill=(220, 220, 220), width=2)

            # labels every 1 step (0, 200, 400, ...)
            for x0 in range(0, w0, step0):
                x = int(round(x0 * scale))
                _text_box(x + 4, 4, str(x0))
            for y0 in range(0, h0, step0):
                y = int(round(y0 * scale))
                _text_box(4, y + 4, str(y0))

        if roi is not None:
            x0, y0, x1, y1 = roi
            x0 = int(round(x0 * scale))
            y0 = int(round(y0 * scale))
            x1 = int(round(x1 * scale))
            y1 = int(round(y1 * scale))
            dr.rectangle([x0, y0, x1, y1], outline=(125, 125, 255), width=3)

    return ImageTk.PhotoImage(im)
