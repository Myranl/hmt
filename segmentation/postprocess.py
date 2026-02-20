import numpy as np
from PIL import Image, ImageDraw
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk


def _fill_holes_u8(mask_u8_255: np.ndarray) -> np.ndarray:
    """Fill holes in a binary mask (uint8 0/255)."""
    m = (mask_u8_255 > 0).astype(np.uint8) * 255
    h, w = m.shape[:2]
    # flood fill background from border
    flood = m.copy()
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ffmask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(m, flood_inv)
    return filled


def smooth_fill_mask(mask01: np.ndarray, *, close_ksize: int = 21, open_ksize: int = 7, blur_sigma: float = 2.5) -> np.ndarray:
    """Connect segments, fill holes, and smooth edges. Input: 0/1 uint8. Output: 0/1 uint8."""
    m = (mask01 > 0).astype(np.uint8) * 255

    # connect gaps
    if close_ksize and close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    # remove tiny spurs
    if open_ksize and open_ksize > 1:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k2)

    # fill internal holes
    m = _fill_holes_u8(m)

    # smooth boundary by blurring and re-thresholding
    if blur_sigma and blur_sigma > 0:
        mf = m.astype(np.float32) / 255.0
        mf = cv2.GaussianBlur(mf, (0, 0), blur_sigma)
        m = (mf >= 0.5).astype(np.uint8) * 255

    return (m > 0).astype(np.uint8)
