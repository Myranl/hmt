from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from ui.brain_mask.mask_morphology import _fill_holes, _largest_component_near_center, _largest_component
from ui.brain_mask.mask_utils import _gray_to_u8, _odd, _put_text_box, smooth_contour
import cv2
import numpy as np

def compute_mask(
    gray_u8: np.ndarray,
    *,
    thr: int,
    pad_eff_ui: int,
    close_r_ui: int,
    open_r_ui: int,
    seed_r_ui: int,
    ) -> np.ndarray:
    # Dark tissue foreground
    bin_fg = (gray_u8 < int(thr)).astype(np.uint8)
    h, w = gray_u8.shape
    # Smooth speckle
    if close_r_ui > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_r_ui + 1, 2 * close_r_ui + 1))
        bin_fg = cv2.morphologyEx(bin_fg, cv2.MORPH_CLOSE, k)
    if open_r_ui > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_r_ui + 1, 2 * open_r_ui + 1))
        bin_fg = cv2.morphologyEx(bin_fg, cv2.MORPH_OPEN, k)

    # Component selection
    center = (h // 2, w // 2)
    core = _largest_component_near_center(bin_fg, center=center, seed_r=int(seed_r_ui))
    if core.sum() == 0:
        return core

    # Fill holes and smooth again
    core = _fill_holes(core, binary=True)
    if close_r_ui > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_r_ui + 1, 2 * close_r_ui + 1))
        core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, k)
    core = _fill_holes(core, binary=True)

    # Pad margin
    if pad_eff_ui > 0:
        kr = int(pad_eff_ui)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kr + 1, 2 * kr + 1))
        core = cv2.dilate(core, k)

    return (core > 0).astype(np.uint8)

def brain_mask_from_threshold(
    img_rgb_or_gray: np.ndarray,
    *,
    thr: int,
    invert: bool = True,
    pre_close: int = 11,
    pre_open: int = 5,
    min_area: int = 20000,
    fill_holes: bool = True,
) -> np.ndarray:
    """Binary brain mask from a single threshold.

    - Assumes background is bright and tissue is darker.
    - If `invert=True`, we treat 'dark' as foreground by using (gray < thr).
    Returns boolean mask.
    """
    if img_rgb_or_gray.ndim == 3:
        g = cv2.cvtColor(img_rgb_or_gray, cv2.COLOR_RGB2GRAY)
    else:
        g = _gray_to_u8(img_rgb_or_gray)

    if invert:
        m = (g < int(thr)).astype(np.uint8) * 255
    else:
        m = (g > int(thr)).astype(np.uint8) * 255

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(pre_close), int(pre_close)))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(pre_open), int(pre_open)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)

    m = _largest_component(m)
    if int((m > 0).sum()) < int(min_area):
        return np.zeros(m.shape, dtype=bool)

    if fill_holes:
        m = _fill_holes(m, binary=False)

    return (m > 0)