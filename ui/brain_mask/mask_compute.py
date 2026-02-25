from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from ui.brain_mask.mask_morphology import _fill_holes_u8, _largest_component_near_center

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
    core = _fill_holes_u8(core)
    if close_r_ui > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_r_ui + 1, 2 * close_r_ui + 1))
        core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, k)
    core = _fill_holes_u8(core)

    # Pad margin
    if pad_eff_ui > 0:
        kr = int(pad_eff_ui)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kr + 1, 2 * kr + 1))
        core = cv2.dilate(core, k)

    return (core > 0).astype(np.uint8)