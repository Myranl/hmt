from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

def _gray_to_u8(gray: np.ndarray) -> np.ndarray:
    """Accept float [0..1] or uint8 and return uint8 grayscale."""
    if gray.dtype == np.uint8:
        return gray
    g = np.asarray(gray)
    if g.ndim != 2:
        raise ValueError("gray must be a 2D array")
    g = np.clip(g, 0.0, 1.0)
    return (g * 255.0 + 0.5).astype(np.uint8)


def _ensure_rgb_u8(img: np.ndarray) -> np.ndarray:
    """Return uint8 RGB image for display."""
    a = np.asarray(img)
    if a.ndim == 2:
        a = cv2.cvtColor(_gray_to_u8(a), cv2.COLOR_GRAY2BGR)
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    elif a.ndim == 3 and a.shape[2] == 3:
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        # assume already RGB
    else:
        raise ValueError("img2 must be HxW or HxWx3")
    return a


def _perimeter_px(mask_u8: np.ndarray) -> float:
    m = (mask_u8 > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0
    return float(sum(cv2.arcLength(c, True) for c in cnts))


def _safe_display_overlay(window: str, msg: str, ms: int = 1000) -> None:
    """Best-effort overlay text. Some OpenCV builds lack displayOverlay or may error."""
    try:
        fn = getattr(cv2, "displayOverlay", None)
        if fn is None:
            return
        fn(window, msg, int(ms))
    except Exception:
        # Never crash the UI because of a HUD.
        return
