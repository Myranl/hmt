from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

def _gray_to_u8(gray: np.ndarray) -> np.ndarray:
    """Accept float [0..1] or uint8 and return uint8 grayscale."""
    if gray.dtype == np.uint8:
        return gray
    g = np.asarray(gray, dtype=np.float32)
    # robust clamp if someone passes weird ranges
    gmin, gmax = np.nanpercentile(g, [1, 99])
    if not np.isfinite(gmin) or not np.isfinite(gmax) or gmax <= gmin:
        gmin, gmax = float(np.nanmin(g)), float(np.nanmax(g))
        if not np.isfinite(gmin) or not np.isfinite(gmax) or gmax <= gmin:
            return np.zeros_like(g, dtype=np.uint8)
    g = (g - gmin) / (gmax - gmin)
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

def _odd(x: int) -> int:
    x = int(x)
    if x < 1:
        return 1
    return x if (x % 2 == 1) else (x + 1)


def _put_text_box(img_bgr, text: str, org: tuple[int, int], *, font_scale: float = 0.7, thickness: int = 2,
                  pad: int = 6):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base_t = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = int(org[0]), int(org[1])
    x0 = max(0, x - pad)
    y0 = max(0, y - th - pad)
    x1 = min(img_bgr.shape[1] - 1, x + tw + pad)
    y1 = min(img_bgr.shape[0] - 1, y + base_t + pad)
    cv2.rectangle(img_bgr, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.putText(img_bgr, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def smooth_contour(cnt: np.ndarray, k: int) -> np.ndarray:
    # cnt: (N,1,2)
    if cnt is None or len(cnt) < 5:
        return cnt
    k = _odd(k)
    pts = cnt[:, 0, :].astype(np.float32)
    if pts.shape[0] < k:
        return cnt
    # circular smoothing
    pad = k // 2
    pts_pad = np.vstack([pts[-pad:], pts, pts[:pad]])
    kernel = np.ones((k,), dtype=np.float32) / float(k)
    xs = np.convolve(pts_pad[:, 0], kernel, mode="valid")
    ys = np.convolve(pts_pad[:, 1], kernel, mode="valid")
    sm = np.stack([xs, ys], axis=1)
    sm = np.round(sm).astype(np.int32)
    sm = sm.reshape((-1, 1, 2))
    return sm