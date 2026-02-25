from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class BrainMaskUIResult:
    mask: np.ndarray  # bool, same HxW as input gray/img2
    params: dict[str, Any]


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


def _largest_component_near_center(bin_u8: np.ndarray, *, center: tuple[int, int], seed_r: int) -> np.ndarray:
    """Pick a single connected component that overlaps a center seed region; fallback to largest."""
    fg = (bin_u8 > 0).astype(np.uint8)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if num <= 1:
        return np.zeros_like(fg, dtype=np.uint8)

    cy, cx = center
    seed = np.zeros_like(fg, dtype=np.uint8)
    cv2.circle(seed, (int(cx), int(cy)), int(seed_r), 1, thickness=-1)

    best_idx = -1
    best_score = -1
    best_area = -1

    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        comp = (lab == idx)
        score = int((comp & (seed > 0)).sum())
        if score > best_score or (score == best_score and area > best_area):
            best_score = score
            best_area = area
            best_idx = idx

    if best_idx < 1 or best_score <= 0:
        # no overlap with center seed; pick the largest component
        best_idx = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)

    return (lab == best_idx).astype(np.uint8)


def _fill_holes_u8(mask_u8: np.ndarray) -> np.ndarray:
    """Fill internal holes (uint8 0/1)."""
    m = (mask_u8 > 0).astype(np.uint8) * 255
    h, w = m.shape
    flood = m.copy()
    tmp = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, tmp, seedPoint=(0, 0), newVal=255)
    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(m, flood_inv)
    return (filled > 0).astype(np.uint8)


def _perimeter_px(mask_u8: np.ndarray) -> float:
    m = (mask_u8 > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0
    return float(sum(cv2.arcLength(c, True) for c in cnts))


def brain_mask_threshold_ui(
    gray_used: np.ndarray,
    img2: np.ndarray,
    *,
    pad: int = 50,
    window: str = "Brain mask (threshold)",
    seed_r: int = 40,
    close_r: int = 7,
    open_r: int = 3,
    alpha: float = 0.35,
    max_side: int = 700,
) -> BrainMaskUIResult | None:
    """Semi-manual brain mask via threshold slider.

    - user drags a threshold
    - we binarize dark tissue
    - automatically select one main component (prefer overlap with a center seed)
    - fill holes + smooth + add `pad` px margin

    Controls:
      - Slider: threshold
      - ENTER: accept
      - ESC: cancel (returns None)
      - R: reset threshold to Otsu

    Returns BrainMaskUIResult(mask(bool), params(dict)) or None if cancelled.
    """
    rgb_full = _ensure_rgb_u8(img2)
    # Threshold on the original image (more stable than preprocessed gray_used)
    gray_u8_full = cv2.cvtColor(rgb_full, cv2.COLOR_RGB2GRAY)

    # Optional sanity check: if gray_used is provided, it must match img2 geometry
    gchk = np.asarray(gray_used)
    if gchk.ndim == 2:
        if gchk.shape[:2] != gray_u8_full.shape[:2]:
            raise ValueError("gray_used and img2 must have the same height/width")
    elif gchk.ndim != 0:
        # allow callers to pass None-like / empty placeholders; otherwise reject weird shapes
        raise ValueError("gray_used must be a 2D array if provided")

    h_full, w_full = gray_u8_full.shape

    # Strong downsample for UI responsiveness
    max_side = int(max_side)
    if max_side <= 0:
        max_side = max(h_full, w_full)

    scale = min(1.0, float(max_side) / float(max(h_full, w_full)))
    if scale < 1.0:
        new_w = max(1, int(round(w_full * scale)))
        new_h = max(1, int(round(h_full * scale)))
        gray_u8 = cv2.resize(gray_u8_full, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.resize(rgb_full, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        gray_u8 = gray_u8_full
        rgb = rgb_full

    h, w = gray_u8.shape

    # Scale UI-space geometry params (pad/seed/morph radii)
    def _sc(v: int) -> int:
        return int(max(1, round(int(v) * scale))) if v > 0 else 0

    pad_ui = _sc(pad)
    seed_r_ui = _sc(seed_r)
    close_r_ui = _sc(close_r)
    open_r_ui = _sc(open_r)

    # initial threshold via Otsu (on inverted? no, we want dark as FG => gray < thr)
    try:
        otsu_thr, _ = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr0 = int(otsu_thr)
    except Exception:
        thr0 = 170

    thr0 = int(np.clip(thr0, 0, 255))

    after_ids: list[int] = []  # local safety, window-scoped
    state = {
        "thr": thr0,
        "need_redraw": True,
        "accepted": False,
        "result": None,
    }

    # UI window
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def _on_trackbar(v: int) -> None:
        state["thr"] = int(v)
        state["need_redraw"] = True

    cv2.createTrackbar("thr (dark<)", window, thr0, 255, _on_trackbar)

    def compute_mask(thr: int) -> np.ndarray:
        # Dark tissue foreground
        bin_fg = (gray_u8 < int(thr)).astype(np.uint8)

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
        if pad_ui > 0:
            kr = int(pad_ui)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kr + 1, 2 * kr + 1))
            core = cv2.dilate(core, k)

        return (core > 0).astype(np.uint8)

    def render(thr: int) -> tuple[np.ndarray, np.ndarray]:
        mask_u8 = compute_mask(thr)
        disp = rgb.copy()  # RGB

        # no fill overlay (keeps tissue texture visible). We only draw the boundary.

        # draw boundary with high contrast (black underlay + colored line)
        m = (mask_u8 > 0).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            # OpenCV expects BGR, so convert for drawing then back
            bgr = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
            # underlay
            cv2.drawContours(bgr, cnts, -1, (0, 0, 0), 5)
            # main line (magenta)
            cv2.drawContours(bgr, cnts, -1, (255, 0, 255), 2)
            disp = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # HUD text with black backing
        area = int(mask_u8.sum())
        perim = _perimeter_px(mask_u8)
        msg1 = f"thr={thr} | area={area} px | perim={perim:.1f} px | pad={pad}px (ui={pad_ui}px) | scale={scale:.3f}"
        msg2 = "Boundary shown (magenta). ENTER accept | ESC cancel | R reset(otsu)"

        bgr = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
        # background rectangles
        cv2.rectangle(bgr, (8, 8), (8 + 820, 8 + 28), (0, 0, 0), thickness=-1)
        cv2.rectangle(bgr, (8, 40), (8 + 820, 40 + 28), (0, 0, 0), thickness=-1)
        cv2.putText(bgr, msg1, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(bgr, msg2, (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mask_u8

    disp_rgb, mask_u8 = render(state["thr"])

    while True:
        if state["need_redraw"]:
            disp_rgb, mask_u8 = render(state["thr"])
            state["need_redraw"] = False

        # show
        cv2.imshow(window, cv2.cvtColor(disp_rgb, cv2.COLOR_RGB2BGR))
        k = cv2.waitKey(20) & 0xFF

        if k in (13, 10):  # Enter
            state["accepted"] = True
            break
        if k == 27:  # ESC
            state["accepted"] = False
            break
        if k in (ord("r"), ord("R")):
            cv2.setTrackbarPos("thr (dark<)", window, thr0)
            state["thr"] = thr0
            state["need_redraw"] = True

    cv2.destroyWindow(window)

    if not state["accepted"]:
        return None

    # Upscale mask back to full resolution (if UI was downsampled)
    if scale < 1.0:
        mask_u8_full = cv2.resize(mask_u8.astype(np.uint8), (w_full, h_full), interpolation=cv2.INTER_NEAREST)
    else:
        mask_u8_full = mask_u8.astype(np.uint8)

    mask_bool = (mask_u8_full > 0)
    params: dict[str, Any] = {
        "accepted": True,
        "thr": int(state["thr"]),
        "pad": int(pad),
        "seed_r": int(seed_r),
        "close_r": int(close_r),
        "open_r": int(open_r),
        "max_side": int(max_side),
        "scale": float(scale),
        "area_px": int(mask_u8_full.sum()),
        "perim_px": float(_perimeter_px(mask_u8_full)),
    }

    return BrainMaskUIResult(mask=mask_bool, params=params)