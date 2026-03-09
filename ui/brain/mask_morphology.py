from __future__ import annotations

import cv2
import numpy as np


def _pick_component(
    bin_u8: np.ndarray,
    *,
    center: tuple[int, int] | None = None,
    seed_r: int = 0,
    out_255: bool = False,
) -> np.ndarray:
    """Pick a single connected component.

    - If `center` is provided and `seed_r > 0`, prefer a component overlapping a circular seed around `center`.
      If no component overlaps the seed, fall back to the largest component.
    - Otherwise, pick the largest component.

    Input can be 0/1 or 0/255; output is 0/1 by default, or 0/255 if `out_255=True`.
    """
    fg = (bin_u8 > 0).astype(np.uint8)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if num <= 1:
        return np.zeros_like(fg if not out_255 else bin_u8, dtype=np.uint8)

    best_idx: int

    if center is not None and int(seed_r) > 0:
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
            best_idx = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
    else:
        best_idx = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)

    out = (lab == best_idx).astype(np.uint8)
    if out_255:
        out = out * 255
    return out

def _largest_component_near_center(bin_u8: np.ndarray, *, center: tuple[int, int], seed_r: int) -> np.ndarray:
    """Back-compat wrapper: returns 0/1 mask."""
    return _pick_component(bin_u8, center=center, seed_r=seed_r, out_255=False)

def _largest_component(mask_u8: np.ndarray) -> np.ndarray:
    """mask_u8 is 0/255. Return 0/255 largest CC."""
    return _pick_component(mask_u8, center=None, seed_r=0, out_255=True)

def _fill_holes(mask_u8: np.ndarray, *, binary: bool = True) -> np.ndarray:
    """Fill internal holes using flood fill.

    Input can be 0/1 or 0/255. Output is 0/1 mask if `binary=True`, or 0/255 mask if `binary=False`.
    """
    m255 = (mask_u8 > 0).astype(np.uint8) * 255
    h, w = m255.shape
    flood = m255.copy()
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff, seedPoint=(0, 0), newVal=255)
    flood_inv = cv2.bitwise_not(flood)
    filled255 = cv2.bitwise_or(m255, flood_inv)

    if binary:
        return (filled255 > 0).astype(np.uint8)
    else:
        return filled255

def _nearest_foreground_xy(m_u8_0_255: np.ndarray, x: int, y: int, *, r: int) -> tuple[int, int] | None:
    """Find nearest foreground pixel (value>0) within radius r. Returns (x, y) or None."""
    h, w = m_u8_0_255.shape[:2]
    r = int(max(0, r))
    if r == 0:
        return None
    x0 = max(0, x - r)
    x1 = min(w - 1, x + r)
    y0 = max(0, y - r)
    y1 = min(h - 1, y + r)
    roi = m_u8_0_255[y0 : y1 + 1, x0 : x1 + 1]
    ys, xs = np.where(roi > 0)
    if ys.size == 0:
        return None
    dx = xs.astype(np.int32) + x0 - x
    dy = ys.astype(np.int32) + y0 - y
    d2 = dx * dx + dy * dy
    i = int(np.argmin(d2))
    return int(xs[i] + x0), int(ys[i] + y0)


def _connected_component_from_seed(mask_u8: np.ndarray, x: int, y: int, *, search_r: int = 12) -> np.ndarray:
    """Return the connected component (uint8 0/255) containing (x,y). If (x,y) not on foreground, search nearest within search_r."""
    m = (mask_u8 > 0).astype(np.uint8) * 255
    h, w = m.shape[:2]
    if h == 0 or w == 0:
        return np.zeros_like(m, dtype=np.uint8)
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    if m[y, x] == 0:
        xy = _nearest_foreground_xy(m, x, y, r=int(search_r))
        if xy is None:
            return np.zeros_like(m, dtype=np.uint8)
        x, y = xy
    flood = m.copy()
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff, (x, y), 128, flags=8)
    comp = (flood == 128).astype(np.uint8) * 255
    return comp


def white_component_at(
    mask_u8: np.ndarray,
    gray_u8: np.ndarray,
    x: int,
    y: int,
    *,
    percentile_low: float = 55.0,
    tolerance: float = 8.0,
) -> np.ndarray:
    """Connected component of background-like (white) pixels inside the mask at (x,y). Returns 0/255 mask."""
    low, high = _outline_background_gray_range(
        mask_u8, gray_u8, percentile_low=percentile_low, tolerance=tolerance
    )
    inside = (mask_u8 > 0).astype(np.uint8)
    gray_f = np.float64(gray_u8)
    bright = ((gray_f >= low) & (gray_f <= high) & (inside > 0)).astype(np.uint8) * 255
    return _connected_component_from_seed(bright, x, y, search_r=15)


def _convex_hull_mask(mask_u8: np.ndarray) -> np.ndarray:
    """Return filled convex hull for a 0/255 mask as 0/255."""
    m = (mask_u8 > 0).astype(np.uint8)
    if m.sum() == 0:
        return np.zeros_like(mask_u8, dtype=np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.zeros_like(mask_u8, dtype=np.uint8)
    cnt = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    out = np.zeros_like(mask_u8, dtype=np.uint8)
    cv2.fillPoly(out, [hull], 255)
    return out

def _apply_edit_layers(auto_u8: np.ndarray, add_u8: np.ndarray, del_u8: np.ndarray) -> np.ndarray:
    """Combine auto mask with persistent add/del layers. All are 0/255."""
    m = (auto_u8 > 0)
    if add_u8 is not None:
        m = m | (add_u8 > 0)
    if del_u8 is not None:
        m = m & ~(del_u8 > 0)
    return (m.astype(np.uint8) * 255)


def _outline_background_gray_range(
    mask_u8: np.ndarray,
    gray_u8: np.ndarray,
    *,
    dilate_r: int = 2,
    percentile_low: float = 55.0,
    tolerance: float = 8.0,
) -> tuple[float, float]:
    """Sample gray values from pixels just outside the mask (outline/contour). Return (low, high) range for background.

    Stricter defaults (p55, tol 8) to avoid eating tissue with white highlights.
    """
    assert mask_u8.shape[:2] == gray_u8.shape[:2]
    m = (mask_u8 > 0).astype(np.uint8)
    if m.sum() == 0:
        return 220.0, 255.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_r + 1, 2 * dilate_r + 1))
    dilated = cv2.dilate(m, k)
    outline_ring = (dilated > 0) & (m == 0)
    samples = np.float64(gray_u8[outline_ring])
    if samples.size < 10:
        return 220.0, 255.0
    ref = float(np.percentile(samples, percentile_low))
    low = max(0.0, ref - tolerance)
    return low, 255.0


def remove_voids_inside_mask(
    mask_u8: np.ndarray,
    gray_u8: np.ndarray,
    *,
    min_void_area: int = 500,
    dilate_r: int = 2,
    percentile_low: float = 55.0,
    tolerance: float = 8.0,
) -> np.ndarray:
    """Find background-colored regions inside the mask using the outline color and return a 0/255 mask of pixels to remove.

    Background reference is taken from pixels just outside the mask (the outline ring); 80%+ of those are white.
    Pixels inside the mask with gray in that range form candidate voids; connected components with area >= min_void_area
    are returned as 255 (to be subtracted from the brain mask).
    """
    assert mask_u8.shape[:2] == gray_u8.shape[:2]
    if (mask_u8 > 0).sum() == 0:
        return np.zeros_like(mask_u8, dtype=np.uint8)
    gray_float = np.float64(gray_u8)
    low, high = _outline_background_gray_range(
        mask_u8, gray_u8, dilate_r=dilate_r, percentile_low=percentile_low, tolerance=tolerance
    )
    inside = (mask_u8 > 0).astype(np.uint8)
    background_like = ((gray_float >= low) & (gray_float <= high) & (inside > 0)).astype(np.uint8) * 255
    num, lab, stats, _ = cv2.connectedComponentsWithStats(background_like, connectivity=8)
    out = np.zeros_like(mask_u8, dtype=np.uint8)
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area >= int(min_void_area):
            out[lab == idx] = 255
    return out


def fill_holes_max_size(mask_u8: np.ndarray, max_hole_area: int, *, binary: bool = True) -> np.ndarray:
    """Fill only holes (background components inside mask) with area <= max_hole_area.

    Larger holes are left as-is (not filled). Input/output 0/255; if binary=True output is 0/1.
    """
    m255 = (mask_u8 > 0).astype(np.uint8) * 255
    h, w = m255.shape
    inv = cv2.bitwise_not(m255)
    flood = inv.copy()
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff, seedPoint=(0, 0), newVal=0)
    exterior = (flood == 0) & (inv > 0)
    interior_holes = (inv > 0) & ~exterior
    interior_u8 = (interior_holes.astype(np.uint8)) * 255
    num, lab, stats, _ = cv2.connectedComponentsWithStats(interior_u8, connectivity=8)
    filled = m255.copy()
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area <= int(max_hole_area):
            filled[lab == idx] = 255
    if binary:
        return (filled > 0).astype(np.uint8)
    return filled
