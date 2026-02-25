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
