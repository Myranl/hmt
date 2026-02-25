from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


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
