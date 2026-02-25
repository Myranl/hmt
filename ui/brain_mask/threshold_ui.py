from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import cv2
import numpy as np

from ui.brain_mask.mask_utils import _ensure_rgb_u8, _perimeter_px, _safe_display_overlay
from ui.brain_mask.mask_compute import compute_mask

@dataclass
class BrainMaskUIContext:
    gray_u8: np.ndarray
    rgb: np.ndarray
    window: str
    scale: float
    pad: int
    pad_ui: int
    seed_r_ui: int
    close_r_ui: int
    open_r_ui: int

    def _sc(self, v: int) -> int:
        return int(max(1, round(int(v) * self.scale))) if v > 0 else 0

@dataclass
class BrainMaskUIResult:
    mask: np.ndarray  # bool, same HxW as input gray/img2
    params: dict[str, Any]


def render(ctx: BrainMaskUIContext, thr_eff: int, thr_base: int, pad_extra: int) -> tuple[np.ndarray, np.ndarray]:
    pad_eff_ui = int(max(0, ctx.pad_ui + ctx._sc(pad_extra)))
    mask_u8 = compute_mask(ctx.gray_u8, thr=thr_eff, pad_eff_ui=pad_eff_ui, close_r_ui=ctx.close_r_ui, open_r_ui=ctx.open_r_ui, seed_r_ui=ctx.seed_r_ui,)
    disp = ctx.rgb.copy()  # RGB

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

    # HUD text overlay (best-effort window overlay)
    area = int(mask_u8.sum())
    perim = _perimeter_px(mask_u8)
    msg = (
        f"thr={thr_base} => {thr_eff} | area={area} px | perim={perim:.1f} px | "
        f"pad={ctx.pad}px (+{pad_extra}px) | scale={ctx.scale:.3f} | ENTER accept | ESC cancel | R reset"
    )
    _safe_display_overlay(ctx.window, msg, 1000)

    return disp, mask_u8


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

    pad_ui = int(max(1, round(int(pad) * scale))) if pad > 0 else 0
    seed_r_ui = int(max(1, round(int(seed_r) * scale))) if seed_r > 0 else 0
    close_r_ui = int(max(1, round(int(close_r) * scale))) if close_r > 0 else 0
    open_r_ui = int(max(1, round(int(open_r) * scale))) if open_r > 0 else 0

    ctx = BrainMaskUIContext(
        gray_u8=gray_u8,
        rgb=rgb,
        window=window,
        scale=scale,
        pad=pad,
        pad_ui=pad_ui,
        seed_r_ui=seed_r_ui,
        close_r_ui=close_r_ui,
        open_r_ui=open_r_ui,
    )

    # initial threshold via Otsu (on inverted? no, we want dark as FG => gray < thr)
    try:
        otsu_thr, _ = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr0 = int(otsu_thr)
    except Exception:
        thr0 = 170

    thr0 = int(np.clip(thr0, 0, 255))

    state = {
        "thr": thr0,
        "pad_extra": 0,
        "need_redraw": True,
        "accepted": False,
    }

    # UI window
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def _on_trackbar(v: int) -> None:
        state["thr"] = int(v)
        state["need_redraw"] = True

    def _on_trackbar_pad(v: int) -> None:
        state["pad_extra"] = int(v)
        state["need_redraw"] = True

    cv2.createTrackbar("thr (dark<)", window, thr0, 255, _on_trackbar)
    cv2.createTrackbar("pad +", window, 0, 200, _on_trackbar_pad)

    thr_eff = int(np.clip(state["thr"], 0, 255))

    disp_rgb, mask_u8 = render(ctx, thr_eff, state["thr"], state["pad_extra"])

    while True:
        if state["need_redraw"]:
            thr_eff = int(np.clip(state["thr"], 0, 255))
            disp_rgb, mask_u8 = render(ctx, thr_eff, state["thr"], state["pad_extra"])
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
            cv2.setTrackbarPos("pad +", window, 0)
            state["thr"] = thr0
            state["pad_extra"] = 0
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
        "thr_base": int(state["thr"]),
        "thr": int(np.clip(state["thr"], 0, 255)),
        "pad": int(pad),
        "pad_extra": int(state["pad_extra"]),
        "pad_effective": int(pad + int(state["pad_extra"])),
        "seed_r": int(seed_r),
        "close_r": int(close_r),
        "open_r": int(open_r),
        "max_side": int(max_side),
        "scale": float(scale),
        "area_px": int(mask_u8_full.sum()),
        "perim_px": float(_perimeter_px(mask_u8_full)),
    }

    return BrainMaskUIResult(mask=mask_bool, params=params)