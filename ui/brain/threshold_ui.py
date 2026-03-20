from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import cv2
import numpy as np

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

from ui.brain.mask_utils import _ensure_rgb_u8, _perimeter_px
from ui.brain.mask_compute import compute_mask
from ui.brain.mask_morphology import _largest_component

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


def brain_mask_auto(
    img_rgb: np.ndarray,
    *,
    pad: int = 50,
    seed_r: int = 40,
    close_r: int = 7,
    open_r: int = 3,
) -> BrainMaskUIResult:
    """Compute brain mask with automatic Otsu threshold, no UI.

    Same pipeline as brain_mask_threshold_ui but with fixed params (Otsu, pad_extra=0).
    Use this by default; open brain_mask_threshold_ui only in exceptional cases.
    """
    rgb = _ensure_rgb_u8(img_rgb)
    gray_u8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray_u8.shape[:2]

    try:
        otsu_thr, _ = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = int(np.clip(otsu_thr, 0, 255))
    except Exception:
        thr = 170

    pad_eff = int(max(0, pad))
    mask_u8 = compute_mask(
        gray_u8,
        thr=thr,
        pad_eff_ui=pad_eff,
        close_r_ui=int(close_r),
        open_r_ui=int(open_r),
        seed_r_ui=int(seed_r),
    )
    mask_bool = (mask_u8 > 0).astype(bool)

    params: dict[str, Any] = {
        "accepted": True,
        "thr_base": thr,
        "thr": thr,
        "pad": int(pad),
        "pad_extra": 0,
        "pad_effective": int(pad),
        "seed_r": int(seed_r),
        "close_r": int(close_r),
        "open_r": int(open_r),
        "area_px": int(mask_u8.sum()),
        "perim_px": float(_perimeter_px(mask_u8)),
    }
    return BrainMaskUIResult(mask=mask_bool, params=params)


def render(ctx: BrainMaskUIContext, thr_eff: int, thr_base: int, pad_extra: int) -> tuple[np.ndarray, np.ndarray]:
    pad_eff_ui = int(max(0, ctx.pad_ui + ctx._sc(pad_extra)))
    mask_u8 = compute_mask(ctx.gray_u8, thr=thr_eff, pad_eff_ui=pad_eff_ui, close_r_ui=ctx.close_r_ui, open_r_ui=ctx.open_r_ui, seed_r_ui=ctx.seed_r_ui,)
    disp = ctx.rgb.copy()  # RGB

    # White fill OUTSIDE the current mask so the brain pops out visually.
    m = (mask_u8 > 0).astype(np.uint8)
    outside = m == 0
    if np.any(outside):
        disp[outside] = (255, 255, 255)

    # Draw boundary with high contrast (black underlay + magenta line)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        # OpenCV expects BGR, so convert for drawing then back
        bgr = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
        # underlay
        cv2.drawContours(bgr, cnts, -1, (0, 0, 0), 5)
        # main line (magenta)
        cv2.drawContours(bgr, cnts, -1, (255, 0, 255), 2)
        disp = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

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
        # extra UI state for manual splitting when two brains are too close
        "mode": "cut_line",           # "cut_line" on by default; clicks = draw break line
        "cut_line_pts": [],           # [(x, y)] for first click
    }

    # --- Tkinter UI window (cross-platform, consistent with other UIs) ---
    root = tk.Tk()
    root.title(window)

    # Layout: image on the left, controls on the right
    frm = ttk.Frame(root, padding=10)
    frm.grid(row=0, column=0, sticky="nsew")
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    frm.rowconfigure(0, weight=1)
    frm.columnconfigure(0, weight=1)

    canvas = tk.Canvas(frm, highlightthickness=0, bg="#111")
    canvas.grid(row=0, column=0, sticky="nsew")

    ctrl = ttk.Frame(frm)
    ctrl.grid(row=0, column=1, padx=(12, 0), sticky="ns")

    # Vars for sliders and zoom
    var_thr = tk.IntVar(value=int(thr0))
    var_pad_extra = tk.IntVar(value=0)
    var_zoom = tk.IntVar(value=100)

    def mark_dirty() -> None:
        state["need_redraw"] = True

    def _on_thr_changed(*_a) -> None:
        try:
            state["thr"] = int(var_thr.get())
        except Exception:
            pass
        mark_dirty()

    def _on_pad_changed(*_a) -> None:
        try:
            state["pad_extra"] = int(var_pad_extra.get())
        except Exception:
            pass
        mark_dirty()

    var_thr.trace_add("write", _on_thr_changed)
    var_pad_extra.trace_add("write", _on_pad_changed)

    # Controls: labels + sliders + buttons
    ttk.Label(ctrl, text="Brain mask threshold", font=("TkDefaultFont", 13, "bold")).grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(0, 6)
    )
    ttk.Label(
        ctrl,
        text="Adjust threshold and padding.\nEnter: accept   Esc: cancel   R: reset",
        justify="left",
    ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 10))

    def _add_slider(row: int, text: str, var: tk.IntVar, frm_to: int, frm_from: int = 0) -> None:
        ttk.Label(ctrl, text=text).grid(row=row, column=0, sticky="w")
        s = ttk.Scale(ctrl, from_=frm_from, to=frm_to, orient="horizontal")
        s.set(float(np.clip(var.get(), frm_from, frm_to)))

        def _on_scale(val: str) -> None:
            try:
                var.set(int(float(val) + 0.5))
            except Exception:
                pass

        s.configure(command=_on_scale)
        s.grid(row=row, column=1, sticky="ew", pady=2)
        ctrl.columnconfigure(1, weight=1)

    # Threshold slider: min = Otsu −20% (so we can go a bit left), max = 255; floor at 50
    thr_slider_min = max(50, int(round(thr0 * 0.80)))
    thr_slider_max = 255
    var_thr.set(int(np.clip(var_thr.get(), thr_slider_min, thr_slider_max)))
    state["thr"] = int(var_thr.get())
    THR_SLIDER_MIN, THR_SLIDER_MAX = thr_slider_min, thr_slider_max
    _add_slider(2, "thr (dark<)", var_thr, THR_SLIDER_MAX, frm_from=THR_SLIDER_MIN)
    _add_slider(3, "pad +", var_pad_extra, 200)

    # Metrics label
    metrics_var = tk.StringVar(value="")
    ttk.Label(ctrl, textvariable=metrics_var, justify="left").grid(
        row=4, column=0, columnspan=2, sticky="w", pady=(8, 10)
    )

    # Buttons
    btns = ttk.Frame(ctrl)
    btns.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(4, 0))
    btns.columnconfigure(0, weight=1)
    btns.columnconfigure(1, weight=1)

    # To hold latest mask from render()
    mask_holder: dict[str, np.ndarray] = {"mask": np.zeros_like(gray_u8, dtype=np.uint8)}

    def do_accept() -> None:
        state["accepted"] = True
        for aid in _tick_id:
            try:
                root.after_cancel(aid)
            except Exception:
                pass
        root.destroy()

    def do_cancel() -> None:
        state["accepted"] = False
        for aid in _tick_id:
            try:
                root.after_cancel(aid)
            except Exception:
                pass
        root.destroy()

    def do_reset() -> None:
        var_thr.set(int(np.clip(thr0, THR_SLIDER_MIN, THR_SLIDER_MAX)))
        var_pad_extra.set(0)
        state["thr"] = int(var_thr.get())
        state["pad_extra"] = 0
        mark_dirty()

    def do_undo() -> None:
        nonlocal cut_mask_u8
        if not undo_stack:
            return
        cut_mask_u8 = undo_stack.pop()
        mark_dirty()

    ttk.Button(btns, text="Accept (Enter)", command=do_accept).grid(row=0, column=0, sticky="ew", padx=(0, 6))
    ttk.Button(btns, text="Cancel (Esc)", command=do_cancel).grid(row=0, column=1, sticky="ew")
    ttk.Button(btns, text="Undo", command=do_undo).grid(row=1, column=0, sticky="ew", padx=(0, 6), pady=(6, 0))
    ttk.Button(btns, text="Reset (R)", command=do_reset).grid(row=1, column=1, sticky="ew", pady=(6, 0))

    def _on_wheel(ev) -> None:
        if hasattr(ev, "delta"):
            delta = ev.delta  # Windows / macOS
        else:
            delta = 120 if getattr(ev, "num", 5) == 4 else -120  # Linux Button-4/5
        cur = var_zoom.get()
        var_zoom.set(max(50, min(300, cur + (10 if delta > 0 else -10))))
        mark_dirty()

    canvas.bind("<MouseWheel>", _on_wheel)
    canvas.bind("<Button-4>", _on_wheel)
    canvas.bind("<Button-5>", _on_wheel)

    # Canvas image handling
    tk_img_ref: dict[str, ImageTk.PhotoImage | None] = {"img": None}
    canvas_img_id: list[int] = []

    # Persistent "cut" mask in UI resolution (same HxW as gray_u8).
    # We keep a strip of 255 values where the user draws cut lines and
    # subtract it from the auto mask before upscaling to full resolution.
    cut_mask_u8 = np.zeros_like(gray_u8, dtype=np.uint8)
    undo_stack: list[np.ndarray] = []  # previous cut_mask_u8 states for Undo
    MAX_UNDO = 50

    def _update_canvas() -> None:
        """Recompute mask for current threshold, apply cut-mask, and redraw."""
        nonlocal cut_mask_u8

        thr_eff = int(np.clip(state["thr"], 0, 255))
        # Base rendering (auto mask, gray-out outside, contour)
        base_disp_rgb, mask_u8 = render(ctx, thr_eff, state["thr"], state["pad_extra"])

        # Apply accumulated cut lines (if any) as "holes" in the mask.
        if cut_mask_u8 is not None and cut_mask_u8.shape == mask_u8.shape and int(cut_mask_u8.sum()) > 0:
            mask_u8 = cv2.bitwise_and(mask_u8, cv2.bitwise_not(cut_mask_u8))

        # Keep only the largest connected component; drop small blobs and the "other" brain after a cut.
        mask_u8 = _largest_component(mask_u8)

        # Rebuild visualization from the (possibly cut) mask so the user
        # always sees the effective brain outline.
        disp_rgb = ctx.rgb.copy()
        m = (mask_u8 > 0).astype(np.uint8)
        outside = m == 0
        if np.any(outside):
            disp_rgb[outside] = (255, 255, 255)

        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            bgr = cv2.cvtColor(disp_rgb, cv2.COLOR_RGB2BGR)
            cv2.drawContours(bgr, cnts, -1, (0, 0, 0), 5)
            cv2.drawContours(bgr, cnts, -1, (255, 0, 255), 2)
            disp_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Show first cut point (if any) as a small red circle.
        cpts = state.get("cut_line_pts") or []
        if len(cpts) == 1:
            px, py = cpts[0]
            cv2.circle(disp_rgb, (int(px), int(py)), 6, (255, 0, 0), 2)

        # Save effective mask (after cuts) for upscaling at the end.
        mask_holder["mask"] = mask_u8.astype(np.uint8)

        # metrics from the effective mask
        area = int(mask_u8.sum())
        perim = float(_perimeter_px(mask_u8))
        metrics_var.set(
            f"thr={state['thr']} → {thr_eff}\n"
            f"area={area} px   perim={perim:.1f} px\n"
            f"pad={pad}px (+{state['pad_extra']} px)   scale={scale:.3f}"
        )

        # Zoom: scale display image by var_zoom (50–300%)
        h, w = disp_rgb.shape[:2]
        zoom_pct = max(50, min(300, int(var_zoom.get())))
        scale_zoom = zoom_pct / 100.0
        state["_zoom_scale"] = scale_zoom
        disp_w = max(1, int(round(w * scale_zoom)))
        disp_h = max(1, int(round(h * scale_zoom)))
        disp_zoomed = cv2.resize(disp_rgb, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

        pil = Image.fromarray(disp_zoomed)
        tk_img = ImageTk.PhotoImage(pil, master=canvas)
        tk_img_ref["img"] = tk_img

        canvas.configure(width=min(disp_w, 900), height=min(disp_h, 700), scrollregion=(0, 0, disp_w, disp_h))
        if not canvas_img_id:
            canvas_img_id.append(canvas.create_image(0, 0, anchor="nw", image=tk_img))
        else:
            canvas.itemconfigure(canvas_img_id[0], image=tk_img)

    _tick_id: list = []

    def _canvas_to_xy(ev) -> tuple[int, int] | None:
        """Map canvas click coordinates to mask/display coordinates."""
        if not canvas_img_id:
            return None
        cx = canvas.canvasx(ev.x)
        cy = canvas.canvasy(ev.y)
        zoom_scale = state.get("_zoom_scale", 1.0)
        # Convert display coords to mask coords
        ix = int(round(cx / zoom_scale))
        iy = int(round(cy / zoom_scale))
        h_m, w_m = mask_holder["mask"].shape[:2]
        if ix < 0 or iy < 0 or ix >= w_m or iy >= h_m:
            return None
        return ix, iy

    def _on_canvas_click(ev) -> None:
        """Handle manual cut-line clicks when in cut_line mode."""
        nonlocal cut_mask_u8
        if state.get("mode") != "cut_line":
            return
        xy = _canvas_to_xy(ev)
        if xy is None:
            return
        x, y = xy
        pts = state.get("cut_line_pts") or []
        if len(pts) == 0:
            # First click – remember start point and show marker.
            state["cut_line_pts"] = [(x, y)]
            mark_dirty()
            return

        # Second click – draw a strip between the two points and add it to cut_mask_u8.
        x1, y1 = pts[0]
        thickness = 9  # match Brain outline cut-line visual width
        if cut_mask_u8.shape != mask_holder["mask"].shape:
            cut_mask_u8 = np.zeros_like(mask_holder["mask"], dtype=np.uint8)
        # Push current cut mask for Undo before applying new line
        undo_stack.append(cut_mask_u8.copy())
        if len(undo_stack) > MAX_UNDO:
            undo_stack.pop(0)
        line_mask = np.zeros_like(cut_mask_u8, dtype=np.uint8)
        cv2.line(line_mask, (int(x1), int(y1)), (int(x), int(y)), 255, thickness=thickness)
        cut_mask_u8 = cv2.bitwise_or(cut_mask_u8, line_mask)
        state["cut_line_pts"] = []
        mark_dirty()

    def _tick() -> None:
        if state["need_redraw"]:
            state["need_redraw"] = False
            _update_canvas()
        _tick_id.clear()
        _tick_id.append(root.after(40, _tick))

    # Key bindings
    def _on_key(ev) -> None:
        ks = (ev.keysym or "").lower()
        if ks in ("return", "kp_enter"):
            do_accept()
        elif ks == "escape":
            do_cancel()
        elif ks == "r":
            do_reset()

    root.bind("<Key>", _on_key)
    canvas.bind("<Button-1>", _on_canvas_click)

    def _on_destroy(_ev=None) -> None:
        for aid in _tick_id:
            try:
                root.after_cancel(aid)
            except Exception:
                pass
        _tick_id.clear()

    root.protocol("WM_DELETE_WINDOW", do_cancel)
    root.bind("<Destroy>", _on_destroy)

    # Force layout then draw so the canvas has size and the image is visible at once
    root.update_idletasks()
    state["need_redraw"] = False
    _update_canvas()
    _tick()
    root.mainloop()

    if not state["accepted"]:
        return None

    # Upscale mask back to full resolution (if UI was downsampled)
    mask_u8 = mask_holder["mask"]
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