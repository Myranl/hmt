from __future__ import annotations

import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

import customtkinter as ctk  # type: ignore[import-untyped]

from ui.common.theme import setup_theme, get_base_font, get_small_muted_font
from ui.common.widgets import (
    create_card_frame,
    create_primary_button,
    create_secondary_button,
    create_status_label,
)
from ui.brain.mask_morphology import _fill_holes, remove_voids_inside_mask


def fill_voids_ui(
    img_rgb: np.ndarray,
    mask_in: np.ndarray,
    *,
    window: str = "Fill voids inside mask",
    init_min_void_area: int = 500,
) -> tuple[np.ndarray, dict]:
    """Step 2 UI: operate only on voids / islands INSIDE an existing mask.

    - The outer contour of `mask_in` is preserved; operations only change interior pixels.
    - Users can auto-remove small voids and ALSO manually add/remove voids with a brush.
    """

    mask_base = (mask_in > 0).astype(np.uint8) * 255
    h, w = mask_base.shape[:2]

    # grayscale once for the whole UI
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Allowed interior region: fill holes once to know where "inside brain" is,
    # but keep mask_base itself unchanged so voids are still visible to the user.
    interior_allowed = _fill_holes(mask_base.copy())

    # Pre-compute distribution of interior void areas (connected components),
    # to suggest a meaningful slider range for "min void area".
    total_interior_px = int((interior_allowed > 0).sum())
    base_min_by_fraction = max(1, int(round(total_interior_px * 0.001)))  # 0.1% of interior pixels

    void_seed = ((interior_allowed > 0) & (mask_base == 0)).astype(np.uint8)
    num_voids, void_labels = cv2.connectedComponents(void_seed, connectivity=4)
    void_areas: list[int] = []
    for lbl in range(1, num_voids):
        area = int((void_labels == lbl).sum())
        if area > 0:
            void_areas.append(area)

    if void_areas:
        min_void_obs = min(void_areas)
        max_void_obs = max(void_areas)
        slider_min_void = max(base_min_by_fraction, min_void_obs)
        slider_max_void = max(slider_min_void + 1, max_void_obs)
        # Start somewhere near the lower end but not below computed min
        init_min_void_area = max(slider_min_void, min(init_min_void_area, slider_max_void))
    else:
        # Fallback when there are no visible voids
        slider_min_void = base_min_by_fraction
        slider_max_void = max(slider_min_void + 1, int(max(h, w) * 0.1) ** 2)

    # edit_add_u8: pixels user explicitly adds back to the mask (fill voids)
    # edit_del_u8: pixels user explicitly erases inside the mask (create voids)
    edit_add_u8 = np.zeros_like(mask_base, dtype=np.uint8)
    edit_del_u8 = np.zeros_like(mask_base, dtype=np.uint8)

    # simple undo stack for manual edits
    undo_stack: list[tuple[np.ndarray, np.ndarray]] = []

    def push_undo() -> None:
        undo_stack.append((edit_add_u8.copy(), edit_del_u8.copy()))
        if len(undo_stack) > 50:
            undo_stack.pop(0)

    def undo_last() -> None:
        if not undo_stack:
            return
        a, d = undo_stack.pop()
        edit_add_u8[:] = a
        edit_del_u8[:] = d

    state: dict = {
        "accepted": False,
        "dirty": True,
        "min_void_area": int(init_min_void_area),
        "auto_remove_voids": True,
        "m_u8": mask_base.copy(),
        "mode": "none",  # "erase_void" | "fill_void"
    }

    setup_theme()
    root = ctk.CTk()
    root.title(window)
    root.minsize(900, 580)
    root.geometry("1100x640")
    root.configure(fg_color="white")
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)

    # Left: image card
    img_card = create_card_frame(root)
    img_card.grid(row=0, column=0, sticky="nsew", padx=(18, 10), pady=18)
    img_card.columnconfigure(0, weight=1)
    img_card.rowconfigure(0, weight=1)
    canvas_holder = tk.Frame(img_card)
    canvas_holder.grid(row=0, column=0, sticky="nsew")
    canvas_holder.columnconfigure(0, weight=1)
    canvas_holder.rowconfigure(0, weight=1)
    scroll_y = ttk.Scrollbar(canvas_holder)
    scroll_x = ttk.Scrollbar(canvas_holder, orient=tk.HORIZONTAL)
    canvas = tk.Canvas(canvas_holder, highlightthickness=0, bg="#e8e8e8")
    canvas.grid(row=0, column=0, sticky="nsew")
    scroll_y.grid(row=0, column=1, sticky="ns")
    scroll_x.grid(row=1, column=0, sticky="ew")
    canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
    scroll_y.configure(command=canvas.yview)
    scroll_x.configure(command=canvas.xview)

    # Right: controls card (same style as folder selection / contour editor)
    ctrl = create_card_frame(root)
    ctrl.grid(row=0, column=1, sticky="ns", padx=(0, 18), pady=18)
    ctrl.grid_propagate(False)
    ctrl.configure(width=300)

    base_font = get_base_font()
    ctk.CTkLabel(ctrl, text="Fill voids", font=ctk.CTkFont(size=16, weight="bold")).grid(
        row=0, column=0, sticky="w", padx=14, pady=(14, 6))
    create_status_label(
        ctrl,
        text="Adjust filling of holes inside the mask. Outer contour is fixed and will not change here.",
        wraplength=260,
    ).grid(row=1, column=0, sticky="w", padx=14, pady=(0, 10))

    var_min_void = tk.IntVar(value=int(init_min_void_area))
    var_auto_remove = tk.BooleanVar(value=True)
    var_show_mask_only = tk.BooleanVar(value=False)

    # --- Automatic void removal (clear rows: label, then −10 / entry / +10, then checkbox) ---
    lf_auto = ctk.CTkFrame(ctrl, fg_color="transparent")
    lf_auto.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 10))
    lf_auto.columnconfigure(0, weight=1)
    ctk.CTkLabel(lf_auto, text="Automatic void removal", font=ctk.CTkFont(size=13, weight="bold")).grid(
        row=0, column=0, sticky="w", pady=(0, 6))

    ctk.CTkLabel(lf_auto, text="min void area (px)", font=base_font).grid(row=1, column=0, sticky="w", pady=(0, 4))

    def _step_min_void(delta: int) -> None:
        try:
            v = var_min_void.get()
            var_min_void.set(max(1, v + delta))
        except Exception:
            var_min_void.set(1)
        state["dirty"] = True
        _update_canvas()

    row_min = ctk.CTkFrame(lf_auto, fg_color="transparent")
    row_min.grid(row=2, column=0, sticky="ew", pady=(0, 8))
    row_min.columnconfigure(1, weight=1)
    ctk.CTkButton(row_min, text="−10", width=50, command=lambda: _step_min_void(-10)).grid(row=0, column=0, padx=(0, 6))
    entry_min_void = ttk.Entry(row_min, textvariable=var_min_void, width=10)
    entry_min_void.grid(row=0, column=1, sticky="ew", padx=4)
    ctk.CTkButton(row_min, text="+10", width=50, command=lambda: _step_min_void(10)).grid(row=0, column=2, padx=(6, 0))

    def _on_entry_commit(*_a) -> None:
        try:
            v = int(var_min_void.get())
            var_min_void.set(max(1, v))
        except Exception:
            var_min_void.set(1)
        state["dirty"] = True
        _update_canvas()

    entry_min_void.bind("<Return>", _on_entry_commit)
    entry_min_void.bind("<FocusOut>", _on_entry_commit)

    def _on_auto_remove() -> None:
        state["dirty"] = True
        _update_canvas()

    ctk.CTkCheckBox(
        lf_auto,
        text="Remove small voids automatically",
        variable=var_auto_remove,
        command=_on_auto_remove,
        font=base_font,
    ).grid(row=3, column=0, sticky="w", pady=(4, 0))

    mode_var = tk.StringVar(value="MODE: AUTO ONLY")

    def set_mode(name: str) -> None:
        state["mode"] = name
        pretty = {"erase_void": "ERASE VOID", "fill_void": "FILL VOID"}.get(name, "AUTO ONLY")
        mode_var.set(f"MODE: {pretty}")

    ctk.CTkLabel(ctrl, textvariable=mode_var, font=ctk.CTkFont(size=12, weight="bold")).grid(
        row=3, column=0, sticky="w", padx=14, pady=(8, 6))

    btn_brush = ctk.CTkFrame(ctrl, fg_color="transparent")
    btn_brush.grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 6))
    btn_brush.columnconfigure(0, weight=1)
    btn_brush.columnconfigure(1, weight=1)
    ctk.CTkButton(
        btn_brush,
        text="Erase void",
        command=lambda: (set_mode("erase_void"), state.__setitem__("dirty", True), _update_canvas()),
        width=100,
    ).grid(row=0, column=0, padx=(0, 6))
    ctk.CTkButton(
        btn_brush,
        text="Fill void",
        command=lambda: (set_mode("fill_void"), state.__setitem__("dirty", True), _update_canvas()),
        width=100,
    ).grid(row=0, column=1)

    def _do_undo() -> None:
        undo_last()
        state["dirty"] = True
        _update_canvas()

    def _do_clear() -> None:
        undo_stack.clear()
        edit_add_u8[:] = 0
        edit_del_u8[:] = 0
        state["dirty"] = True
        _update_canvas()

    btn_undo = ctk.CTkFrame(ctrl, fg_color="transparent")
    btn_undo.grid(row=5, column=0, sticky="ew", padx=14, pady=(0, 6))
    btn_undo.columnconfigure(0, weight=1)
    create_secondary_button(btn_undo, text="Undo", command=_do_undo).grid(row=0, column=0, sticky="w", padx=(0, 6))
    create_secondary_button(btn_undo, text="Clear edits", command=_do_clear).grid(row=0, column=1, sticky="e")

    def _on_show_mask_only() -> None:
        state["dirty"] = True
        _update_canvas()

    ctk.CTkCheckBox(
        ctrl,
        text="Mask only (B&W)",
        variable=var_show_mask_only,
        command=_on_show_mask_only,
        font=base_font,
    ).grid(row=6, column=0, sticky="w", padx=14, pady=(8, 0))

    def do_accept() -> None:
        state["accepted"] = True
        root.destroy()

    def do_skip() -> None:
        state["accepted"] = False
        root.destroy()

    bar = ctk.CTkFrame(ctrl, fg_color="transparent")
    bar.grid(row=7, column=0, sticky="ew", padx=14, pady=(12, 14))
    bar.columnconfigure(0, weight=1)
    create_secondary_button(bar, text="Skip", command=do_skip).grid(row=0, column=0, sticky="w")
    create_primary_button(bar, text="Accept", command=do_accept).grid(row=0, column=1, sticky="e")

    tk_img_ref: dict[str, ImageTk.PhotoImage | None] = {"img": None}
    canvas_img_id: list[int] = []

    def _get_canvas_size() -> tuple[int, int]:
        cw = canvas_holder.winfo_width() or 700
        ch = canvas_holder.winfo_height() or 500
        if cw < 200:
            cw = 700
        if ch < 200:
            ch = 500
        return (cw, ch)

    disp_scale = 1.0
    disp_w = w
    disp_h = h

    def _recompute_mask() -> None:
        m = mask_base.copy()

        # Apply automatic void removal inside the existing mask only
        if bool(var_auto_remove.get()):
            min_a = int(var_min_void.get())
            to_remove = remove_voids_inside_mask(m, gray, min_void_area=min_a)
            m = cv2.bitwise_and(m, cv2.bitwise_not(to_remove))

        # Apply manual edits but clamp them to the original interior region
        inside = interior_allowed > 0
        if np.any(edit_del_u8):
            m[np.logical_and(edit_del_u8 > 0, inside)] = 0
        if np.any(edit_add_u8):
            m[np.logical_and(edit_add_u8 > 0, inside)] = 255

        state["m_u8"] = m

    def _update_canvas() -> None:
        nonlocal disp_scale, disp_w, disp_h
        cw, ch = _get_canvas_size()
        disp_scale = min(1.0, cw / float(w), ch / float(h))
        disp_w = int(round(w * disp_scale))
        disp_h = int(round(h * disp_scale))
        canvas.configure(width=disp_w, height=disp_h)

        _recompute_mask()
        m = state["m_u8"]
        if bool(var_show_mask_only.get()):
            vis_bgr = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        else:
            alpha = 0.30
            vis_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            green = np.zeros_like(vis_bgr, dtype=np.uint8)
            green[:, :, 1] = 255
            m_bool = (m > 0)
            vis_bgr[m_bool] = cv2.addWeighted(vis_bgr[m_bool], 1.0 - alpha, green[m_bool], alpha, 0.0)
        if disp_scale < 1.0:
            vis_bgr = cv2.resize(vis_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(vis_rgb)
        tk_img = ImageTk.PhotoImage(pil, master=canvas)
        tk_img_ref["img"] = tk_img
        if not canvas_img_id:
            canvas_img_id.append(canvas.create_image(0, 0, anchor="nw", image=tk_img))
        else:
            canvas.itemconfigure(canvas_img_id[0], image=tk_img)
        canvas.configure(scrollregion=(0, 0, disp_w, disp_h))

    # --- Mouse tools for manual void edits (click = whole component) ---

    def _canvas_to_img(ev: tk.Event) -> tuple[int, int] | None:
        cx = canvas.canvasx(ev.x)
        cy = canvas.canvasy(ev.y)
        if cx < 0 or cy < 0 or cx >= disp_w or cy >= disp_h:
            return None
        ix = int(round(cx / disp_scale))
        iy = int(round(cy / disp_scale))
        return max(0, min(ix, w - 1)), max(0, min(iy, h - 1))

    def _apply_brush(ev: tk.Event) -> None:
        if state["mode"] not in ("erase_void", "fill_void"):
            return
        pos = _canvas_to_img(ev)
        if pos is None:
            return
        x, y = pos
        # operate only inside allowed interior (contour fixed, but voids included)
        if interior_allowed[y, x] == 0:
            return

        # Recompute current mask so we click on up-to-date void / filled region
        push_undo()
        _recompute_mask()
        m = state["m_u8"]

        # Choose connected component: either current void or bright filled area
        if state["mode"] == "fill_void":
            target_binary = ((m == 0) & (interior_allowed > 0)).astype(np.uint8)
        else:  # erase_void – only over bright (white-ish) regions
            # estimate bright threshold from interior background once
            bg_vals = gray[(interior_allowed > 0) & (mask_base == 0)]
            if bg_vals.size > 0:
                thr_bright = int(np.percentile(bg_vals, 70))
            else:
                thr_bright = 200
            target_binary = ((m > 0) & (interior_allowed > 0) & (gray >= thr_bright)).astype(np.uint8)

        if target_binary[y, x] == 0:
            return

        num, labels = cv2.connectedComponents(target_binary, connectivity=4)
        lbl = labels[y, x]
        if lbl == 0:
            return
        comp = labels == lbl

        if state["mode"] == "fill_void":
            edit_add_u8[comp] = 255
            edit_del_u8[comp] = 0
        else:  # erase_void
            edit_del_u8[comp] = 255
            edit_add_u8[comp] = 0

        state["dirty"] = True

    canvas.bind("<Button-1>", _apply_brush)
    # Для whole-component поведения достаточно реакции на одиночные клики.

    def _tick() -> None:
        if state["dirty"]:
            state["dirty"] = False
            _update_canvas()
        root.after(80, _tick)

    def _on_holder_configure(_ev: tk.Event) -> None:
        _update_canvas()

    canvas_holder.bind("<Configure>", _on_holder_configure)
    root.update_idletasks()
    _update_canvas()
    _tick()
    root.mainloop()

    return state["m_u8"], {
        "accepted": bool(state["accepted"]),
        "min_void_area": int(var_min_void.get()),
        "auto_remove_voids": bool(var_auto_remove.get()),
    }

