from __future__ import annotations

import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

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

    root = tk.Tk()
    root.title(window)

    frm = ttk.Frame(root, padding=8)
    frm.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    frm.columnconfigure(0, weight=1)
    frm.rowconfigure(0, weight=1)

    canvas = tk.Canvas(frm, highlightthickness=0, bg="#111")
    canvas.grid(row=0, column=0, sticky="nsew")

    ctrl = ttk.Frame(frm, width=260)
    ctrl.grid(row=0, column=1, sticky="nsw", padx=(8, 0))
    ctrl.grid_propagate(False)

    ttk.Label(ctrl, text="Step 2: fill voids", font=("TkDefaultFont", 13, "bold")).grid(
        row=0, column=0, sticky="w", pady=(0, 4)
    )
    ttk.Label(
        ctrl,
        text="Adjust filling of holes inside the mask.\nOuter contour is fixed and will not change here.",
        justify="left",
    ).grid(row=1, column=0, sticky="w", pady=(0, 6))

    var_min_void = tk.IntVar(value=int(init_min_void_area))
    var_auto_remove = tk.BooleanVar(value=True)
    var_show_mask_only = tk.BooleanVar(value=False)

    lf_auto = ttk.LabelFrame(ctrl, text="Automatic void removal", padding=6)
    lf_auto.grid(row=2, column=0, sticky="ew", pady=(0, 6))
    lf_auto.columnconfigure(1, weight=1)
    ttk.Label(lf_auto, text="min void area (px)").grid(row=0, column=0, sticky="w")
    s_min = ttk.Scale(lf_auto, from_=float(slider_min_void), to=float(slider_max_void), orient="horizontal")
    s_min.set(float(var_min_void.get()))

    def _on_min_scale(val: str) -> None:
        try:
            var_min_void.set(max(1, int(float(val) + 0.5)))
        except Exception:
            pass
        state["dirty"] = True
        _update_canvas()

    s_min.configure(command=_on_min_scale)
    s_min.grid(row=0, column=1, sticky="ew", pady=2)

    def _on_min_var(*_a) -> None:
        try:
            s_min.set(float(var_min_void.get()))
        except Exception:
            pass
        state["dirty"] = True
        _update_canvas()

    var_min_void.trace_add("write", _on_min_var)

    def _on_auto_remove() -> None:
        state["dirty"] = True
        _update_canvas()

    ttk.Checkbutton(
        lf_auto,
        text="Remove small voids automatically",
        variable=var_auto_remove,
        command=_on_auto_remove,
    ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

    mode_var = tk.StringVar(value="MODE: AUTO ONLY")

    def set_mode(name: str) -> None:
        state["mode"] = name
        pretty = {"erase_void": "ERASE VOID", "fill_void": "FILL VOID"}.get(name, "AUTO ONLY")
        mode_var.set(f"MODE: {pretty}")

    ttk.Label(ctrl, textvariable=mode_var, font=("TkDefaultFont", 11, "bold")).grid(
        row=3, column=0, sticky="w", pady=(4, 4)
    )

    btns = ttk.Frame(ctrl)
    btns.grid(row=4, column=0, sticky="ew")
    btns.columnconfigure(0, weight=1)
    btns.columnconfigure(1, weight=1)

    def do_accept() -> None:
        state["accepted"] = True
        root.destroy()

    def do_skip() -> None:
        state["accepted"] = False
        root.destroy()

    ttk.Button(btns, text="Accept", command=do_accept).grid(row=0, column=0, sticky="ew", padx=(0, 4))
    ttk.Button(btns, text="Skip", command=do_skip).grid(row=0, column=1, sticky="ew")

    # Manual brush modes
    btn_brush = ttk.Frame(ctrl)
    btn_brush.grid(row=5, column=0, sticky="ew", pady=(6, 0))
    btn_brush.columnconfigure(0, weight=1)
    btn_brush.columnconfigure(1, weight=1)
    ttk.Button(
        btn_brush,
        text="Erase void",
        command=lambda: (set_mode("erase_void"), state.__setitem__("dirty", True), _update_canvas()),
    ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
    ttk.Button(
        btn_brush,
        text="Fill void",
        command=lambda: (set_mode("fill_void"), state.__setitem__("dirty", True), _update_canvas()),
    ).grid(row=0, column=1, sticky="ew")

    # Undo / clear for manual edits
    btn_undo = ttk.Frame(ctrl)
    btn_undo.grid(row=7, column=0, sticky="ew", pady=(6, 0))
    btn_undo.columnconfigure(0, weight=1)
    btn_undo.columnconfigure(1, weight=1)

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

    ttk.Button(btn_undo, text="Undo", command=_do_undo).grid(row=0, column=0, sticky="ew", padx=(0, 4))
    ttk.Button(btn_undo, text="Clear edits", command=_do_clear).grid(row=0, column=1, sticky="ew")

    # View options
    lf_view = ttk.LabelFrame(ctrl, text="View", padding=4)
    lf_view.grid(row=6, column=0, sticky="ew", pady=(6, 0))
    def _on_show_mask_only() -> None:
        state["dirty"] = True
        _update_canvas()

    ttk.Checkbutton(
        lf_view,
        text="Mask only (B&W)",
        variable=var_show_mask_only,
        command=_on_show_mask_only,
    ).grid(row=0, column=0, sticky="w")

    tk_img_ref: dict[str, ImageTk.PhotoImage | None] = {"img": None}
    canvas_img_id: list[int] = []

    # Fit image into a reasonable window size (similar to outline UI)
    try:
        sw = int(root.winfo_screenwidth())
        sh = int(root.winfo_screenheight())
    except Exception:
        sw, sh = 1400, 900
    max_canvas_w = max(400, min(int(sw * 0.68), int(w)))
    max_canvas_h = max(300, min(int(sh * 0.80), int(h)))
    disp_scale = min(1.0, max_canvas_w / float(w), max_canvas_h / float(h))
    disp_w = int(round(w * disp_scale))
    disp_h = int(round(h * disp_scale))
    canvas.configure(width=max_canvas_w, height=max_canvas_h)

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

    _update_canvas()
    _tick()
    root.mainloop()

    return state["m_u8"], {
        "accepted": bool(state["accepted"]),
        "min_void_area": int(var_min_void.get()),
        "auto_remove_voids": bool(var_auto_remove.get()),
    }

