import numpy as np
import cv2
from ui.brain_mask.mask_utils import _gray_to_u8, _odd, _put_text_box
from ui.brain_mask.mask_morphology import (
    _fill_holes,
    _largest_component,
    _convex_hull_mask,
    _apply_edit_layers,
    _connected_component_from_seed,
    remove_voids_inside_mask,
    white_component_at,
)
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk


# --- Helper functions ---


def brain_outline_ui(
    img_rgb: np.ndarray,
    *,
    init_mask: np.ndarray | None = None,
    window: str = "Brain outline",
    init_thr: int = 170,
    init_smooth: int = 15,
    init_close: int = 11,
    init_open: int = 5,
    min_area: int = 20000,
    downsample_max_side: int = 1200,
    crop_pad: int = 20,
) -> tuple[np.ndarray, dict]:
    """Tk UI to tune threshold + contour smoothing + quick manual mask edits.

    If init_mask is provided (mask from previous step), the image is cropped to its bbox
    and shown with minimal reduction (scale only to fit window). Otherwise same as before.

    Returns:
      (mask_bool_fullres, params_dict) — mask is in same shape as img_rgb.
    """

    img0 = img_rgb
    h0, w0 = img0.shape[:2]
    crop_bbox: tuple[int, int, int, int] | None = None  # (y0, x0, y1, x1) in full image

    scale = 1.0
    if init_mask is not None and init_mask.shape[:2] == (h0, w0) and np.any(init_mask):
        ys, xs = np.where(init_mask)
        if ys.size > 0 and xs.size > 0:
            y0 = max(0, int(ys.min()) - crop_pad)
            y1 = min(h0, int(ys.max()) + 1 + crop_pad)
            x0 = max(0, int(xs.min()) - crop_pad)
            x1 = min(w0, int(xs.max()) + 1 + crop_pad)
            crop_bbox = (y0, x0, y1, x1)
            img = img0[y0:y1, x0:x1].copy()
            # no pre-downsample: minimal reduction (only fit to canvas later)
        else:
            img = img0
    else:
        # No crop: optional downsample for large images
        scale = min(1.0, downsample_max_side / float(max(h0, w0)))
        if scale < 1.0:
            new_w = int(round(w0 * scale))
            new_h = int(round(h0 * scale))
            img = cv2.resize(img0, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img = img0

    h, w = img.shape[:2]

    # grayscale on the UI image
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # manual edit layers (UI scale)
    edit_add_u8 = np.zeros((h, w), dtype=np.uint8)
    edit_del_u8 = np.zeros((h, w), dtype=np.uint8)

    # undo stack stores (add_layer, del_layer)
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

    # state
    state = {
        "m_u8": np.zeros((h, w), dtype=np.uint8),
        "mode": "erase_protrusion",
        "show_mask": True,
        "show_protrusions": True,
        "accepted": False,
        "cancelled": False,
        "dirty": True,
        "last_thr": None,
        "last_close": None,
        "last_open": None,
        "last_smooth": None,
        "effective_scale": 1.0,
        "disp_w_zoomed": 0,
        "disp_h_zoomed": 0,
        "_cache_auto": None,
        "_cache_auto_key": None,
        "protrusions_u8": None,
        "_cache_m_voids_key": None,
        "_cache_m_after_voids": None,
        "_cache_protrusions_key": None,
    }

    # ------------------------
    # Tk window + layout
    # ------------------------
    root = tk.Tk()
    root.title(window)

    # Make the UI reasonably sized on small displays
    try:
        sw = int(root.winfo_screenwidth())
        sh = int(root.winfo_screenheight())
    except Exception:
        sw, sh = 1400, 900

    # Main frame
    frm = ttk.Frame(root, padding=10)
    frm.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    frm.columnconfigure(0, weight=1)
    frm.rowconfigure(0, weight=1)

    # Canvas (image) with scrollbars for zoom
    frm_canvas = ttk.Frame(frm)
    frm_canvas.grid(row=0, column=0, sticky="nsew")
    scroll_y = ttk.Scrollbar(frm_canvas)
    scroll_x = ttk.Scrollbar(frm_canvas, orient=tk.HORIZONTAL)
    canvas = tk.Canvas(frm_canvas, highlightthickness=0, bg="#111")
    canvas.grid(row=0, column=0, sticky="nsew")
    scroll_y.grid(row=0, column=1, sticky="ns")
    scroll_x.grid(row=1, column=0, sticky="ew")
    canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
    scroll_y.configure(command=canvas.yview)
    scroll_x.configure(command=canvas.xview)
    frm_canvas.columnconfigure(0, weight=1)
    frm_canvas.rowconfigure(0, weight=1)

    # Right controls
    ctrl = ttk.Frame(frm)
    ctrl.grid(row=0, column=1, padx=(12, 0), sticky="ns")

    # Vars
    var_thr = tk.IntVar(value=int(init_thr))
    var_smooth = tk.IntVar(value=int(init_smooth))
    var_close = tk.IntVar(value=int(init_close))
    var_open = tk.IntVar(value=int(init_open))
    var_edit_open = tk.IntVar(value=41)
    var_min_void_area = tk.IntVar(value=500)

    var_show_mask = tk.BooleanVar(value=True)
    var_show_mask_only = tk.BooleanVar(value=False)
    var_remove_voids = tk.BooleanVar(value=True)
    var_zoom = tk.IntVar(value=100)

    # Labels / instructions
    lbl_title = ttk.Label(ctrl, text="Brain outline", font=("TkDefaultFont", 13, "bold"))
    lbl_title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

    lbl_hint = ttk.Label(
        ctrl,
        text=(
            "Click: E=erase protrusion, W=erase white blob, A=add indent.\n"
            "U undo, C clear, M mask. Enter accept, Esc cancel"
        ),
        justify="left",
    )
    lbl_hint.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 10))

    # Sliders
    def _add_slider(row: int, text: str, var: tk.IntVar, frm_to: int) -> None:
        ttk.Label(ctrl, text=text).grid(row=row, column=0, sticky="w")
        s = ttk.Scale(ctrl, from_=0, to=frm_to, orient="horizontal", command=lambda _v: mark_dirty())
        # ttk.Scale is float; sync via set/get
        s.set(float(var.get()))
        def _on_var(*_a):
            try:
                s.set(float(var.get()))
            except Exception:
                pass
            mark_dirty()
        var.trace_add("write", _on_var)

        def _on_scale(val: str):
            try:
                var.set(int(float(val) + 0.5))
            except Exception:
                pass
        s.configure(command=_on_scale)
        s.grid(row=row, column=1, sticky="ew", pady=2)
        ctrl.columnconfigure(1, weight=1)

    def mark_dirty() -> None:
        state["dirty"] = True

    _add_slider(2, "thr", var_thr, 255)
    _add_slider(3, "smooth", var_smooth, 101)
    _add_slider(4, "close", var_close, 101)
    _add_slider(5, "open", var_open, 101)
    _add_slider(6, "edit_open (protrusions)", var_edit_open, 101)
    _add_slider(7, "min void area (px)", var_min_void_area, 5000)
    _add_slider(8, "zoom %", var_zoom, 200)

    # Toggles
    def _toggle_mask() -> None:
        state["show_mask"] = bool(var_show_mask.get())
        mark_dirty()

    chk_mask = ttk.Checkbutton(ctrl, text="show mask (M)", variable=var_show_mask, command=_toggle_mask)
    chk_mask.grid(row=10, column=0, columnspan=2, sticky="w", pady=(10, 0))
    chk_voids = ttk.Checkbutton(
        ctrl, text="Remove voids (recomputed when sliders change)", variable=var_remove_voids, command=mark_dirty
    )
    chk_voids.grid(row=11, column=0, columnspan=2, sticky="w")
    chk_mask_only = ttk.Checkbutton(
        ctrl, text="Mask only (B&W)", variable=var_show_mask_only, command=mark_dirty
    )
    chk_mask_only.grid(row=12, column=0, columnspan=2, sticky="w")

    # Mode indicator (short, no param dump to avoid panel reflow)
    mode_var = tk.StringVar(value="MODE: ERASE")
    lbl_mode = ttk.Label(ctrl, textvariable=mode_var, font=("TkDefaultFont", 11, "bold"))
    lbl_mode.grid(row=13, column=0, columnspan=2, sticky="w", pady=(12, 6))

    # Buttons
    btns = ttk.Frame(ctrl)
    btns.grid(row=14, column=0, columnspan=2, sticky="ew")

    def do_accept() -> None:
        state["accepted"] = True
        root.destroy()

    def do_cancel() -> None:
        state["cancelled"] = True
        root.destroy()

    def do_clear() -> None:
        push_undo()
        edit_add_u8[:] = 0
        edit_del_u8[:] = 0
        mark_dirty()

    def set_mode(kind: str) -> None:
        state["mode"] = kind
        mode_var.set(f"MODE: {kind.upper().replace('_', ' ')}")

    ttk.Button(btns, text="Accept (Enter)", command=do_accept).grid(row=0, column=0, sticky="ew", padx=(0, 6))
    ttk.Button(btns, text="Cancel (Esc)", command=do_cancel).grid(row=0, column=1, sticky="ew")
    ttk.Button(btns, text="Undo (U)", command=lambda: (undo_last(), mark_dirty())).grid(row=1, column=0, sticky="ew", pady=(6, 0), padx=(0, 6))
    ttk.Button(btns, text="Clear edits (C)", command=do_clear).grid(row=1, column=1, sticky="ew", pady=(6, 0))
    ttk.Button(btns, text="Erase protrusion (E)", command=lambda: set_mode("erase_protrusion")).grid(row=2, column=0, sticky="ew", pady=(6, 0), padx=(0, 6))
    ttk.Button(btns, text="Erase white (W)", command=lambda: set_mode("erase_white")).grid(row=2, column=1, sticky="ew", pady=(6, 0))
    ttk.Button(btns, text="Add indent (A)", command=lambda: set_mode("add_indent")).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
    btns.columnconfigure(0, weight=1)
    btns.columnconfigure(1, weight=1)

    # ------------------------
    # Display scaling (fit to screen)
    # ------------------------
    # Fit the UI image into the available canvas area (rough estimate based on screen)
    max_canvas_w = max(400, min(int(sw * 0.68), int(w)))
    max_canvas_h = max(300, min(int(sh * 0.80), int(h)))

    disp_scale = min(1.0, max_canvas_w / float(w), max_canvas_h / float(h))
    disp_w = int(round(w * disp_scale))
    disp_h = int(round(h * disp_scale))
    canvas.configure(width=max_canvas_w, height=max_canvas_h)

    # Tk image handle to avoid GC
    tk_img_ref = {"img": None}
    canvas_img_id = None

    def _render_vis() -> None:
        nonlocal canvas_img_id

        # Read slider values
        thr = int(var_thr.get())
        smk = int(var_smooth.get())
        csz = int(var_close.get())
        osz = int(var_open.get())
        eop = int(var_edit_open.get())

        csz_odd = _odd(max(1, int(csz)))
        osz_odd = _odd(max(1, int(osz)))
        smk_odd = _odd(max(1, int(smk)))

        # cache auto mask when thr/close/open unchanged (biggest cost)
        cache_key = (thr, csz_odd, osz_odd)
        if state.get("_cache_auto_key") == cache_key and state.get("_cache_auto") is not None:
            auto_m = state["_cache_auto"]
        else:
            auto_m = (g < int(thr)).astype(np.uint8) * 255
            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (csz_odd, csz_odd))
            k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (osz_odd, osz_odd))
            auto_m = cv2.morphologyEx(auto_m, cv2.MORPH_CLOSE, k_close)
            auto_m = cv2.morphologyEx(auto_m, cv2.MORPH_OPEN, k_open)
            auto_m = _largest_component(auto_m)
            if int((auto_m > 0).sum()) >= int(min_area):
                auto_m = _fill_holes(auto_m)
            state["_cache_auto"] = auto_m
            state["_cache_auto_key"] = cache_key

        m = _apply_edit_layers(auto_m, edit_add_u8, edit_del_u8)
        min_a = int(var_min_void_area.get())
        remove_voids_on = bool(var_remove_voids.get())
        # cache m after voids so moving only zoom/display does not recompute remove_voids
        m_voids_key = (
            thr, csz_odd, osz_odd, min_a, remove_voids_on,
            int(np.sum(edit_add_u8 > 0)), int(np.sum(edit_del_u8 > 0)),
        )
        if state.get("_cache_m_voids_key") == m_voids_key and state.get("_cache_m_after_voids") is not None:
            m = state["_cache_m_after_voids"]
        else:
            if remove_voids_on:
                to_remove = remove_voids_inside_mask(m, g, min_void_area=min_a)
                m = cv2.bitwise_and(m, cv2.bitwise_not(to_remove))
            state["_cache_m_after_voids"] = m.copy()
            state["_cache_m_voids_key"] = m_voids_key

        state["m_u8"] = m

        # Protrusions: cache by same fingerprint so zoom-only changes skip morphology
        eop_odd = eop if eop >= 3 and eop % 2 == 1 else (eop + 1) if eop >= 3 else 3
        protrusions_key = (m_voids_key, eop_odd)
        if state.get("_cache_protrusions_key") == protrusions_key and state.get("protrusions_u8") is not None:
            protrusions = state["protrusions_u8"]
        else:
            k_edit = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eop_odd, eop_odd))
            base = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_edit)
            protrusions = cv2.bitwise_and(m, cv2.bitwise_not(base))
            state["protrusions_u8"] = protrusions.copy()
            state["_cache_protrusions_key"] = protrusions_key

        # base RGB for display
        if var_show_mask_only.get():
            mask_u8 = (m > 0).astype(np.uint8) * 255
            vis_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
        else:
            vis_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if bool(state["show_mask"]):
                alpha = 0.28
                green = np.zeros_like(vis_bgr, dtype=np.uint8)
                green[:, :, 1] = 255
                m_bool = (m > 0)
                vis_bgr[m_bool] = cv2.addWeighted(vis_bgr[m_bool], 1.0 - alpha, green[m_bool], alpha, 0.0)
            # protrusions overlay (red) — use cached protrusions from above
            protrusions = state.get("protrusions_u8")
            if protrusions is not None and int((protrusions > 0).sum()) > 0:
                p_mask = (protrusions > 0)
                red = np.zeros_like(vis_bgr, dtype=np.uint8)
                red[:, :, 2] = 255
                vis_bgr[p_mask] = cv2.addWeighted(vis_bgr[p_mask], 1.0 - 0.25, red[p_mask], 0.25, 0.0)

        # apply zoom and scale for canvas
        zoom_factor = max(0.5, min(3.0, int(var_zoom.get()) / 100.0))
        effective_scale = disp_scale * zoom_factor
        state["effective_scale"] = effective_scale
        disp_w_zoomed = int(round(w * effective_scale))
        disp_h_zoomed = int(round(h * effective_scale))
        state["disp_w_zoomed"] = disp_w_zoomed
        state["disp_h_zoomed"] = disp_h_zoomed
        vis_bgr = cv2.resize(vis_bgr, (disp_w_zoomed, disp_h_zoomed), interpolation=cv2.INTER_AREA)

        # convert to Tk image (RGB)
        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(vis_rgb)
        tk_img = ImageTk.PhotoImage(pil)
        tk_img_ref["img"] = tk_img

        canvas.configure(scrollregion=(0, 0, disp_w_zoomed, disp_h_zoomed))
        if canvas_img_id is None:
            canvas_img_id = canvas.create_image(0, 0, anchor="nw", image=tk_img)
        else:
            canvas.itemconfigure(canvas_img_id, image=tk_img)
        canvas.coords(canvas_img_id, 0, 0)

    def _tick() -> None:
        if state["dirty"]:
            state["dirty"] = False
            _render_vis()
        root.after(150, _tick)

    # ------------------------
    # Mouse editing on canvas
    # ------------------------
    def _canvas_to_ui_xy(ev) -> tuple[int, int] | None:
        cx = canvas.canvasx(ev.x)
        cy = canvas.canvasy(ev.y)
        eff = state.get("effective_scale") or disp_scale
        dw = state.get("disp_w_zoomed") or disp_w
        dh = state.get("disp_h_zoomed") or disp_h
        if cx < 0 or cy < 0 or cx >= dw or cy >= dh:
            return None
        ix = int(round(cx / eff))
        iy = int(round(cy / eff))
        ix = int(np.clip(ix, 0, w - 1))
        iy = int(np.clip(iy, 0, h - 1))
        return ix, iy

    def on_click(ev) -> None:
        xy = _canvas_to_ui_xy(ev)
        if xy is None:
            return
        ix, iy = xy
        m_current = state["m_u8"]
        if m_current is None or m_current.size == 0:
            return
        mode = state["mode"]
        if mode == "erase_protrusion":
            protrusions = state.get("protrusions_u8")
            if protrusions is None or protrusions.shape != m_current.shape:
                eop = int(var_edit_open.get())
                eop_odd = eop if eop >= 3 and eop % 2 == 1 else max(3, eop + 1)
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eop_odd, eop_odd))
                base = cv2.morphologyEx(m_current, cv2.MORPH_OPEN, k)
                protrusions = cv2.bitwise_and(m_current, cv2.bitwise_not(base))
            cc = _connected_component_from_seed(protrusions, ix, iy, search_r=12)
            if cc.sum() > 0:
                push_undo()
                edit_del_u8[:] = cv2.bitwise_or(edit_del_u8, cc)
                mark_dirty()
        elif mode == "erase_white":
            cc = white_component_at(m_current, g, ix, iy)
            if cc.sum() > 0:
                push_undo()
                edit_del_u8[:] = cv2.bitwise_or(edit_del_u8, cc)
                mark_dirty()
        else:
            hull = _convex_hull_mask(m_current)
            indent = cv2.bitwise_and(hull, cv2.bitwise_not(m_current))
            cc = _connected_component_from_seed(indent, ix, iy, search_r=12)
            if cc.sum() > 0:
                push_undo()
                edit_add_u8[:] = cv2.bitwise_or(edit_add_u8, cc)
                mark_dirty()

    canvas.bind("<Button-1>", on_click)

    def on_wheel(ev) -> None:
        delta = 0
        if ev.num == 5 or (hasattr(ev, "delta") and ev.delta < 0):
            delta = -10
        elif ev.num == 4 or (hasattr(ev, "delta") and ev.delta > 0):
            delta = 10
        if delta == 0:
            return
        z = int(var_zoom.get()) + delta
        z = max(50, min(200, z))
        var_zoom.set(z)
        mark_dirty()

    canvas.bind("<MouseWheel>", on_wheel)
    canvas.bind("<Button-4>", on_wheel)
    canvas.bind("<Button-5>", on_wheel)

    # ------------------------
    # Keybindings
    # ------------------------
    def on_key(ev) -> None:
        ks = (ev.keysym or "").lower()
        if ks in ("return", "kp_enter"):
            do_accept()
            return
        if ks == "escape":
            do_cancel()
            return
        if ks == "e":
            set_mode("erase_protrusion")
            return
        if ks == "w":
            set_mode("erase_white")
            return
        if ks == "a":
            set_mode("add_indent")
            return
        if ks == "u":
            undo_last()
            mark_dirty()
            return
        if ks == "c":
            do_clear()
            return
        if ks == "m":
            var_show_mask.set(not bool(var_show_mask.get()))
            _toggle_mask()
            return

    root.bind("<Key>", on_key)

    # initial mode
    set_mode("erase_protrusion")

    # initial draw
    mark_dirty()
    _tick()

    # Start
    root.mainloop()

    # ------------------------
    # Finalize
    # ------------------------
    if bool(state["cancelled"]) or not bool(state["accepted"]):
        return np.zeros((h0, w0), dtype=bool), {
            "accepted": False,
            "thr": int(var_thr.get()),
            "close": int(_odd(max(1, int(var_close.get())))),
            "open": int(_odd(max(1, int(var_open.get())))),
            "smooth": int(_odd(max(1, int(var_smooth.get())))),
            "scale": float(scale),
            "area_px": 0,
            "perim_px": 0.0,
        }

    last_ui = (state["m_u8"] > 0).astype(np.uint8)

    if crop_bbox is not None:
        y0, x0, y1, x1 = crop_bbox
        last = np.zeros((h0, w0), dtype=bool)
        last[y0:y1, x0:x1] = last_ui.astype(bool)
    else:
        # upscale accepted mask back to original size if we had downsampled
        if scale < 1.0:
            last_u8 = cv2.resize(last_ui.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST)
            last = (last_u8 > 0)
        else:
            last = (last_ui > 0).astype(bool)

    # compute metrics on the final (full-res) accepted mask
    area_px_final = int(last.sum())
    m_u8 = (last.astype(np.uint8) * 255)
    cnts_f, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts_f:
        cnt_f = max(cnts_f, key=cv2.contourArea)
        perim_px_final = float(cv2.arcLength(cnt_f, True))
    else:
        perim_px_final = 0.0

    params = {
        "accepted": True,
        "thr": int(var_thr.get()),
        "close": int(_odd(max(1, int(var_close.get())))),
        "open": int(_odd(max(1, int(var_open.get())))),
        "smooth": int(_odd(max(1, int(var_smooth.get())))),
        "scale": float(scale),
        "area_px": area_px_final,
        "perim_px": perim_px_final,
    }
    return last, params


def overlay_mask_outline_rgb(img_rgb: np.ndarray, mask: np.ndarray, *, color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Draw outer contour of mask on an RGB image (returns RGB)."""
    out = img_rgb.copy()
    m = (mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2 uses BGR, so swap
    bgr = (int(color[2]), int(color[1]), int(color[0]))
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.drawContours(out_bgr, cnts, -1, bgr, int(thickness))
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)