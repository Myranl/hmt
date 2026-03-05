import numpy as np
import cv2
from ui.brain_mask.mask_utils import _gray_to_u8, _odd, _put_text_box, smooth_contour
from ui.brain_mask.mask_morphology import _fill_holes, _largest_component, _convex_hull_mask, _apply_edit_layers
from ui.brain_mask.mask_compute import compute_mask

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk


# --- Helper functions ---

def _connected_component_from_seed(mask_u8: np.ndarray, x: int, y: int, *, search_r: int = 12) -> np.ndarray:
    """Return the connected component (uint8 0/255) selected by a click.

    - `mask_u8` can be 0/1 or 0/255; any non-zero is treated as foreground.
    - If (x, y) is not on foreground, we search the nearest foreground pixel within `search_r`.
    - Uses floodFill (fast) to extract the component with 8-connectivity.
    """
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

    # Flood fill the foreground component containing (x, y)
    flood = m.copy()
    ff = np.zeros((h + 2, w + 2), np.uint8)

    new_val = 128  # marker value distinct from 0 and 255
    flags = 8  # 8-connectivity
    cv2.floodFill(flood, ff, (x, y), new_val, flags=flags)

    comp = (flood == new_val).astype(np.uint8) * 255
    return comp


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

    # pick closest in Euclidean distance
    dx = xs.astype(np.int32) + x0 - x
    dy = ys.astype(np.int32) + y0 - y
    d2 = dx * dx + dy * dy
    i = int(np.argmin(d2))
    return int(xs[i] + x0), int(ys[i] + y0)

def brain_outline_ui(
    img_rgb: np.ndarray,
    *,
    window: str = "Brain outline",
    init_thr: int = 170,
    init_smooth: int = 15,
    init_close: int = 11,
    init_open: int = 5,
    min_area: int = 20000,
    downsample_max_side: int = 1200,
) -> tuple[np.ndarray, dict]:
    """Tk UI to tune threshold + contour smoothing + quick manual mask edits.

    This replaces the OpenCV HighGUI window/trackbars to avoid Windows DPI blur/flicker.

    Controls:
      - thr: threshold (0..255) where tissue is darker
      - smooth: contour smoothing strength (odd kernel size)
      - close/open: morphology kernel sizes
      - edit_open: how aggressive ERASE detects protrusions

    Mouse:
      - Click on mask: ERASE protrusion (mode=ERASE)
      - Click on hull-indent: ADD indentation (mode=ADD)

    Keys:
      - Enter: Accept
      - Esc: Cancel
      - E/A: switch ERASE/ADD
      - U: undo
      - C: clear manual edits
      - M: toggle mask overlay
      - P: toggle protrusions overlay

    Returns:
      (mask_bool_fullres, params_dict)
    """

    # ------------------------
    # Downsample for UI speed
    # ------------------------
    img0 = img_rgb
    h0, w0 = img0.shape[:2]
    scale = min(1.0, downsample_max_side / float(max(h0, w0)))
    if scale < 1.0:
        new_w = int(round(w0 * scale))
        new_h = int(round(h0 * scale))
        img = cv2.resize(img0, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img = img0

    # grayscale on the UI image
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = g.shape[:2]

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
        "mode": "erase",
        "show_mask": True,
        "show_protrusions": True,
        "accepted": False,
        "cancelled": False,
        "dirty": True,
        "last_thr": None,
        "last_close": None,
        "last_open": None,
        "last_smooth": None,
        "last_edit_open": None,
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

    # Canvas (image)
    canvas = tk.Canvas(frm, highlightthickness=0, bg="#111")
    canvas.grid(row=0, column=0, sticky="nsew")

    # Right controls
    ctrl = ttk.Frame(frm)
    ctrl.grid(row=0, column=1, padx=(12, 0), sticky="ns")

    # Vars
    var_thr = tk.IntVar(value=int(init_thr))
    var_smooth = tk.IntVar(value=int(init_smooth))
    var_close = tk.IntVar(value=int(init_close))
    var_open = tk.IntVar(value=int(init_open))
    var_edit_open = tk.IntVar(value=41)

    var_show_mask = tk.BooleanVar(value=True)
    var_show_prot = tk.BooleanVar(value=True)

    # Labels / instructions
    lbl_title = ttk.Label(ctrl, text="Brain outline", font=("TkDefaultFont", 13, "bold"))
    lbl_title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

    lbl_hint = ttk.Label(
        ctrl,
        text=(
            "Mouse: click to ERASE protrusions / ADD indents\n"
            "Keys: E/A mode, U undo, C clear, M mask, P protrusions\n"
            "Enter accept, Esc cancel"
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
    _add_slider(6, "edit_open", var_edit_open, 151)

    # Toggles
    def _toggle_mask() -> None:
        state["show_mask"] = bool(var_show_mask.get())
        mark_dirty()

    def _toggle_prot() -> None:
        state["show_protrusions"] = bool(var_show_prot.get())
        mark_dirty()

    chk_mask = ttk.Checkbutton(ctrl, text="show mask (M)", variable=var_show_mask, command=_toggle_mask)
    chk_prot = ttk.Checkbutton(ctrl, text="show protrusions (P)", variable=var_show_prot, command=_toggle_prot)
    chk_mask.grid(row=7, column=0, columnspan=2, sticky="w", pady=(10, 0))
    chk_prot.grid(row=8, column=0, columnspan=2, sticky="w")

    # Mode indicator
    mode_var = tk.StringVar(value="MODE: ERASE")
    lbl_mode = ttk.Label(ctrl, textvariable=mode_var, font=("TkDefaultFont", 12, "bold"))
    lbl_mode.grid(row=9, column=0, columnspan=2, sticky="w", pady=(12, 6))

    # Metrics
    metrics_var = tk.StringVar(value="")
    lbl_metrics = ttk.Label(ctrl, textvariable=metrics_var, justify="left")
    lbl_metrics.grid(row=10, column=0, columnspan=2, sticky="w", pady=(0, 10))

    # Buttons
    btns = ttk.Frame(ctrl)
    btns.grid(row=11, column=0, columnspan=2, sticky="ew")

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
        mode_var.set(f"MODE: {kind.upper()}")

    ttk.Button(btns, text="Accept (Enter)", command=do_accept).grid(row=0, column=0, sticky="ew", padx=(0, 6))
    ttk.Button(btns, text="Cancel (Esc)", command=do_cancel).grid(row=0, column=1, sticky="ew")
    ttk.Button(btns, text="Undo (U)", command=lambda: (undo_last(), mark_dirty())).grid(row=1, column=0, sticky="ew", pady=(6, 0), padx=(0, 6))
    ttk.Button(btns, text="Clear edits (C)", command=do_clear).grid(row=1, column=1, sticky="ew", pady=(6, 0))
    ttk.Button(btns, text="ERASE (E)", command=lambda: set_mode("erase")).grid(row=2, column=0, sticky="ew", pady=(6, 0), padx=(0, 6))
    ttk.Button(btns, text="ADD (A)", command=lambda: set_mode("add")).grid(row=2, column=1, sticky="ew", pady=(6, 0))
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
    canvas.configure(width=disp_w, height=disp_h)

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

        # normalize odd sizes
        csz_odd = _odd(max(1, int(csz)))
        osz_odd = _odd(max(1, int(osz)))
        smk_odd = _odd(max(1, int(smk)))

        # compute auto mask
        auto_m = (g < int(thr)).astype(np.uint8) * 255
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (csz_odd, csz_odd))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (osz_odd, osz_odd))
        auto_m = cv2.morphologyEx(auto_m, cv2.MORPH_CLOSE, k_close)
        auto_m = cv2.morphologyEx(auto_m, cv2.MORPH_OPEN, k_open)
        auto_m = _largest_component(auto_m)
        if int((auto_m > 0).sum()) >= int(min_area):
            auto_m = _fill_holes(auto_m)

        m = _apply_edit_layers(auto_m, edit_add_u8, edit_del_u8)
        state["m_u8"] = m

        # contour (for metrics)
        cnts, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea) if cnts else None
        cnt_s = smooth_contour(cnt, smk_odd) if cnt is not None else None

        area_px = int((m > 0).sum())
        cnt_use = cnt_s if cnt_s is not None else cnt
        perim_px = float(cv2.arcLength(cnt_use, True)) if cnt_use is not None else 0.0
        metrics_var.set(f"area={area_px} px\nperim={perim_px:.1f} px\nthr={thr}  close={csz_odd} open={osz_odd} smooth={smk_odd}  edit_open={eop}")

        # base RGB for display
        vis_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # mask overlay
        if bool(state["show_mask"]):
            alpha = 0.28
            green = np.zeros_like(vis_bgr, dtype=np.uint8)
            green[:, :, 1] = 255
            m_bool = (m > 0)
            vis_bgr[m_bool] = cv2.addWeighted(vis_bgr[m_bool], 1.0 - alpha, green[m_bool], alpha, 0.0)

        # protrusions overlay (subtle red)
        if bool(state["show_protrusions"]):
            edit_open = int(eop)
            if edit_open < 3:
                edit_open = 3
            if edit_open % 2 == 0:
                edit_open += 1
            k_edit = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edit_open, edit_open))
            base = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_edit)
            protrusions = cv2.bitwise_and(m, cv2.bitwise_not(base))
            if int((protrusions > 0).sum()) > 0:
                p_mask = (protrusions > 0)
                alpha = 0.25
                red = np.zeros_like(vis_bgr, dtype=np.uint8)
                red[:, :, 2] = 255
                vis_bgr[p_mask] = cv2.addWeighted(vis_bgr[p_mask], 1.0 - alpha, red[p_mask], alpha, 0.0)

        # downscale for canvas
        if disp_scale < 1.0:
            vis_bgr = cv2.resize(vis_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

        # convert to Tk image (RGB)
        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(vis_rgb)
        tk_img = ImageTk.PhotoImage(pil)
        tk_img_ref["img"] = tk_img

        if canvas_img_id is None:
            canvas_img_id = canvas.create_image(0, 0, anchor="nw", image=tk_img)
        else:
            canvas.itemconfigure(canvas_img_id, image=tk_img)

    def _tick() -> None:
        # only redraw when something changed
        if state["dirty"]:
            state["dirty"] = False
            _render_vis()
        root.after(30, _tick)

    # ------------------------
    # Mouse editing on canvas
    # ------------------------
    def _canvas_to_ui_xy(ev) -> tuple[int, int] | None:
        x = int(ev.x)
        y = int(ev.y)
        if x < 0 or y < 0 or x >= disp_w or y >= disp_h:
            return None
        # map back to UI image coords
        ix = int(round(x / disp_scale)) if disp_scale > 0 else x
        iy = int(round(y / disp_scale)) if disp_scale > 0 else y
        ix = int(np.clip(ix, 0, w - 1))
        iy = int(np.clip(iy, 0, h - 1))
        return ix, iy

    def on_click(ev) -> None:
        xy = _canvas_to_ui_xy(ev)
        if xy is None:
            return
        ix, iy = xy

        m_current_u8 = state["m_u8"]
        if m_current_u8 is None or m_current_u8.size == 0:
            return

        if state["mode"] == "erase":
            edit_open = int(var_edit_open.get())
            if edit_open < 3:
                edit_open = 3
            if edit_open % 2 == 0:
                edit_open += 1
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edit_open, edit_open))
            base = cv2.morphologyEx(m_current_u8, cv2.MORPH_OPEN, k)
            protrusions = cv2.bitwise_and(m_current_u8, cv2.bitwise_not(base))
            cc = _connected_component_from_seed(protrusions, ix, iy, search_r=12)
            if cc.sum() > 0:
                push_undo()
                edit_del_u8[:] = cv2.bitwise_or(edit_del_u8, cc)
                mark_dirty()
        else:
            hull = _convex_hull_mask(m_current_u8)
            indent = cv2.bitwise_and(hull, cv2.bitwise_not(m_current_u8))
            cc = _connected_component_from_seed(indent, ix, iy, search_r=12)
            if cc.sum() > 0:
                push_undo()
                edit_add_u8[:] = cv2.bitwise_or(edit_add_u8, cc)
                mark_dirty()

    canvas.bind("<Button-1>", on_click)

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
            set_mode("erase")
            return
        if ks == "a":
            set_mode("add")
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
        if ks == "p":
            var_show_prot.set(not bool(var_show_prot.get()))
            _toggle_prot()
            return

    root.bind("<Key>", on_key)

    # initial mode
    set_mode("erase")

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

    # upscale accepted mask back to original size
    if scale < 1.0:
        last_u8 = cv2.resize(last_ui, (w0, h0), interpolation=cv2.INTER_NEAREST)
        last = (last_u8 > 0)
    else:
        last = (last_ui > 0)

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