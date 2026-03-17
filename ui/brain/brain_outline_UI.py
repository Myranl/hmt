import numpy as np
import cv2
from ui.brain.mask_utils import _gray_to_u8, _odd, _put_text_box
from ui.brain.threshold_ui import brain_mask_threshold_ui
from ui.brain.mask_morphology import (
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


def _enforce_no_internal_voids(mask_u8: np.ndarray) -> np.ndarray:
    """Force mask to have no internal holes / branches.

    For each row and column we keep only a single contiguous run between
    the first and last foreground pixel; intersection of row- and
    column-wise strips yields a simply connected, hole-free mask.
    """
    m = (mask_u8 > 0).astype(np.uint8)
    h, w = m.shape

    row_filled = np.zeros_like(m, dtype=np.uint8)
    for y in range(h):
        xs = np.where(m[y] > 0)[0]
        if xs.size == 0:
            continue
        x0, x1 = xs[0], xs[-1]
        row_filled[y, x0 : x1 + 1] = 1

    col_filled = np.zeros_like(m, dtype=np.uint8)
    for x in range(w):
        ys = np.where(m[:, x] > 0)[0]
        if ys.size == 0:
            continue
        y0, y1 = ys[0], ys[-1]
        col_filled[y0 : y1 + 1, x] = 1

    simple = (row_filled & col_filled).astype(np.uint8) * 255
    return simple


def _bind_tooltip(widget: tk.Widget, text: str) -> None:
    tip_holder: list = []

    def on_enter(ev: tk.Event) -> None:
        if tip_holder:
            return
        tw = tk.Toplevel(widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{ev.x_root + 12}+{ev.y_root + 12}")
        lbl = ttk.Label(tw, text=text, justify="left", padding=6)
        lbl.pack()
        tip_holder.append(tw)

    def on_leave(_ev: tk.Event) -> None:
        if tip_holder:
            tip_holder[0].destroy()
            tip_holder.clear()

    widget.bind("<Enter>", on_enter)
    widget.bind("<Leave>", on_leave)


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

    var_show_mask = tk.BooleanVar(value=True)
    var_show_mask_only = tk.BooleanVar(value=False)
    var_zoom = tk.IntVar(value=100)
    var_non_complete_contour = tk.BooleanVar(value=False)

    def mark_dirty() -> None:
        state["dirty"] = True

    # Title + short hint
    lbl_title = ttk.Label(ctrl, text="Brain outline", font=("TkDefaultFont", 13, "bold"))
    lbl_title.grid(row=0, column=0, sticky="w", pady=(0, 2))
    lbl_help_title = ttk.Label(ctrl, text=" ? ", cursor="question_arrow")
    lbl_help_title.grid(row=0, column=1, sticky="w")
    lbl_hint = ttk.Label(
        ctrl,
        text="E/W/A/B = modes · U/C/M = undo, clear, mask · Enter/Esc = accept, cancel",
        justify="left",
    )
    lbl_hint.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 8))

    def _add_slider_row(parent: ttk.Frame, row: int, label_text: str, tooltip_text: str, var: tk.IntVar, frm_to: int) -> None:
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w", padx=(0, 2))
        q = ttk.Label(parent, text="?", cursor="question_arrow")
        q.grid(row=row, column=1, sticky="w")
        _bind_tooltip(q, tooltip_text)
        s = ttk.Scale(parent, from_=0, to=frm_to, orient="horizontal", command=lambda _v: mark_dirty())
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
        s.grid(row=row, column=2, sticky="ew", pady=2)
        parent.columnconfigure(2, weight=1)

    # --- Threshold & morphology ---
    lf_morph = ttk.LabelFrame(ctrl, text="Threshold & morphology", padding=6)
    lf_morph.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 6))
    lf_morph.columnconfigure(2, weight=1)
    _add_slider_row(lf_morph, 0, "thr", "Binarization threshold: pixels darker than this are considered tissue.", var_thr, 255)
    _add_slider_row(lf_morph, 1, "smooth", "Contour smoothing (kernel size).", var_smooth, 101)
    _add_slider_row(lf_morph, 2, "close", "Morphological closing: fills small holes in the mask.", var_close, 101)
    _add_slider_row(lf_morph, 3, "open", "Morphological opening: removes small protrusions.", var_open, 101)

    # --- Edit ---
    lf_edit = ttk.LabelFrame(ctrl, text="Edit", padding=6)
    lf_edit.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 6))
    lf_edit.columnconfigure(2, weight=1)
    _add_slider_row(
        lf_edit,
        0,
        "edit_open (protrusions)",
        "Radius for erasing protrusions (E): larger value affects a bigger area per click.",
        var_edit_open,
        101,
    )

    # --- View ---
    lf_view = ttk.LabelFrame(ctrl, text="View", padding=6)
    lf_view.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 6))
    lf_view.columnconfigure(2, weight=1)
    ttk.Label(lf_view, text="zoom %").grid(row=0, column=0, sticky="w", padx=(0, 2))
    q_zoom = ttk.Label(lf_view, text="?", cursor="question_arrow")
    q_zoom.grid(row=0, column=1, sticky="w")
    _bind_tooltip(q_zoom, "Display zoom level.")
    s_zoom = ttk.Scale(lf_view, from_=100, to=200, orient="horizontal")
    s_zoom.set(float(var_zoom.get()))
    def _on_zoom_var(*_a):
        try:
            s_zoom.set(float(var_zoom.get()))
        except Exception:
            pass
        mark_dirty()
    var_zoom.trace_add("write", _on_zoom_var)
    def _on_zoom_scale(val: str):
        try:
            var_zoom.set(max(100, int(float(val) + 0.5)))
        except Exception:
            pass
    s_zoom.configure(command=_on_zoom_scale)
    s_zoom.grid(row=0, column=2, sticky="ew", pady=2)

    def _toggle_mask() -> None:
        state["show_mask"] = bool(var_show_mask.get())
        mark_dirty()

    def _chk_with_help(parent: ttk.Frame, row: int, text: str, var: tk.BooleanVar, cmd, tooltip: str) -> None:
        chk = ttk.Checkbutton(parent, text=text, variable=var, command=cmd)
        chk.grid(row=row, column=0, columnspan=2, sticky="w")
        q = ttk.Label(parent, text="?", cursor="question_arrow")
        q.grid(row=row, column=2, sticky="w")
        _bind_tooltip(q, tooltip)

    _chk_with_help(lf_view, 1, "Show mask (M)", var_show_mask, _toggle_mask, "Show or hide the outline overlay on the image.")
    _chk_with_help(lf_view, 2, "Mask only (B&W)", var_show_mask_only, mark_dirty, "Show only the mask (black & white), no background image.")

    # --- Contour incomplete ---
    frm_non_complete = ttk.LabelFrame(ctrl, text="  Contour incomplete?  ", padding=8)
    frm_non_complete.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(0, 6))
    row_chk = ttk.Frame(frm_non_complete)
    row_chk.pack(anchor="w")
    chk_non_complete = ttk.Checkbutton(
        row_chk,
        text="Non-complete contour (fix in next step)",
        variable=var_non_complete_contour,
        command=mark_dirty,
    )
    chk_non_complete.pack(side="left")
    q_non_complete = ttk.Label(row_chk, text=" ? ", cursor="question_arrow")
    q_non_complete.pack(side="left")
    _bind_tooltip(q_non_complete, "Check if the outline has a gap or is broken; will be handled in the next step.")
    ttk.Label(
        frm_non_complete,
        text="Check if the outline is broken or has a gap.",
        font=("TkDefaultFont", 9),
        foreground="gray",
    ).pack(anchor="w")

    # Mode indicator
    mode_var = tk.StringVar(value="MODE: ERASE PROTRUSION")
    lbl_mode = ttk.Label(ctrl, textvariable=mode_var, font=("TkDefaultFont", 11, "bold"))
    lbl_mode.grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 6))

    # Buttons
    btns = ttk.Frame(ctrl)
    btns.grid(row=7, column=0, columnspan=2, sticky="ew")

    def do_accept() -> None:
        state["accepted"] = True
        for aid in _tick_id:
            try:
                root.after_cancel(aid)
            except Exception:
                pass
        root.destroy()

    def do_cancel() -> None:
        state["cancelled"] = True
        for aid in _tick_id:
            try:
                root.after_cancel(aid)
            except Exception:
                pass
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
    ttk.Button(btns, text="Erase protrusion (E)", command=lambda: set_mode("erase_protrusion")).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))

    ttk.Button(
        btns,
        text="Erase protrusions (brush)",
        command=lambda: set_mode("erase_protrusion_brush"),
    ).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))

    def do_rerun_threshold() -> None:
        gray0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
        bm_res = brain_mask_threshold_ui(gray0, img0, pad=50)
        if bm_res is None:
            return
        mask_full = bm_res.mask  # bool, (h0, w0)
        if crop_bbox is not None:
            y0, x0, y1, x1 = crop_bbox
            mask_crop = (mask_full[y0:y1, x0:x1].astype(np.uint8)) * 255
        else:
            if scale < 1.0:
                new_w, new_h = int(round(w0 * scale)), int(round(h0 * scale))
                mask_crop = cv2.resize(
                    (mask_full.astype(np.uint8) * 255), (new_w, new_h), interpolation=cv2.INTER_NEAREST
                )
            else:
                mask_crop = (mask_full.astype(np.uint8)) * 255
        state["_cache_auto"] = mask_crop.copy()
        state["_cache_auto_key"] = ("threshold_override",)
        edit_add_u8[:] = 0
        edit_del_u8[:] = 0
        mark_dirty()
        _render_vis()

    ttk.Button(btns, text="Re-run threshold…", command=do_rerun_threshold).grid(
        row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0)
    )
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

        # --- Base auto mask, cached by (thr, close, open) ---
        base_key = (thr, csz_odd, osz_odd)
        if state.get("_cache_auto_base_key") == base_key and state.get("_cache_auto_base") is not None:
            base_m = state["_cache_auto_base"].copy()
        else:
            if state.get("_cache_auto_key") == ("threshold_override",) and state.get("_cache_auto") is not None:
                base_m = state["_cache_auto"].copy()
            else:
                auto_m0 = (g < int(thr)).astype(np.uint8) * 255
                k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (csz_odd, csz_odd))
                k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (osz_odd, osz_odd))
                auto_m0 = cv2.morphologyEx(auto_m0, cv2.MORPH_CLOSE, k_close)
                auto_m0 = cv2.morphologyEx(auto_m0, cv2.MORPH_OPEN, k_open)
                auto_m0 = _largest_component(auto_m0)
                base_m = auto_m0
            state["_cache_auto_base"] = base_m.copy()
            state["_cache_auto_base_key"] = base_key

        # --- Smooth version cached by (base_key, smk_odd) ---
        smooth_key = (base_key, smk_odd)
        if state.get("_cache_auto_key") == smooth_key and state.get("_cache_auto") is not None:
            auto_m = state["_cache_auto"].copy()
        else:
            auto_m = base_m.copy()
            if smk_odd > 1:
                k_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smk_odd, smk_odd))
                auto_m = cv2.morphologyEx(auto_m, cv2.MORPH_OPEN, k_s)
                auto_m = cv2.morphologyEx(auto_m, cv2.MORPH_CLOSE, k_s)
            state["_cache_auto"] = auto_m.copy()
            state["_cache_auto_key"] = smooth_key

        m = _apply_edit_layers(auto_m, edit_add_u8, edit_del_u8)
        # On the outline step we want a solid brain region with no internal voids,
        # but we still allow arbitrary contour shape. Just fill all holes here.
        m = _fill_holes(m, binary=True)
        state["m_u8"] = m

        # Protrusions: cache by same fingerprint so zoom-only changes skip morphology
        eop_odd = eop if eop >= 3 and eop % 2 == 1 else (eop + 1) if eop >= 3 else 3
        protrusions_key = ((thr, csz_odd, osz_odd), eop_odd, int(np.sum(edit_add_u8 > 0)), int(np.sum(edit_del_u8 > 0)))
        if state.get("_cache_protrusions_key") == protrusions_key and state.get("protrusions_u8") is not None:
            protrusions = state["protrusions_u8"]
        else:
            k_edit = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eop_odd, eop_odd))
            base = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_edit)
            protrusions = cv2.bitwise_and(m, cv2.bitwise_not(base))
            state["protrusions_u8"] = protrusions.copy()
            state["_cache_protrusions_key"] = protrusions_key

        # base RGB for display: emphasize contour, not filled mask
        mask_u8 = (m > 0).astype(np.uint8) * 255
        if var_show_mask_only.get():
            vis_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
        else:
            vis_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if bool(state["show_mask"]):
                # draw only the outer contour in green
                cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if cnts:
                    cv2.polylines(vis_bgr, cnts, True, (0, 255, 0), 4)
            # protrusions overlay (bright red)
            protrusions = state.get("protrusions_u8")
            if protrusions is not None and int((protrusions > 0).sum()) > 0:
                p_mask = (protrusions > 0)
                red = np.zeros_like(vis_bgr, dtype=np.uint8)
                red[:, :, 2] = 255
                vis_bgr[p_mask] = cv2.addWeighted(vis_bgr[p_mask], 1.0 - 0.55, red[p_mask], 0.55, 0.0)

        # apply zoom and scale for canvas
        zoom_factor = max(1.0, min(3.0, int(var_zoom.get()) / 100.0))
        effective_scale = disp_scale * zoom_factor
        state["effective_scale"] = effective_scale
        disp_w_zoomed = int(round(w * effective_scale))
        disp_h_zoomed = int(round(h * effective_scale))
        state["disp_w_zoomed"] = disp_w_zoomed
        state["disp_h_zoomed"] = disp_h_zoomed
        vis_bgr = cv2.resize(vis_bgr, (disp_w_zoomed, disp_h_zoomed), interpolation=cv2.INTER_AREA)

        # show brush radius around cursor for brush modes
        # convert to Tk image (RGB)
        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(vis_rgb)
        tk_img = ImageTk.PhotoImage(pil, master=canvas)
        tk_img_ref["img"] = tk_img

        canvas.configure(scrollregion=(0, 0, disp_w_zoomed, disp_h_zoomed))
        if canvas_img_id is None:
            canvas_img_id = canvas.create_image(0, 0, anchor="nw", image=tk_img)
        else:
            canvas.itemconfigure(canvas_img_id, image=tk_img)
        canvas.coords(canvas_img_id, 0, 0)

    _tick_id: list = []  # mutable to store id for cancel on destroy

    def _canvas_to_ui_xy(ev: tk.Event) -> tuple[int, int] | None:
        cx = canvas.canvasx(ev.x)
        cy = canvas.canvasy(ev.y)
        eff = max(state.get("effective_scale", disp_scale), 1e-6)
        dw = state.get("disp_w_zoomed") or int(round(w * eff))
        dh = state.get("disp_h_zoomed") or int(round(h * eff))
        if cx < 0 or cy < 0 or cx >= dw or cy >= dh:
            return None
        ix = int(round(cx / eff))
        iy = int(round(cy / eff))
        ix = int(np.clip(ix, 0, w - 1))
        iy = int(np.clip(iy, 0, h - 1))
        return ix, iy

    def _tick() -> None:
        if state["dirty"]:
            state["dirty"] = False
            _render_vis()
        _tick_id.clear()
        _tick_id.append(root.after(150, _tick))

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
        state["brush_cx"] = ix
        state["brush_cy"] = iy
        m_current = state["m_u8"]
        if m_current is None or m_current.size == 0:
            return
        mode = state["mode"]
        if mode == "erase_protrusion_brush":
            # local removal of protrusions within brush radius
            protrusions = state.get("protrusions_u8")
            if protrusions is None or protrusions.shape != m_current.shape:
                # recompute protrusions on current mask
                eop = int(var_edit_open.get())
                eop_odd = eop if eop >= 3 and eop % 2 == 1 else max(3, eop + 1)
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eop_odd, eop_odd))
                base = cv2.morphologyEx(m_current, cv2.MORPH_OPEN, k)
                protrusions = cv2.bitwise_and(m_current, cv2.bitwise_not(base))
            brush_r = int(50 / max(state.get("effective_scale", 1.0), 1e-6))
            brush = np.zeros_like(m_current, dtype=np.uint8)
            cv2.circle(brush, (ix, iy), max(1, brush_r), 1, thickness=-1)
            local = (protrusions > 0) & (brush > 0)
            if local.any():
                push_undo()
                edit_del_u8[local] = 255
                # recompute protrusions only, based on updated mask
                m_after = _apply_edit_layers(m_current, edit_add_u8, edit_del_u8)
                eop = int(var_edit_open.get())
                eop_odd = eop if eop >= 3 and eop % 2 == 1 else max(3, eop + 1)
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eop_odd, eop_odd))
                base = cv2.morphologyEx(m_after, cv2.MORPH_OPEN, k)
                protrusions_new = cv2.bitwise_and(m_after, cv2.bitwise_not(base))
                state["protrusions_u8"] = protrusions_new.copy()
                mark_dirty()
        elif mode == "erase_protrusion":
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
                # пересчитываем только protrusions после изменения маски
                m_after = _apply_edit_layers(m_current, edit_add_u8, edit_del_u8)
                eop = int(var_edit_open.get())
                eop_odd = eop if eop >= 3 and eop % 2 == 1 else max(3, eop + 1)
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eop_odd, eop_odd))
                base = cv2.morphologyEx(m_after, cv2.MORPH_OPEN, k)
                protrusions_new = cv2.bitwise_and(m_after, cv2.bitwise_not(base))
                state["protrusions_u8"] = protrusions_new.copy()
                mark_dirty()
        elif mode == "erase_white":
            cc = white_component_at(m_current, g, ix, iy)
            if cc.sum() > 0:
                push_undo()
                edit_del_u8[:] = cv2.bitwise_or(edit_del_u8, cc)
                mark_dirty()
        elif mode == "add_white":
            # Add a white void back into the mask.
            # Here we don't try to be clever with intensities – we just take
            # the connected background component (0 region) inside the current
            # brain contour and add it to the mask.
            #
            # 1) background pixels (where current mask == 0)
            bg_u8 = ((m_current == 0).astype(np.uint8)) * 255
            # 2) restrict to convex hull so we never grow outside the brain
            hull_u8 = _convex_hull_mask(m_current)
            search_u8 = cv2.bitwise_and(bg_u8, hull_u8)
            # 3) connected component from the click
            cc = _connected_component_from_seed(search_u8, ix, iy, search_r=15)
            if cc.sum() > 0:
                push_undo()
                edit_add_u8[:] = cv2.bitwise_or(edit_add_u8, cc)
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

    def on_motion(ev: tk.Event) -> None:
        xy = _canvas_to_ui_xy(ev)
        if xy is None:
            return
        ix, iy = xy
        state["brush_cx"] = ix
        state["brush_cy"] = iy
        if state.get("mode") == "erase_protrusion_brush":
            mark_dirty()

    canvas.bind("<Motion>", on_motion)

    def on_wheel(ev) -> None:
        delta = 0
        if ev.num == 5 or (hasattr(ev, "delta") and ev.delta < 0):
            delta = -10
        elif ev.num == 4 or (hasattr(ev, "delta") and ev.delta > 0):
            delta = 10
        if delta == 0:
            return
        z = int(var_zoom.get()) + delta
        z = max(100, min(200, z))
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
        if ks == "b":
            set_mode("add_white")
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

    def _on_destroy(_ev=None) -> None:
        for aid in _tick_id:
            try:
                root.after_cancel(aid)
            except Exception:
                pass
        _tick_id.clear()

    root.protocol("WM_DELETE_WINDOW", do_cancel)
    root.bind("<Destroy>", _on_destroy)

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
            "cancelled": bool(state["cancelled"]),
            "thr": int(var_thr.get()),
            "close": int(_odd(max(1, int(var_close.get())))),
            "open": int(_odd(max(1, int(var_open.get())))),
            "smooth": int(_odd(max(1, int(var_smooth.get())))),
            "scale": float(scale),
            "area_px": 0,
            "perim_px": 0.0,
            "non_complete_contour": bool(var_non_complete_contour.get()),
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
        "cancelled": False,
        "thr": int(var_thr.get()),
        "close": int(_odd(max(1, int(var_close.get())))),
        "open": int(_odd(max(1, int(var_open.get())))),
        "smooth": int(_odd(max(1, int(var_smooth.get())))),
        "scale": float(scale),
        "area_px": area_px_final,
        "perim_px": perim_px_final,
        "non_complete_contour": bool(var_non_complete_contour.get()),
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