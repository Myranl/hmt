import numpy as np
import cv2
from typing import Any, Dict, Literal, Tuple, Union
import json
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

import customtkinter as ctk  # type: ignore[import-untyped]

from ui.common.theme import setup_theme, get_base_font, get_small_muted_font
from ui.common.widgets import (
    create_card_frame,
    create_primary_button,
    create_secondary_button,
)
from ui.file_selection.settings import load_folder_choices


# Helper: select components on a background RGB image (CTk UI, same style as folders / brain outline)
def select_components_on_background(
    sketch_u8_roi: np.ndarray,
    bg_rgb_roi: np.ndarray,
    *,
    window: str,
    init_selected: np.ndarray | None = None,
    init_cuts=None,
    allow_open_bins_ui: bool = False,
) -> Union[tuple[np.ndarray, np.ndarray], Literal["edit_bins"]]:
    """Click to toggle connected components. CTk UI: image card + controls (same style as brain outline).

    If ``allow_open_bins_ui`` is True, a toolbar control opens the 3-bin sketch step: this window closes
    and the caller should return ``\"edit_bins\"`` so the pipeline can run ``run_bins_ui`` and reopen pick.
    """

    base0 = sketch_u8_roi.copy()
    base = base0.copy()

    if init_selected is not None and init_selected.shape == base.shape:
        selected = (init_selected > 0).astype(np.uint8).copy()
    else:
        selected = np.zeros(base.shape, dtype=np.uint8)

    history: list[int] = []

    bg = bg_rgb_roi.copy()
    if bg.ndim == 2:
        bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2RGB)
    bg_bgr0 = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)

    mode: str = "pick"
    pending_pt: tuple[int, int] | None = None
    cut_thickness = 7
    add_thickness = 7
    cuts: list[tuple[tuple[int, int], tuple[int, int]]] = []
    adds: list[tuple[tuple[int, int], tuple[int, int]]] = []
    undo_stack: list[tuple[str, object]] = []

    def recompute_labels() -> tuple[np.ndarray, np.ndarray]:
        nonlocal base
        base = base0.copy()
        if cuts:
            for (x1, y1), (x2, y2) in cuts:
                cv2.line(base, (int(x1), int(y1)), (int(x2), int(y2)), 127, int(cut_thickness))
        if adds:
            for (x1, y1), (x2, y2) in adds:
                cv2.line(base, (int(x1), int(y1)), (int(x2), int(y2)), 255, int(add_thickness))
        fg = (base != 127).astype(np.uint8)
        _num, lab, _stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        return lab, fg

    lab, fg = recompute_labels()
    selected[(base == 127)] = 0

    def compute_edges() -> np.ndarray:
        """Crisp boundaries between discrete sketch bins.

        We work directly on the quantized values in `base` instead of Canny.
        This is more stable for images with a few discrete levels (0 / 127 / 255)
        and gives cleaner contours after downscaling.
        """
        a = np.asarray(base, dtype=np.uint8)
        h_e, w_e = a.shape[:2]
        edges = np.zeros((h_e, w_e), dtype=np.uint8)

        # mark boundaries where neighboring pixels belong to different bins
        diff_r = a[:, 1:] != a[:, :-1]
        diff_d = a[1:, :] != a[:-1, :]

        edges[:, 1:][diff_r] = 255
        edges[:, :-1][diff_r] = 255
        edges[1:, :][diff_d] = 255
        edges[:-1, :][diff_d] = 255

        return edges

    edges = compute_edges()

    # Build display image: no text/black box on image — all instructions in right panel
    def redraw() -> np.ndarray:
        disp = bg_bgr0.copy()
        disp = (0.85 * disp).astype(np.uint8)
        disp[edges > 0] = (255, 255, 255)
        m = selected > 0
        if np.any(m):
            alpha = 0.4
            green = np.zeros_like(disp)
            green[:] = (0, 255, 0)
            disp[m] = (alpha * green[m] + (1 - alpha) * disp[m]).astype(np.uint8)
        if mode in ("cut", "add") and pending_pt is not None:
            cv2.circle(disp, (int(pending_pt[0]), int(pending_pt[1])), 7, (0, 255, 255), -1)
        return disp

    # --- CTk layout (toolbar + dark canvas + stats; wheel zoom, space+drag pan) ---
    setup_theme()
    parent = tk._default_root
    if parent is not None and isinstance(parent, ctk.CTk):
        root = ctk.CTkToplevel(parent)
        root.transient(parent)
    else:
        root = ctk.CTk()
    root.title(window)
    root.configure(fg_color="white")
    root.minsize(960, 520)
    root.grid_columnconfigure(1, weight=1)
    # No vertical stretch on body — avoids empty white band between image and footer
    root.grid_rowconfigure(1, weight=0)
    try:
        root.grab_set()
    except Exception:
        pass

    img_h, img_w = int(base.shape[0]), int(base.shape[1])
    try:
        screen_w = int(root.winfo_screenwidth())
        screen_h = int(root.winfo_screenheight())
    except Exception:
        screen_w, screen_h = 1400, 900

    # Layout constants (second-screen style)
    HDR_H = 52
    FTR_ROW = 44  # footer: zoom + Skip Image (left) · hint + Done (right)
    LBAR_W = 76
    GRID_PAD_X = 36  # horizontal margins (padx) around content
    GRID_PAD_Y = 32
    max_win_w = min(1400, int(screen_w * 0.92))
    max_body_h = int(screen_h * 0.86) - HDR_H - FTR_ROW - GRID_PAD_Y

    target_canvas_w = max(480, max_win_w - LBAR_W - GRID_PAD_X)
    fit_scale = target_canvas_w / float(img_w)
    nat_canvas_h = int(round(img_h * fit_scale))
    # Horizontal scrollbar row under the canvas (ttk scrollbar height)
    SCROLL_X_H = 22
    # Canvas viewport: never force a min height > image (was max(420,…) → black band below bitmap)
    initial_canvas_h = max(80, min(max_body_h, nat_canvas_h))
    viewport_w = int(target_canvas_w)
    # Keep canvas viewport size fixed when zooming: only scrollregion grows. Using max_body_h here
    # made ch jump up to ~full screen on zoom and pushed the footer below the window.
    canvas_viewport_h_max = int(initial_canvas_h)

    zoom_mul = [1.0]  # multiplier on top of fit-to-width scale
    disp_scale = [fit_scale * zoom_mul[0]]
    disp_w = [int(round(img_w * disp_scale[0]))]
    disp_h = [int(round(img_h * disp_scale[0]))]

    settings = load_folder_choices()
    graphs_convert = bool(settings.get("graphs_convert_enabled", False))
    graphs_unit = str(settings.get("graphs_unit", "mm")).lower()
    try:
        graphs_ppu = float(settings.get("graphs_pixels_per_unit", 100.0))
    except Exception:
        graphs_ppu = 100.0

    def fmt_px(n: int) -> str:
        return f"{int(n):,}".replace(",", " ")

    def count_selected_regions() -> int:
        m = (selected > 0).astype(np.uint8)
        if not np.any(m):
            return 0
        n, _, _, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        return int(n - 1)

    def area_in_phys_units(n_px: int) -> str | None:
        if not graphs_convert or graphs_ppu <= 0:
            return None
        if graphs_unit == "mm":
            mm2 = n_px / (graphs_ppu**2)
            return f"{mm2:.2f}"
        if graphs_unit == "cm":
            cm2 = n_px / (graphs_ppu**2)
            return f"{cm2:.2f}"
        return None

    # -------- Header --------
    header = ctk.CTkFrame(root, fg_color="transparent")
    header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=18, pady=(10, 6))
    header.grid_columnconfigure(1, weight=1)

    ctk.CTkLabel(
        header,
        text="Pick hippocampus (green)",
        font=ctk.CTkFont(size=17, weight="bold"),
    ).grid(row=0, column=0, sticky="w")

    badge_lbl = ctk.CTkLabel(
        header,
        text="PICK",
        font=ctk.CTkFont(size=12, weight="bold"),
        corner_radius=8,
        fg_color=("#cfe8f6", "#2b4a5e"),
        text_color=("gray20", "gray90"),
        width=56,
        height=28,
    )
    badge_lbl.grid(row=0, column=1, sticky="w", padx=(12, 0))

    header_sel = ctk.CTkLabel(
        header,
        text="Selected: 0 px · 0 reg.",
        font=get_base_font(),
        text_color=("gray35", "gray70"),
        anchor="e",
    )
    header_sel.grid(row=0, column=2, sticky="e")

    def _header_selection_text(n_sel: int) -> str:
        nreg = count_selected_regions()
        parts = [f"Selected: {fmt_px(n_sel)} px", f"{nreg} reg."]
        phys = area_in_phys_units(n_sel)
        unit_s = "mm²" if graphs_unit == "mm" else ("cm²" if graphs_unit == "cm" else "")
        if phys is not None and unit_s:
            parts.append(f"≈ {phys} {unit_s}")
        return " · ".join(parts)

    # -------- Left toolbar (square tool buttons) --------
    tool_card = create_card_frame(root)
    tool_card.grid(row=1, column=0, sticky="n", padx=(18, 6), pady=(0, 0))
    tool_card.grid_propagate(False)
    tool_card.configure(width=LBAR_W)

    tool_font = ctk.CTkFont(size=18, weight="bold")
    inactive_fg = ("#e8e8e8", "gray35")
    active_add = ("#8fd4a8", "#2d7a52")
    active_cut = ("#f0c4c4", "#8b3a3a")

    def _make_tool_btn(parent, text_main: str, key: str, command) -> ctk.CTkButton:
        return ctk.CTkButton(
            parent,
            text=f"{text_main}\n{key}",
            width=52,
            height=52,
            corner_radius=10,
            font=tool_font,
            command=command,
            fg_color=inactive_fg,
            hover_color=("#d0d0d0", "gray45"),
            text_color=("gray20", "gray90"),
        )

    # -------- Center: dark canvas --------
    img_card = create_card_frame(root, fg_color=("#2a2a2a", "#1a1a1a"))
    # "new" = top-align; avoids stretching the dark card when row is taller than the image
    img_card.grid(row=1, column=1, sticky="new", padx=(6, 18), pady=(0, 0))
    img_card.columnconfigure(0, weight=1)
    img_card.rowconfigure(0, weight=0)
    canvas_holder = tk.Frame(img_card, bg="#2a2a2a")
    canvas_holder.grid(row=0, column=0, sticky="new", padx=2, pady=2)
    canvas_holder.columnconfigure(0, weight=1)
    canvas_holder.rowconfigure(0, weight=0)
    canvas_holder.rowconfigure(1, weight=0)
    scroll_y = ttk.Scrollbar(canvas_holder)
    scroll_x = ttk.Scrollbar(canvas_holder, orient=tk.HORIZONTAL)
    canvas = tk.Canvas(canvas_holder, highlightthickness=0, bg="#252525")
    canvas.grid(row=0, column=0, sticky="nw")
    scroll_y.grid(row=0, column=1, sticky="ns")
    scroll_x.grid(row=1, column=0, sticky="ew")
    canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
    scroll_y.configure(command=canvas.yview)
    scroll_x.configure(command=canvas.xview)
    canvas.configure(width=viewport_w, height=initial_canvas_h)

    # -------- Footer (zoom + hint + compact actions) --------
    footer = ctk.CTkFrame(root, fg_color="transparent")
    footer.grid(row=2, column=0, columnspan=2, sticky="ew", padx=18, pady=(8, 12))
    footer.grid_columnconfigure(2, weight=1)
    zoom_var = tk.StringVar(value="Zoom: 100%")
    ctk.CTkLabel(
        footer,
        textvariable=zoom_var,
        font=get_small_muted_font(),
        text_color="gray50",
    ).grid(row=0, column=0, sticky="w")
    ctk.CTkLabel(
        footer,
        text=(
            "Scroll to zoom · Space + drag to pan · Esc = skip image"
            + (" · B = 3-bin sketch" if allow_open_bins_ui else "")
        ),
        font=get_small_muted_font(),
        text_color="gray50",
    ).grid(row=0, column=3, sticky="e", padx=(0, 6))

    result: list[tuple[np.ndarray, np.ndarray] | None] = [None]
    cancelled = [False]
    edit_bins_redirect = [False]
    space_held = [False]
    pan_drag = [False]
    tk_img_ref: dict = {}
    canvas_img_id: list = []

    def _sync_badge_and_tools() -> None:
        if mode == "add":
            badge_lbl.configure(text="ADD", fg_color=("#c8f0d4", "#1e5c36"))
        elif mode == "cut":
            badge_lbl.configure(text="CUT", fg_color=("#f5d0d0", "#6b2a2a"))
        else:
            badge_lbl.configure(text="PICK", fg_color=("#cfe8f6", "#2b4a5e"))
        try:
            btn_add.configure(fg_color=active_add if mode == "add" else inactive_fg)
            btn_cut.configure(fg_color=active_cut if mode == "cut" else inactive_fg)
        except Exception:
            pass

    def _recompute_scale_from_zoom() -> None:
        disp_scale[0] = fit_scale * zoom_mul[0]
        disp_w[0] = max(1, int(round(img_w * disp_scale[0])))
        disp_h[0] = max(1, int(round(img_h * disp_scale[0])))
        pct = int(round(zoom_mul[0] * 100))
        zoom_var.set(f"Zoom: {pct}%")

    def _refresh() -> None:
        _recompute_scale_from_zoom()
        disp = redraw()
        ds = disp_scale[0]
        dw, dh = disp_w[0], disp_h[0]
        if abs(ds - 1.0) > 1e-9 or dw != img_w or dh != img_h:
            disp = cv2.resize(disp, (dw, dh), interpolation=cv2.INTER_NEAREST)
        n_sel = int((selected > 0).sum())
        header_sel.configure(text=_header_selection_text(n_sel))
        _sync_badge_and_tools()
        # Snug to image when smaller than caps; never grow height past opening viewport (zoom → scroll)
        cw = int(min(viewport_w, max(1, dw)))
        ch = int(min(canvas_viewport_h_max, max(1, dh)))
        canvas.configure(width=cw, height=ch)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tk_img = ImageTk.PhotoImage(pil, master=root)
        tk_img_ref["img"] = tk_img
        if not canvas_img_id:
            canvas_img_id.append(canvas.create_image(0, 0, anchor="nw", image=tk_img))
        else:
            canvas.itemconfigure(canvas_img_id[0], image=tk_img)
        canvas.configure(scrollregion=(0, 0, dw, dh))

    def _canvas_xy(ev) -> tuple[int, int] | None:
        cx, cy = canvas.canvasx(ev.x), canvas.canvasy(ev.y)
        dw, dh = disp_w[0], disp_h[0]
        if cx < 0 or cy < 0 or cx >= dw or cy >= dh:
            return None
        ix = int(round(cx / disp_scale[0]))
        iy = int(round(cy / disp_scale[0]))
        return (min(max(ix, 0), img_w - 1), min(max(iy, 0), img_h - 1))

    def _apply_zoom_at(ev, direction: int) -> None:
        """direction +1 = zoom in, -1 = zoom out; keep point under cursor stable."""
        canvas.update_idletasks()
        mx = canvas.canvasx(ev.x)
        my = canvas.canvasy(ev.y)
        dw0, dh0 = disp_w[0], disp_h[0]
        if dw0 <= 0 or dh0 <= 0:
            return
        fx = mx / float(dw0)
        fy = my / float(dh0)
        vis_w = max(1, canvas.winfo_width())
        vis_h = max(1, canvas.winfo_height())
        step = 1.12 if direction > 0 else 1.0 / 1.12
        zoom_mul[0] = min(8.0, max(0.2, zoom_mul[0] * step))

        _refresh()
        dw1, dh1 = disp_w[0], disp_h[0]

        # New scroll so (fx, fy) stays under (ev.x, ev.y)
        n_scroll_x = fx * dw1 - ev.x
        n_scroll_y = fy * dh1 - ev.y
        sx1 = max(0.0, min(max(0.0, dw1 - vis_w), n_scroll_x))
        sy1 = max(0.0, min(max(0.0, dh1 - vis_h), n_scroll_y))
        frac_x = sx1 / max(1e-6, dw1 - vis_w) if dw1 > vis_w else 0.0
        frac_y = sy1 / max(1e-6, dh1 - vis_h) if dh1 > vis_h else 0.0
        canvas.xview_moveto(frac_x)
        canvas.yview_moveto(frac_y)

    def _on_mousewheel(ev) -> None:
        d = int(getattr(ev, "delta", 0) or 0)
        if d == 0:
            return
        # Windows: ±120 steps; macOS may send smaller deltas
        direction = 1 if d > 0 else -1
        _apply_zoom_at(ev, direction)

    def _on_wheel_linux_up(_ev) -> None:
        class E:
            x, y, delta = _ev.x, _ev.y, 120

        _on_mousewheel(E())

    def _on_wheel_linux_down(_ev) -> None:
        class E:
            x, y, delta = _ev.x, _ev.y, -120

        _on_mousewheel(E())

    canvas.bind("<MouseWheel>", _on_mousewheel)
    canvas.bind("<Button-4>", _on_wheel_linux_up)
    canvas.bind("<Button-5>", _on_wheel_linux_down)
    canvas.bind("<Enter>", lambda _e: canvas.focus_set())

    def do_done() -> None:
        result[0] = (selected.copy(), base.copy())
        try:
            root.grab_release()
        except Exception:
            pass
        root.destroy()

    def do_cancel() -> None:
        cancelled[0] = True
        selected[:] = 0
        result[0] = (selected.copy(), base.copy())
        try:
            root.grab_release()
        except Exception:
            pass
        root.destroy()

    def do_cut() -> None:
        nonlocal mode, pending_pt
        mode = "pick" if mode == "cut" else "cut"
        pending_pt = None
        _refresh()

    def do_add() -> None:
        nonlocal mode, pending_pt
        mode = "pick" if mode == "add" else "add"
        pending_pt = None
        _refresh()

    def do_clear_strokes() -> None:
        nonlocal lab, fg, edges
        cuts.clear()
        adds.clear()
        pending_pt = None
        lab, fg = recompute_labels()
        edges = compute_edges()
        selected[(base == 127)] = 0
        _refresh()

    def do_undo() -> None:
        nonlocal lab, fg, edges
        if not undo_stack:
            return
        kind, payload = undo_stack.pop()
        if kind == "stroke":
            if payload == "cut" and cuts:
                cuts.pop()
            elif payload == "add" and adds:
                adds.pop()
            pending_pt = None
            lab, fg = recompute_labels()
            edges = compute_edges()
            selected[(base == 127)] = 0
        elif kind == "sel":
            rr, cc, prev_vals = payload
            selected[rr, cc] = prev_vals
        _refresh()

    def do_reset_sel() -> None:
        selected[:] = 0
        history.clear()
        _refresh()

    def do_open_bins_ui() -> None:
        """Close pick UI; pipeline opens 3-bin sketch, then returns here (like brain outline ↔ threshold)."""
        edit_bins_redirect[0] = True
        try:
            root.grab_release()
        except Exception:
            pass
        root.destroy()

    btn_add = _make_tool_btn(tool_card, "+", "A", do_add)
    btn_add.pack(padx=10, pady=(12, 6))
    btn_cut = _make_tool_btn(tool_card, "−", "C", do_cut)
    btn_cut.pack(padx=10, pady=6)
    btn_undo = _make_tool_btn(tool_card, "↶", "U", do_undo)
    btn_undo.pack(padx=10, pady=6)
    btn_clear = _make_tool_btn(tool_card, "✕", "X", do_clear_strokes)
    btn_clear.pack(padx=10, pady=6)
    btn_reset = _make_tool_btn(tool_card, "↻", "R", do_reset_sel)
    btn_reset.pack(padx=10, pady=(6, 6 if allow_open_bins_ui else 12))
    if allow_open_bins_ui:
        btn_bins = _make_tool_btn(tool_card, "≡", "B", do_open_bins_ui)
        btn_bins.pack(padx=10, pady=(6, 12))

    done_w, skip_w, action_h = 100, 118, 30
    skip_btn = create_secondary_button(
        footer, text="Skip Image", command=do_cancel, width=skip_w, height=action_h
    )
    skip_btn.grid(row=0, column=1, sticky="w", padx=(10, 0))
    done_btn = create_primary_button(
        footer, text="Done", command=do_done, width=done_w, height=action_h
    )
    done_btn.grid(row=0, column=4, sticky="e")

    total_w = viewport_w + LBAR_W + GRID_PAD_X
    total_h = HDR_H + initial_canvas_h + SCROLL_X_H + FTR_ROW + GRID_PAD_Y + 12
    total_w = min(total_w, int(screen_w * 0.98))
    total_h = min(total_h, int(screen_h * 0.94))
    root.geometry(f"{total_w}x{total_h}")
    root.update_idletasks()
    rw = root.winfo_width()
    rh = root.winfo_height()
    x0 = max(0, (screen_w - rw) // 2)
    y0 = max(0, (screen_h - rh) // 2)
    root.geometry(f"{rw}x{rh}+{x0}+{y0}")

    def on_press(ev) -> None:
        nonlocal pending_pt, lab, fg, edges
        if space_held[0]:
            pan_drag[0] = True
            canvas.scan_mark(int(ev.x), int(ev.y))
            return
        xy = _canvas_xy(ev)
        if xy is None:
            return
        x, y = xy
        if mode in ("cut", "add"):
            if pending_pt is None:
                pending_pt = (x, y)
            else:
                seg = (pending_pt, (x, y))
                pending_pt = None
                if mode == "cut":
                    cuts.append(seg)
                    undo_stack.append(("stroke", "cut"))
                else:
                    adds.append(seg)
                    undo_stack.append(("stroke", "add"))
                lab, fg = recompute_labels()
                edges = compute_edges()
                selected[(base == 127)] = 0
            _refresh()
            return
        idx = int(lab[y, x])
        if idx <= 0:
            return
        mask = (lab == idx)
        rr, cc = np.where(mask)
        if rr.size == 0:
            return
        prev_vals = selected[rr, cc].copy()
        if np.any(prev_vals):
            selected[rr, cc] = 0
        else:
            selected[rr, cc] = 1
        undo_stack.append(("sel", (rr, cc, prev_vals)))
        if not np.any(prev_vals):
            history.append(idx)
        _refresh()

    def on_motion(ev) -> None:
        if pan_drag[0]:
            canvas.scan_dragto(int(ev.x), int(ev.y), gain=1)

    def on_release(_ev) -> None:
        pan_drag[0] = False

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_motion)
    canvas.bind("<ButtonRelease-1>", on_release)

    def on_key(ev) -> None:
        k = (ev.keysym or "").lower()
        if k in ("return", "kp_enter"):
            do_done()
            return
        if k == "escape":
            do_cancel()
            return
        if k == "c":
            do_cut()
            return
        if k == "a":
            do_add()
            return
        if k == "u":
            do_undo()
            return
        if k == "r":
            do_reset_sel()
            return
        if k == "x":
            do_clear_strokes()
            return
        if k == "b" and allow_open_bins_ui:
            do_open_bins_ui()
            return

    def on_space_press(_ev) -> None:
        space_held[0] = True
        try:
            canvas.configure(cursor="fleur")
        except Exception:
            pass

    def on_space_release(_ev) -> None:
        space_held[0] = False
        pan_drag[0] = False
        try:
            canvas.configure(cursor="")
        except Exception:
            pass

    root.bind("<Key>", on_key)
    root.bind("<KeyPress-space>", on_space_press)
    root.bind("<KeyRelease-space>", on_space_release)

    _sync_badge_and_tools()
    _refresh()
    root.wait_window(root)

    if edit_bins_redirect[0]:
        return "edit_bins"

    if cancelled[0] and result[0] is not None:
        out_sel, out_base = result[0]
        out_sel[:] = 0
        return out_sel, out_base
    if result[0] is not None:
        return result[0]
    return selected, base

def pick_hippocampus_and_split_by_midline(
    *,
    sketch_u8_roi: np.ndarray,
    bg_roi_rgb: np.ndarray,
    midline_params: Dict[str, Any],
    roi_x0: int,
    roi_y0: int,
    window: str = "Pick hippocampus components (green)",
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Literal["edit_bins"]]:
    """
    1) Let user click connected components on sketch within ROI (returns sel_roi, sketch_after).
    2) Split sel_roi into left/right hemispheres using midline points (from full-image coords).
    Returns: (left_roi_sel_u8, right_roi_sel_u8, sketch_after_u8), all in ROI coords,
    or the literal ``\"edit_bins\"`` if the user chose to adjust 3-bin thresholds (pipeline should
    run ``run_bins_ui`` and call this again).
    """

    picked = select_components_on_background(
        sketch_u8_roi,
        bg_roi_rgb,
        window=window,
        allow_open_bins_ui=True,
    )
    if picked == "edit_bins":
        return "edit_bins"
    sel_roi_u8, sketch_after = picked
    sel = (sel_roi_u8 > 0)

    # Parse and shift midline points into ROI coords
    pts = np.array(json.loads(midline_params["midline_pts"]), dtype=np.float32)  # (x,y) in full image
    pts[:, 0] -= float(roi_x0)
    pts[:, 1] -= float(roi_y0)

    # Sort by y for stable interpolation
    order = np.argsort(pts[:, 1])
    pts = pts[order]

    h, w = sel.shape
    Y, X = np.indices((h, w))

    mid_y = pts[:, 1]
    mid_x = pts[:, 0]

    # Avoid edge cases (duplicate y)
    # If duplicates exist, make y strictly increasing by small jitter
    dy = np.diff(mid_y)
    if np.any(dy == 0):
        # stable: add tiny increments where needed
        for i in range(1, len(mid_y)):
            if mid_y[i] <= mid_y[i - 1]:
                mid_y[i] = mid_y[i - 1] + 1e-3

    x_mid = np.interp(Y[:, 0].astype(np.float32), mid_y, mid_x, left=mid_x[0], right=mid_x[-1])
    x_mid_map = x_mid[:, None]  # (h,1) broadcasts to (h,w)

    left = sel & (X < x_mid_map)
    right = sel & (X >= x_mid_map)

    left_u8 = left.astype(np.uint8)
    right_u8 = right.astype(np.uint8)
    return left_u8, right_u8, sketch_after