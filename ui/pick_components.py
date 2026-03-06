import numpy as np
import cv2
from typing import Tuple, Dict, Any
import json
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


# Helper: select components on a background RGB image (Tk UI, same style as brain_outline)
def select_components_on_background(
    sketch_u8_roi: np.ndarray,
    bg_rgb_roi: np.ndarray,
    *,
    window: str,
    init_selected: np.ndarray | None = None,
    init_cuts=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Click to toggle connected components. Tk UI: canvas left, instructions/buttons in right panel (no text on image)."""

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
        return cv2.Canny(base, 50, 150)

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

    # --- Tk layout (top controls, canvas below) ---
    parent = tk._default_root
    if parent is None:
        root = tk.Tk()
    else:
        root = tk.Toplevel(parent)
        root.transient(parent)
    root.title(window)
    root.update_idletasks()
    try:
        root.grab_set()
    except Exception:
        pass

    frm = ttk.Frame(root, padding=8)
    frm.pack(fill="both", expand=True)
    frm.columnconfigure(0, weight=1)
    frm.rowconfigure(1, weight=1)

    ctrl = ttk.Frame(frm)
    ctrl.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    ctrl.columnconfigure(0, weight=1)

    frm_canvas = ttk.Frame(frm)
    frm_canvas.grid(row=1, column=0, sticky="nsew")
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

    top_info = ttk.Frame(ctrl)
    top_info.grid(row=0, column=0, sticky="ew")
    top_info.columnconfigure(1, weight=1)

    lbl_title = ttk.Label(top_info, text="Pick hippocampus (green)", font=("TkDefaultFont", 14, "bold"))
    lbl_title.grid(row=0, column=0, sticky="w", padx=(0, 14))

    mode_var = tk.StringVar(value="MODE: PICK")
    lbl_mode = ttk.Label(top_info, textvariable=mode_var, font=("TkDefaultFont", 11, "bold"))
    lbl_mode.grid(row=0, column=1, sticky="w", padx=(0, 14))

    status_var = tk.StringVar(value="")
    lbl_status = ttk.Label(top_info, textvariable=status_var, justify="left")
    lbl_status.grid(row=0, column=2, sticky="w")

    lbl_hint = ttk.Label(
        ctrl,
        text="PICK: click regions to toggle selection.  C: CUT  A: ADD  X: clear strokes  U: undo  R: reset sel  Enter: done  Esc: cancel",
        justify="left",
    )
    lbl_hint.grid(row=1, column=0, sticky="w", pady=(6, 8))

    result: list[tuple[np.ndarray, np.ndarray] | None] = [None]
    cancelled = [False]

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
        mode_var.set("MODE: CUT" if mode == "cut" else "MODE: PICK")
        _refresh()

    def do_add() -> None:
        nonlocal mode, pending_pt
        mode = "pick" if mode == "add" else "add"
        pending_pt = None
        mode_var.set("MODE: ADD" if mode == "add" else "MODE: PICK")
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

    btns = ttk.Frame(ctrl)
    btns.grid(row=2, column=0, sticky="ew")
    for i in range(7):
        btns.columnconfigure(i, weight=1)
    ttk.Button(btns, text="Done (Enter)", command=do_done).grid(row=0, column=0, sticky="ew", padx=(0, 6))
    ttk.Button(btns, text="Cancel (Esc)", command=do_cancel).grid(row=0, column=1, sticky="ew", padx=(0, 6))
    ttk.Button(btns, text="Cut (C)", command=do_cut).grid(row=0, column=2, sticky="ew", padx=(0, 6))
    ttk.Button(btns, text="Add (A)", command=do_add).grid(row=0, column=3, sticky="ew", padx=(0, 6))
    ttk.Button(btns, text="Clear strokes (X)", command=do_clear_strokes).grid(row=0, column=4, sticky="ew", padx=(0, 6))
    ttk.Button(btns, text="Undo (U)", command=do_undo).grid(row=0, column=5, sticky="ew", padx=(0, 6))
    ttk.Button(btns, text="Reset sel (R)", command=do_reset_sel).grid(row=0, column=6, sticky="ew")

    h, w = base.shape[:2]
    try:
        screen_w = int(root.winfo_screenwidth())
        screen_h = int(root.winfo_screenheight())
    except Exception:
        screen_w, screen_h = 1400, 900

    max_canvas_w = int(screen_w * 0.90)
    max_canvas_h = int(screen_h * 0.75)
    disp_scale = min(1.0, max_canvas_w / float(w), max_canvas_h / float(h))
    disp_w = int(round(w * disp_scale))
    disp_h = int(round(h * disp_scale))
    canvas.configure(width=min(disp_w, max_canvas_w), height=min(disp_h, max_canvas_h))

    # Tight window: controls on top + image below, no huge empty side panel
    root.update_idletasks()
    ctrl_h = int(ctrl.winfo_reqheight())
    total_w = min(max(disp_w + 24, 700), int(screen_w * 0.95))
    total_h = min(max(ctrl_h + disp_h + 40, 500), int(screen_h * 0.92))
    root.geometry(f"{total_w}x{total_h}")

    tk_img_ref: dict = {}
    canvas_img_id: list = []  # mutable to store id after first create

    def _refresh() -> None:
        disp = redraw()
        if disp_scale < 1.0:
            disp = cv2.resize(disp, (disp_w, disp_h))
        n_sel = int((selected > 0).sum())
        status_var.set(f"Selected: {n_sel} px")
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tk_img = ImageTk.PhotoImage(pil, master=root)
        tk_img_ref["img"] = tk_img
        if not canvas_img_id:
            canvas_img_id.append(canvas.create_image(0, 0, anchor="nw", image=tk_img))
        else:
            canvas.itemconfigure(canvas_img_id[0], image=tk_img)
        canvas.configure(scrollregion=(0, 0, disp_w, disp_h))

    def _canvas_xy(ev) -> tuple[int, int] | None:
        cx, cy = canvas.canvasx(ev.x), canvas.canvasy(ev.y)
        if cx < 0 or cy < 0 or cx >= disp_w or cy >= disp_h:
            return None
        ix = int(round(cx / disp_scale))
        iy = int(round(cy / disp_scale))
        return (min(max(ix, 0), w - 1), min(max(iy, 0), h - 1))

    def on_click(ev) -> None:
        nonlocal pending_pt, lab, fg, edges
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

    canvas.bind("<Button-1>", on_click)

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
        if k == "x":
            do_clear_strokes()
            return
        if k == "u":
            do_undo()
            return
        if k == "r":
            do_reset_sel()
            return

    root.bind("<Key>", on_key)
    _refresh()
    root.wait_window(root)

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1) Let user click connected components on sketch within ROI (returns sel_roi, sketch_after).
    2) Split sel_roi into left/right hemispheres using midline points (from full-image coords).
    Returns: (left_roi_sel_u8, right_roi_sel_u8, sketch_after_u8), all in ROI coords.
    """

    sel_roi_u8, sketch_after = select_components_on_background(
        sketch_u8_roi, bg_roi_rgb, window=window
    )
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