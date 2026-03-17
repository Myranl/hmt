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


def _get_contour_points(mask_u8: np.ndarray) -> np.ndarray | None:
    """Get main contour as (N, 2) array of (x, y). Returns None if no contour."""
    cnts, _ = cv2.findContours(
        (mask_u8 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    return cnt.reshape(-1, 2).astype(np.int32)


def _snap_to_contour(pts: np.ndarray, x: int, y: int, max_dist: float = 35) -> int | None:
    """Return index of contour point nearest to (x,y), or None if beyond max_dist."""
    if pts is None or len(pts) == 0:
        return None
    d2 = (pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2
    i = int(np.argmin(d2))
    if d2[i] <= max_dist * max_dist:
        return i
    return None


def _arc_length(pts: np.ndarray, i: int, j: int) -> float:
    """Length of contour from index i to j (inclusive, wrapping)."""
    n = len(pts)
    if n == 0:
        return 0.0
    total = 0.0
    k = i
    while k != j:
        nxt = (k + 1) % n
        total += float(np.hypot(pts[nxt, 0] - pts[k, 0], pts[nxt, 1] - pts[k, 1]))
        k = nxt
    return total


def _arc_indices(pts: np.ndarray, i: int, j: int, keep_shorter: bool) -> np.ndarray:
    """Indices from i to j along contour (wrapping). If keep_shorter, take the shorter arc."""
    n = len(pts)
    len_ij = _arc_length(pts, i, j)
    len_ji = _arc_length(pts, j, i)
    if keep_shorter and len_ji < len_ij:
        i, j = j, i
        len_ij = len_ji
    out = []
    k = i
    while True:
        out.append(k)
        if k == j:
            break
        k = (k + 1) % n
    return np.array(out, dtype=np.int32)


def _apply_break(pts: np.ndarray, idx1: int, idx2: int) -> np.ndarray:
    """New contour: keep longer arc (main contour), replace shorter arc (bulge) with chord. Returns contour (N,2)."""
    n = len(pts)
    len_12 = _arc_length(pts, idx1, idx2)
    len_21 = _arc_length(pts, idx2, idx1)
    if len_12 < len_21:
        idx1, idx2 = idx2, idx1
    # keep arc from idx1 to idx2 (the longer one), chord from idx2 to idx1
    keep = _arc_indices(pts, idx1, idx2, keep_shorter=False)
    arc_pts = pts[keep]
    p1 = pts[idx1].astype(np.float32)
    p2 = pts[idx2].astype(np.float32)
    chord = np.array([p2, p1], dtype=np.int32)
    new_contour = np.vstack([arc_pts, chord])
    return new_contour


def _bezier_sample(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, n: int = 50) -> np.ndarray:
    """Quadratic Bezier from p0 to p2 with control p1. Returns (n+1, 2) int32."""
    t = np.linspace(0, 1, n + 1)
    # B(t) = (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
    y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
    return np.column_stack([x, y]).astype(np.int32)


def edit_contour_ui(
    brain_mask: np.ndarray,
    original_image_rgb: np.ndarray,
    *,
    window: str = "Contour Editor",
) -> tuple[np.ndarray, dict]:
    """UI for editing non-complete brain contour: Break (cut arc) and Missing (bridge gap)."""

    h, w = brain_mask.shape[:2]
    img_bgr = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR)

    # Working mask (uint8 0/255) — break does NOT modify it; only "missing" (bridge) does
    mask_u8 = (brain_mask.astype(np.uint8)) * 255
    undo_stack: list[np.ndarray] = []

    # Break = subset of contour points excluded from counted perimeter. Stored as (i,j): shorter arc i→j is "break".
    # Mask and area stay unchanged; only which arc counts toward perimeter changes.
    break_arcs: list[tuple[int, int]] = []

    mode = "break"  # "break" | "missing"
    pending_i: int | None = None
    pending_j: int | None = None
    bridge_control: tuple[float, float] | None = None  # (x,y) for Missing curve

    result_container: list[np.ndarray] = [brain_mask.copy()]
    result_params: dict = {"accepted": False, "edited": False, "correction_type": "none"}

    setup_theme()
    root = ctk.CTk()
    root.title(window)
    root.minsize(900, 620)
    root.geometry("1200x700")
    root.configure(fg_color="white")
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)

    base_font = get_base_font()
    small_font = get_small_muted_font()

    # Left: image area (card) — canvas fits inside, scroll when image is large
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

    # Right: controls (card) — same style as folder selection
    ctrl = create_card_frame(root)
    ctrl.grid(row=0, column=1, sticky="ns", padx=(0, 18), pady=18)
    ctrl.grid_propagate(False)
    ctrl.configure(width=320)

    ctk.CTkLabel(ctrl, text="Contour Editor", font=ctk.CTkFont(size=16, weight="bold")).grid(
        row=0, column=0, sticky="w", padx=14, pady=(14, 6))
    hint_text = (
        "Break: click two points — the arc between them is marked gray and excluded from perimeter. "
        "Mask and area stay unchanged.\n\n"
        "Missing: click start and end of gap, then Apply to add a bridge."
    )
    create_status_label(ctrl, text=hint_text, wraplength=280).grid(
        row=1, column=0, sticky="w", padx=14, pady=(0, 10))

    mode_var = tk.StringVar(value="MODE: Break")
    ctk.CTkLabel(ctrl, textvariable=mode_var, font=ctk.CTkFont(size=12, weight="bold")).grid(
        row=2, column=0, sticky="w", padx=14, pady=(0, 4))
    status_var = tk.StringVar(value="")
    status_lbl = create_status_label(ctrl, textvariable=status_var, wraplength=280)
    status_lbl.grid(row=3, column=0, sticky="w", padx=14, pady=(0, 10))

    def set_mode(m: str) -> None:
        nonlocal mode, pending_i, pending_j, bridge_control
        mode = m
        pending_i = pending_j = None
        bridge_control = None
        mode_var.set("MODE: Break" if m == "break" else "MODE: Missing")
        # Highlight active mode button (muted when selected)
        _sel, _unsel = ("gray70", "gray35"), ("#3B8E3B", "#2d6b2d")
        if m == "break":
            btn_break.configure(fg_color=_sel)
            btn_missing.configure(fg_color=_unsel)
        else:
            btn_missing.configure(fg_color=_sel)
            btn_break.configure(fg_color=_unsel)
        _refresh()

    def push_undo() -> None:
        undo_stack.append(mask_u8.copy())

    def do_undo() -> None:
        nonlocal mask_u8, pending_i, pending_j, bridge_control
        if break_arcs:
            break_arcs.pop()
            pending_i = pending_j = None
            _refresh()
            return
        if not undo_stack:
            return
        mask_u8 = undo_stack.pop()
        pending_i = pending_j = None
        bridge_control = None
        _refresh()

    def apply_break() -> None:
        """Break: mark the arc between the two points as excluded from the counted perimeter.
        Mask and area are NOT changed — only which contour segment is "break" (gray, not counted).
        """
        nonlocal pending_i, pending_j
        pts = _get_contour_points(mask_u8)
        if pts is None or pending_i is None or pending_j is None:
            return
        if pending_i == pending_j:
            pending_i = pending_j = None
            _refresh()
            return

        # Store the shorter arc (i,j) so it is excluded from perimeter; mask unchanged
        len_ij = _arc_length(pts, pending_i, pending_j)
        len_ji = _arc_length(pts, pending_j, pending_i)
        i, j = (pending_i, pending_j) if len_ij <= len_ji else (pending_j, pending_i)
        break_arcs.append((i, j))
        result_params["edited"] = True
        result_params["correction_type"] = "break"
        pending_i = pending_j = None
        _refresh()

    def apply_bridge() -> None:
        nonlocal mask_u8, pending_i, pending_j, bridge_control
        break_arcs.clear()  # contour changes, so stored break indices would be invalid
        pts = _get_contour_points(mask_u8)
        if pts is None or pending_i is None or pending_j is None:
            return
        if pending_i == pending_j:
            pending_i = pending_j = None
            bridge_control = None
            _refresh()
            return
        p1 = pts[pending_i].astype(np.float64)
        p2 = pts[pending_j].astype(np.float64)
        if bridge_control is None:
            mid = (p1 + p2) / 2
            centroid = pts.mean(axis=0)
            out_dir = mid - centroid
            norm = np.hypot(out_dir[0], out_dir[1]) or 1.0
            out_dir /= norm
            ctrl_pt = mid + 40 * out_dir
        else:
            ctrl_pt = np.array(bridge_control, dtype=np.float64)
        curve = _bezier_sample(p1, ctrl_pt, p2, n=60)
        # Polygon: contour arc from pending_j to pending_i (the existing boundary) + curve from p1 to p2
        keep = _arc_indices(pts, pending_j, pending_i, keep_shorter=True)
        arc_pts = pts[keep]
        poly = np.vstack([arc_pts, curve])
        push_undo()
        cap = np.zeros_like(mask_u8)
        cv2.fillPoly(cap, [poly.astype(np.int32)], 255)
        mask_u8 = cv2.bitwise_or(mask_u8, cap)
        pending_i = pending_j = None
        bridge_control = None
        result_params["edited"] = True
        result_params["correction_type"] = "bridge"
        _refresh()

    mode_btns = ctk.CTkFrame(ctrl, fg_color="transparent")
    mode_btns.grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 8))
    mode_btns.columnconfigure(0, weight=1)
    mode_btns.columnconfigure(1, weight=1)
    btn_break = ctk.CTkButton(mode_btns, text="Break", command=lambda: set_mode("break"), width=100)
    btn_break.grid(row=0, column=0, padx=(0, 6))
    btn_missing = ctk.CTkButton(mode_btns, text="Missing", command=lambda: set_mode("missing"), width=100)
    btn_missing.grid(row=0, column=1)

    create_secondary_button(ctrl, text="Undo", command=do_undo).grid(row=5, column=0, sticky="ew", padx=14, pady=(0, 4))
    create_secondary_button(ctrl, text="Apply break", command=apply_break).grid(row=6, column=0, sticky="ew", padx=14, pady=(0, 2))
    create_secondary_button(ctrl, text="Apply bridge", command=apply_bridge).grid(row=7, column=0, sticky="ew", padx=14, pady=(0, 10))

    def do_accept() -> None:
        result_container[0] = (mask_u8 > 0).astype(brain_mask.dtype)
        result_params["accepted"] = True
        root.destroy()

    def do_skip() -> None:
        result_params["accepted"] = True
        result_params["edited"] = False
        result_params["correction_type"] = "none"
        root.destroy()

    bar = ctk.CTkFrame(ctrl, fg_color="transparent")
    bar.grid(row=8, column=0, sticky="ew", padx=14, pady=(4, 14))
    bar.columnconfigure(0, weight=1)
    create_secondary_button(bar, text="Skip (keep original)", command=do_skip).grid(row=0, column=0, sticky="w")
    create_primary_button(bar, text="Accept (continue)", command=do_accept).grid(row=0, column=1, sticky="e")

    def _get_canvas_size() -> tuple[int, int]:
        cw = canvas_holder.winfo_width() or 800
        ch = canvas_holder.winfo_height() or 600
        if cw < 200:
            cw = 800
        if ch < 200:
            ch = 600
        return (cw, ch)

    disp_scale = 1.0
    disp_w = w
    disp_h = h
    tk_img_ref: dict = {}
    canvas_img_id: list = []

    def _refresh() -> None:
        nonlocal disp_scale, disp_w, disp_h
        cw, ch = _get_canvas_size()
        disp_scale = min(1.0, cw / float(w), ch / float(h))
        disp_w = int(round(w * disp_scale))
        disp_h = int(round(h * disp_scale))
        canvas.configure(width=disp_w, height=disp_h)

        pts = _get_contour_points(mask_u8)
        vis = img_bgr.copy()
        # Dim background outside mask
        outside = (mask_u8 == 0)
        vis[outside] = (vis[outside] * 0.5 + np.array([20, 20, 20])).astype(np.uint8)
        # Contour: green = counted perimeter, gray = break (excluded from perimeter). Mask/area unchanged.
        if pts is not None:
            n_pts = len(pts)
            excluded: set[int] = set()
            for (i, j) in break_arcs:
                idx = _arc_indices(pts, i, j, keep_shorter=True)
                excluded.update(idx.tolist())
            # Preview: show pending arc as gray too (visual only; not in break_arcs yet)
            if mode == "break" and pending_i is not None and pending_j is not None:
                len_ij = _arc_length(pts, pending_i, pending_j)
                len_ji = _arc_length(pts, pending_j, pending_i)
                pending_excl = _arc_indices(pts, pending_i, pending_j, keep_shorter=(len_ji < len_ij))
                excluded.update(pending_excl.tolist())
            # Build runs of consecutive same-status indices (wrapping)
            if excluded and n_pts > 0:
                arr = [i in excluded for i in range(n_pts)]
                runs: list[tuple[list[int], bool]] = []
                used = [False] * n_pts
                for start in range(n_pts):
                    if used[start]:
                        continue
                    is_excl = arr[start]
                    run = [start]
                    used[start] = True
                    j = start
                    while True:
                        nxt = (j + 1) % n_pts
                        if nxt == start or arr[nxt] != is_excl:
                            break
                        run.append(nxt)
                        used[nxt] = True
                        j = nxt
                    runs.append((run, is_excl))
                for run, is_excl in runs:
                    arc_pts = pts[run]
                    color = (128, 128, 128) if is_excl else (0, 255, 0)
                    thick = 4 if is_excl else 2
                    cv2.polylines(vis, [arc_pts], False, color, thick)
                if mode == "break" and pending_i is not None and pending_j is not None:
                    cv2.circle(vis, (int(pts[pending_i, 0]), int(pts[pending_i, 1])), 10, (128, 128, 128), -1)
                    cv2.circle(vis, (int(pts[pending_j, 0]), int(pts[pending_j, 1])), 10, (128, 128, 128), -1)
            else:
                cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
                if mode == "break" and pending_i is not None:
                    cv2.circle(vis, (int(pts[pending_i, 0]), int(pts[pending_i, 1])), 8, (0, 255, 255), 2)
        # Pending Missing: endpoints + curve preview
        if mode == "missing" and pts is not None and pending_i is not None:
            cv2.circle(vis, (int(pts[pending_i, 0]), int(pts[pending_i, 1])), 8, (0, 255, 255), 2)
            if pending_j is not None:
                cv2.circle(vis, (int(pts[pending_j, 0]), int(pts[pending_j, 1])), 8, (0, 255, 255), 2)
                p1 = pts[pending_i].astype(np.float64)
                p2 = pts[pending_j].astype(np.float64)
                if bridge_control is None:
                    mid = (p1 + p2) / 2
                    centroid = pts.mean(axis=0)
                    out_dir = mid - centroid
                    norm = np.hypot(out_dir[0], out_dir[1]) or 1.0
                    out_dir /= norm
                    ctrl_pt = mid + 40 * out_dir
                else:
                    ctrl_pt = np.array(bridge_control, dtype=np.float64)
                curve = _bezier_sample(p1, ctrl_pt, p2, n=50)
                cv2.polylines(vis, [curve], False, (0, 255, 255), 2)
                cv2.circle(vis, (int(ctrl_pt[0]), int(ctrl_pt[1])), 6, (255, 255, 0), -1)
        # Resize for display
        if disp_scale < 1.0:
            vis = cv2.resize(vis, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        area = int((mask_u8 > 0).sum())
        perim_full = 0.0
        perim_break = 0.0
        if pts is not None and len(pts) > 1:
            perim_full = float(cv2.arcLength(pts.astype(np.float32), True))
            for (i, j) in break_arcs:
                len_ij = _arc_length(pts, i, j)
                len_ji = _arc_length(pts, j, i)
                perim_break += min(len_ij, len_ji)
        perim_active = perim_full - perim_break
        if perim_break > 0:
            status_var.set(f"Area: {area} px  Perim: {perim_active:.1f} (break: {perim_break:.1f})")
        else:
            status_var.set(f"Area: {area} px  Perim: {perim_active:.1f}")
        pil = Image.fromarray(vis_rgb)
        tk_img = ImageTk.PhotoImage(pil, master=canvas)
        tk_img_ref["img"] = tk_img
        if not canvas_img_id:
            canvas_img_id.append(canvas.create_image(0, 0, anchor="nw", image=tk_img))
        else:
            canvas.itemconfigure(canvas_img_id[0], image=tk_img)
        canvas.configure(scrollregion=(0, 0, disp_w, disp_h))

    def _canvas_to_img(ev) -> tuple[int, int] | None:
        cx = canvas.canvasx(ev.x)
        cy = canvas.canvasy(ev.y)
        if cx < 0 or cy < 0 or cx >= disp_w or cy >= disp_h:
            return None
        ix = int(round(cx / disp_scale))
        iy = int(round(cy / disp_scale))
        return (max(0, min(ix, w - 1)), max(0, min(iy, h - 1)))

    def on_click(ev) -> None:
        nonlocal pending_i, pending_j
        xy = _canvas_to_img(ev)
        if xy is None:
            return
        ix, iy = xy
        pts = _get_contour_points(mask_u8)
        idx = _snap_to_contour(pts, ix, iy) if pts is not None else None
        if idx is None:
            return
        if pending_i is None:
            pending_i = idx
            status_var.set("Click second point on contour")
        else:
            pending_j = idx
            if mode == "break":
                apply_break()
            else:
                status_var.set("Adjust curve (optional) and click Apply bridge")
        _refresh()

    def on_right(ev) -> None:
        nonlocal pending_i, pending_j, bridge_control
        pending_i = pending_j = None
        bridge_control = None
        status_var.set("")
        _refresh()

    canvas.bind("<Button-3>", on_right)

    def on_motion(ev) -> None:
        if mode != "missing" or pending_i is None or pending_j is None:
            return
        xy = _canvas_to_img(ev)
        if xy is None:
            return
        nonlocal bridge_control
        bridge_control = (float(xy[0]), float(xy[1]))
        _refresh()

    canvas.bind("<Button-1>", on_click)
    canvas.bind("<B1-Motion>", on_motion)

    def _on_holder_configure(_ev) -> None:
        _refresh()

    canvas_holder.bind("<Configure>", _on_holder_configure)

    set_mode("break")
    root.update_idletasks()
    _refresh()
    root.wait_window()

    return result_container[0], result_params
