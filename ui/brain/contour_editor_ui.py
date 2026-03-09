import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


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

    # Working mask (uint8 0/255)
    mask_u8 = (brain_mask.astype(np.uint8)) * 255
    undo_stack: list[np.ndarray] = []

    mode = "break"  # "break" | "missing"
    pending_i: int | None = None
    pending_j: int | None = None
    bridge_control: tuple[float, float] | None = None  # (x,y) for Missing curve

    result_container: list[np.ndarray] = [brain_mask.copy()]
    result_params: dict = {"accepted": False, "edited": False, "correction_type": "none"}

    root = tk.Tk()
    root.title(window)
    root.minsize(800, 600)
    root.geometry("1100x750")

    frm = ttk.Frame(root, padding=8)
    frm.pack(fill="both", expand=True)
    frm.columnconfigure(0, weight=1)
    frm.rowconfigure(0, weight=1)

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

    ctrl = ttk.Frame(frm, width=280)
    ctrl.grid(row=0, column=1, sticky="ns", padx=(12, 0))
    ctrl.grid_propagate(False)

    ttk.Label(ctrl, text="Contour Editor", font=("TkDefaultFont", 14, "bold")).grid(
        row=0, column=0, sticky="w", pady=(0, 6))
    lbl_hint = ttk.Label(
        ctrl,
        text="Break: click two points on contour to cut off that arc.\nMissing: click start and end of gap, then Apply to add a bridge.",
        justify="left",
    )
    lbl_hint.grid(row=1, column=0, sticky="w", pady=(0, 8))

    mode_var = tk.StringVar(value="MODE: Break")
    ttk.Label(ctrl, textvariable=mode_var, font=("TkDefaultFont", 11, "bold")).grid(
        row=2, column=0, sticky="w", pady=(0, 4))
    status_var = tk.StringVar(value="")
    ttk.Label(ctrl, textvariable=status_var, justify="left").grid(
        row=3, column=0, sticky="w", pady=(0, 8))

    def set_mode(m: str) -> None:
        nonlocal mode, pending_i, pending_j, bridge_control
        mode = m
        pending_i = pending_j = None
        bridge_control = None
        mode_var.set("MODE: Break" if m == "break" else "MODE: Missing")
        _refresh()

    def push_undo() -> None:
        undo_stack.append(mask_u8.copy())

    def do_undo() -> None:
        nonlocal mask_u8, pending_i, pending_j, bridge_control
        if not undo_stack:
            return
        mask_u8 = undo_stack.pop()
        pending_i = pending_j = None
        bridge_control = None
        _refresh()

    def apply_break() -> None:
        nonlocal mask_u8, pending_i, pending_j
        pts = _get_contour_points(mask_u8)
        if pts is None or pending_i is None or pending_j is None:
            return
        if pending_i == pending_j:
            pending_i = pending_j = None
            _refresh()
            return
        push_undo()
        new_contour = _apply_break(pts, pending_i, pending_j)
        mask_u8 = np.zeros_like(mask_u8)
        cv2.fillPoly(mask_u8, [new_contour], 255)
        pending_i = pending_j = None
        result_params["edited"] = True
        result_params["correction_type"] = "break"
        _refresh()

    def apply_bridge() -> None:
        nonlocal mask_u8, pending_i, pending_j, bridge_control
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

    btns = ttk.Frame(ctrl)
    btns.grid(row=4, column=0, sticky="ew", pady=(0, 8))
    ttk.Button(btns, text="Break", command=lambda: set_mode("break")).grid(row=0, column=0, padx=(0, 4))
    ttk.Button(btns, text="Missing", command=lambda: set_mode("missing")).grid(row=0, column=1)
    ttk.Button(ctrl, text="Undo", command=do_undo).grid(row=5, column=0, sticky="ew", pady=(0, 4))
    ttk.Button(ctrl, text="Apply break", command=apply_break).grid(row=6, column=0, sticky="ew", pady=(0, 2))
    ttk.Button(ctrl, text="Apply bridge", command=apply_bridge).grid(row=7, column=0, sticky="ew", pady=(0, 8))

    def do_accept() -> None:
        result_container[0] = (mask_u8 > 0).astype(brain_mask.dtype)
        result_params["accepted"] = True
        root.destroy()

    def do_skip() -> None:
        result_params["accepted"] = True
        result_params["edited"] = False
        result_params["correction_type"] = "none"
        root.destroy()

    ttk.Button(ctrl, text="Accept (continue)", command=do_accept).grid(row=8, column=0, sticky="ew", pady=(0, 4))
    ttk.Button(ctrl, text="Skip (keep original)", command=do_skip).grid(row=9, column=0, sticky="ew")

    max_canvas_w, max_canvas_h = 900, 700
    disp_scale = min(1.0, max_canvas_w / float(w), max_canvas_h / float(h))
    disp_w = int(round(w * disp_scale))
    disp_h = int(round(h * disp_scale))
    canvas.configure(width=min(disp_w, max_canvas_w), height=min(disp_h, max_canvas_h))
    tk_img_ref: dict = {}
    canvas_img_id: list = []

    def _refresh() -> None:
        pts = _get_contour_points(mask_u8)
        vis = img_bgr.copy()
        # Dim background outside mask
        outside = (mask_u8 == 0)
        vis[outside] = (vis[outside] * 0.5 + np.array([20, 20, 20])).astype(np.uint8)
        # Contour
        if pts is not None:
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        # Pending Break: gray arc + chord + two points
        if mode == "break" and pts is not None and pending_i is not None:
            cv2.circle(vis, (int(pts[pending_i, 0]), int(pts[pending_i, 1])), 8, (0, 255, 255), 2)
            if pending_j is not None:
                cv2.circle(vis, (int(pts[pending_j, 0]), int(pts[pending_j, 1])), 8, (0, 255, 255), 2)
                # Draw arc that will be removed (gray) and chord (yellow)
                len_12 = _arc_length(pts, pending_i, pending_j)
                len_21 = _arc_length(pts, pending_j, pending_i)
                remove_arc = _arc_indices(pts, pending_i, pending_j, keep_shorter=(len_21 < len_12))
                arc_pts = pts[remove_arc]
                cv2.polylines(vis, [arc_pts], False, (128, 128, 128), 3)
                cv2.line(vis, (int(pts[pending_i, 0]), int(pts[pending_i, 1])),
                         (int(pts[pending_j, 0]), int(pts[pending_j, 1])), (0, 255, 255), 2)
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
        perim = 0.0
        if pts is not None and len(pts) > 1:
            perim = float(cv2.arcLength(pts.astype(np.float32), True))
        status_var.set(f"Area: {area} px  Perim: {perim:.1f}")
        pil = Image.fromarray(vis_rgb)
        tk_img = ImageTk.PhotoImage(pil)
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
    # Update Bezier control point only while left button is pressed (drag),
    # so форма кривой не \"уплывает\", когда уводим мышь к кнопкам.
    canvas.bind("<B1-Motion>", on_motion)

    set_mode("break")
    _refresh()
    root.wait_window()

    return result_container[0], result_params
