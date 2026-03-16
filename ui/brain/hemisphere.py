from __future__ import annotations
import numpy as np
import cv2
import json
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1  # x0,y0,x1,y1


def _crop_with_pad(shape_hw: tuple[int, int], bbox: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    h, w = shape_hw
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    return x0, y0, x1, y1


def _pca_midline_from_mask(mask: np.ndarray, *, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Return 2 endpoints (xy float) of an auto midline in full-image coords.
    q controls robustness to missing chunks: uses q..(1-q) percentiles along major axis.
    """
    ys, xs = np.where(mask)
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)  # (N,2) xy

    c = pts.mean(axis=0)
    X = pts - c
    # covariance + eig
    C = (X.T @ X) / max(1, (X.shape[0] - 1))
    evals, evecs = np.linalg.eigh(C)  # ascending
    v1 = evecs[:, 1]  # major axis (largest eigenvalue)
    v2 = evecs[:, 0]  # minor axis (perpendicular), direction of midline

    # project onto major axis to find robust center position
    t = X @ v1
    lo = np.quantile(t, q)
    hi = np.quantile(t, 1.0 - q)
    t0 = 0.5 * (lo + hi)  # robust center along v1

    # anchor point on midline
    p0 = c + v1 * t0

    # choose line length: span across mask extent along v2 (robust)
    s = X @ v2
    s_lo = np.quantile(s, q)
    s_hi = np.quantile(s, 1.0 - q)
    # extend a bit beyond
    ext = 1.15
    a = p0 + v2 * (s_lo * ext)
    b = p0 + v2 * (s_hi * ext)
    return a, b


def midline_ui(
    img_rgb: np.ndarray,
    brain_mask: np.ndarray,
    *,
    pad: int = 50,
    overlay_alpha: float = 0.35,
    line_color_bgr: tuple[int, int, int] = (0, 0, 0),
    line_thickness: int = 5,
    window: str = "MIDLINE",
) -> dict | None:
    """UI: shows cropped image centered by brain_mask and lets user adjust midline by dragging endpoints.
    ENTER accept, R reset, ESC cancel.
    Returns dict with endpoints in full-image coords.
    """
    if brain_mask.dtype != np.bool_:
        brain_mask = brain_mask.astype(bool)

    bbox = _bbox_from_mask(brain_mask)
    if bbox is None:
        return None

    x0, y0, x1, y1 = _crop_with_pad(brain_mask.shape[:2], bbox, pad)
    roi = img_rgb[y0:y1, x0:x1].copy()
    mroi = brain_mask[y0:y1, x0:x1]

    # auto midline in full coords then convert to ROI coords
    a_full, b_full = _pca_midline_from_mask(brain_mask, q=0.05)
    a = a_full - np.array([x0, y0], dtype=np.float32)
    b = b_full - np.array([x0, y0], dtype=np.float32)

    # clamp endpoints into ROI bounds (just for display)
    h, w = mroi.shape

    def clamp(p):
        return np.array([np.clip(p[0], 0, w - 1), np.clip(p[1], 0, h - 1)], dtype=np.float32)

    a = clamp(a)
    b = clamp(b)

    # prepare overlay
    disp_base = roi.copy()
    if disp_base.ndim == 2:
        disp_base = cv2.cvtColor(disp_base, cv2.COLOR_GRAY2BGR)
    mask_vis = np.zeros_like(disp_base)
    mask_vis[mroi] = (0, 255, 0)
    disp_base = cv2.addWeighted(disp_base, 1.0, mask_vis, float(overlay_alpha), 0.0)

    state = {
        "a": a,
        "b": b,
        "drag": None,
        "poly_mode": True,
        "pts": [a.copy(), b.copy()],
    }

    HANDLE_R = 12

    def redraw() -> np.ndarray:
        d = disp_base.copy()
        if state["poly_mode"]:
            pts = [p.copy() for p in state["pts"]]
            pts.sort(key=lambda p: float(p[1]))
            if len(pts) >= 2:
                poly = np.round(np.stack(pts, axis=0)).astype(np.int32)
                cv2.polylines(d, [poly], isClosed=False, color=line_color_bgr, thickness=line_thickness, lineType=cv2.LINE_AA)
            for p in pts:
                pp = tuple(np.round(p).astype(int))
                # white center with black outline
                cv2.circle(d, pp, HANDLE_R + 2, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(d, pp, HANDLE_R, (255, 255, 255), -1, cv2.LINE_AA)
        else:
            aa = tuple(np.round(state["a"]).astype(int))
            bb = tuple(np.round(state["b"]).astype(int))
            cv2.line(d, aa, bb, line_color_bgr, line_thickness, cv2.LINE_AA)
            cv2.circle(d, aa, HANDLE_R + 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(d, aa, HANDLE_R, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(d, bb, HANDLE_R + 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(d, bb, HANDLE_R, (255, 255, 255), -1, cv2.LINE_AA)
        return d

    def pick_handle(x: int, y: int, p: np.ndarray, r: int | None = None) -> bool:
        if r is None:
            r = HANDLE_R + 4
        return (x - float(p[0])) ** 2 + (y - float(p[1])) ** 2 <= float(r * r)

    def _nearest_point_index(x: int, y: int) -> int | None:
        pts = state["pts"]
        if not pts:
            return None
        d2 = [float((x - p[0]) ** 2 + (y - p[1]) ** 2) for p in pts]
        j = int(np.argmin(d2))
        if d2[j] <= float((HANDLE_R + 4) ** 2):
            return j
        return None

    # --- Tk window ---
    parent = tk._default_root
    if parent is None:
        root = tk.Tk()
    else:
        root = tk.Toplevel(parent)
        root.transient(parent)
    try:
        root.grab_set()
    except Exception:
        pass
    root.title("Midline")

    frm = ttk.Frame(root, padding=8)
    frm.pack(fill="both", expand=True)
    frm.columnconfigure(0, weight=1)
    frm.rowconfigure(1, weight=1)

    ctrl = ttk.Frame(frm)
    ctrl.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    ctrl.columnconfigure(0, weight=1)

    top_info = ttk.Frame(ctrl)
    top_info.grid(row=0, column=0, sticky="ew")
    top_info.columnconfigure(1, weight=1)

    ttk.Label(top_info, text="Midline", font=("TkDefaultFont", 14, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 14))

    mode_var = tk.StringVar(value="MODE: POLYLINE")
    ttk.Label(top_info, textvariable=mode_var, font=("TkDefaultFont", 11, "bold")).grid(row=0, column=1, sticky="w", padx=(0, 14))

    status_var = tk.StringVar(value="")
    ttk.Label(top_info, textvariable=status_var, justify="left").grid(row=0, column=2, sticky="w")

    ttk.Label(
        ctrl,
        text="Drag endpoints/points. P: polyline  U: undo  R: reset  Enter: accept  Esc: cancel",
        justify="left",
    ).grid(row=1, column=0, sticky="w", pady=(6, 8))

    btns = ttk.Frame(ctrl)
    btns.grid(row=2, column=0, sticky="ew")
    for i in range(5):
        btns.columnconfigure(i, weight=1)

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

    root.update_idletasks()
    ctrl_h = int(ctrl.winfo_reqheight())
    total_w = min(max(disp_w + 24, 700), int(screen_w * 0.95))
    total_h = min(max(ctrl_h + disp_h + 40, 500), int(screen_h * 0.92))
    root.geometry(f"{total_w}x{total_h}")

    tk_img_ref: dict[str, ImageTk.PhotoImage] = {}
    canvas_img_id: list[int] = []
    result = {"accepted": False, "cancelled": False}
    undo_stack: list[tuple[bool, list[np.ndarray], np.ndarray, np.ndarray]] = []

    def _push_undo() -> None:
        undo_stack.append((bool(state["poly_mode"]), [p.copy() for p in state["pts"]], state["a"].copy(), state["b"].copy()))
        if len(undo_stack) > 50:
            undo_stack.pop(0)

    def _set_mode_label() -> None:
        mode_var.set("MODE: POLYLINE" if state["poly_mode"] else "MODE: LINE")

    def _refresh() -> None:
        disp = redraw()
        if disp_scale < 1.0:
            disp = cv2.resize(disp, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
        pts_n = len(state["pts"]) if state["poly_mode"] else 2
        status_var.set(f"points={pts_n}")
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tk_img = ImageTk.PhotoImage(pil, master=root)
        tk_img_ref["img"] = tk_img
        if not canvas_img_id:
            canvas_img_id.append(canvas.create_image(0, 0, anchor="nw", image=tk_img))
        else:
            canvas.itemconfigure(canvas_img_id[0], image=tk_img)
        canvas.configure(scrollregion=(0, 0, disp_w, disp_h))
        _set_mode_label()

    def _canvas_xy(ev) -> tuple[int, int] | None:
        cx = canvas.canvasx(ev.x)
        cy = canvas.canvasy(ev.y)
        if cx < 0 or cy < 0 or cx >= disp_w or cy >= disp_h:
            return None
        ix = int(round(cx / disp_scale)) if disp_scale > 0 else int(cx)
        iy = int(round(cy / disp_scale)) if disp_scale > 0 else int(cy)
        return (min(max(ix, 0), w - 1), min(max(iy, 0), h - 1))

    def on_press(ev) -> None:
        xy = _canvas_xy(ev)
        if xy is None:
            return
        x, y = xy
        if state["poly_mode"]:
            j = _nearest_point_index(x, y)
            if j is not None:
                _push_undo()
                state["drag"] = j
                return
            _push_undo()
            p = clamp(np.array([x, y], dtype=np.float32))
            state["pts"].append(p)
            state["drag"] = len(state["pts"]) - 1
            _refresh()
            return
        if pick_handle(x, y, state["a"]):
            _push_undo()
            state["drag"] = "a"
        elif pick_handle(x, y, state["b"]):
            _push_undo()
            state["drag"] = "b"

    def on_motion(ev) -> None:
        if state["drag"] is None:
            return
        xy = _canvas_xy(ev)
        if xy is None:
            return
        x, y = xy
        p = clamp(np.array([x, y], dtype=np.float32))
        if state["poly_mode"]:
            if isinstance(state["drag"], int):
                state["pts"][state["drag"]] = p
        else:
            state[state["drag"]] = p
        _refresh()

    def on_release(ev) -> None:
        state["drag"] = None

    def do_accept() -> None:
        result["accepted"] = True
        try:
            root.grab_release()
        except Exception:
            pass
        root.destroy()

    def do_cancel() -> None:
        result["cancelled"] = True
        try:
            root.grab_release()
        except Exception:
            pass
        root.destroy()

    def do_undo() -> None:
        if not undo_stack:
            return
        poly_mode_prev, pts_prev, a_prev, b_prev = undo_stack.pop()
        state["poly_mode"] = poly_mode_prev
        state["pts"] = [p.copy() for p in pts_prev]
        state["a"] = a_prev.copy()
        state["b"] = b_prev.copy()
        state["drag"] = None
        _refresh()

    def do_reset() -> None:
        state["a"] = a.copy()
        state["b"] = b.copy()
        state["pts"] = [a.copy(), b.copy()]
        state["poly_mode"] = True
        state["drag"] = None
        undo_stack.clear()
        _refresh()

    def do_toggle_poly() -> None:
        _push_undo()
        state["poly_mode"] = not bool(state["poly_mode"])
        if state["poly_mode"]:
            state["pts"] = [state["a"].copy(), state["b"].copy()]
        else:
            pts = [p.copy() for p in state["pts"]] if state.get("pts") else [state["a"].copy(), state["b"].copy()]
            pts.sort(key=lambda p: float(p[1]))
            state["a"], state["b"] = pts[0].copy(), pts[-1].copy()
        state["drag"] = None
        _refresh()

    ttk.Button(btns, text="Accept (Enter)", command=do_accept).grid(row=0, column=0, sticky="ew", padx=(0, 6))
    ttk.Button(btns, text="Cancel (Esc)", command=do_cancel).grid(row=0, column=1, sticky="ew", padx=(0, 6))
    ttk.Button(btns, text="Undo (U)", command=do_undo).grid(row=0, column=2, sticky="ew", padx=(0, 6))
    ttk.Button(btns, text="Reset (R)", command=do_reset).grid(row=0, column=3, sticky="ew", padx=(0, 6))
    ttk.Button(btns, text="Polyline (P)", command=do_toggle_poly).grid(row=0, column=4, sticky="ew")

    canvas.bind("<Button-1>", on_press)
    canvas.bind("<B1-Motion>", on_motion)
    canvas.bind("<ButtonRelease-1>", on_release)

    root.bind("<Return>", lambda _e: do_accept())
    root.bind("<Escape>", lambda _e: do_cancel())
    root.bind("u", lambda _e: do_undo())
    root.bind("U", lambda _e: do_undo())
    root.bind("r", lambda _e: do_reset())
    root.bind("R", lambda _e: do_reset())
    root.bind("p", lambda _e: do_toggle_poly())
    root.bind("P", lambda _e: do_toggle_poly())

    _refresh()
    root.wait_window(root)

    if result["cancelled"] or not result["accepted"]:
        return None

    # back to full-image coords
    a_full = state["a"] + np.array([x0, y0], dtype=np.float32)
    b_full = state["b"] + np.array([x0, y0], dtype=np.float32)

    # Build midline control points (ROI coords) and their full-image coordinates
    if state["poly_mode"] and isinstance(state.get("pts"), list) and len(state["pts"]) >= 2:
        pts_roi = [p.astype(np.float32).copy() for p in state["pts"]]
        pts_roi.sort(key=lambda p: float(p[1]))
    else:
        pts_roi = [state["a"].astype(np.float32).copy(), state["b"].astype(np.float32).copy()]
        pts_roi.sort(key=lambda p: float(p[1]))

    pts_full = [p + np.array([x0, y0], dtype=np.float32) for p in pts_roi]

    # --- split brain into halves by the midline (in ROI coords for speed) ---
    a_roi = pts_roi[0].astype(np.float32)
    b_roi = pts_roi[-1].astype(np.float32)
    v = b_roi - a_roi

    # If the line is degenerate, fall back to vertical split through ROI center
    if float(v[0] * v[0] + v[1] * v[1]) < 1e-6:
        v = np.array([0.0, 1.0], dtype=np.float32)
        a_roi = np.array([w * 0.5, 0.0], dtype=np.float32)
        b_roi = np.array([w * 0.5, float(h - 1)], dtype=np.float32)

    yy, xx = np.mgrid[0:h, 0:w]
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)

    # Try to split by using the midline polyline as a barrier inside the brain mask.
    barrier = np.zeros((h, w), dtype=np.uint8)
    if len(pts_roi) >= 2:
        poly = np.round(np.stack(pts_roi, axis=0)).astype(np.int32)
        # draw a slightly thick barrier so it really disconnects components
        cv2.polylines(barrier, [poly], isClosed=False, color=1, thickness=max(3, int(line_thickness)), lineType=cv2.LINE_8)

    allowed = (mroi.astype(np.uint8) & (1 - barrier)).astype(np.uint8)

    # Connected components on allowed area
    num, lab, _stats, _cent = cv2.connectedComponentsWithStats(allowed, connectivity=8)

    def _seed_label(which: str) -> int:
        ys2, xs2 = np.where(allowed > 0)
        if xs2.size == 0:
            return 0
        if which == "left":
            j = int(np.argmin(xs2))
        else:
            j = int(np.argmax(xs2))
        return int(lab[int(ys2[j]), int(xs2[j])])

    lbl_left = _seed_label("left")
    lbl_right = _seed_label("right")

    if lbl_left > 0 and lbl_right > 0 and lbl_left != lbl_right:
        half_left_cc = (lab == lbl_left)
        half_right_cc = (lab == lbl_right)
        half_pos = half_left_cc  # temporary naming, will re-map to left/right below by mean-x
        half_neg = half_right_cc
    else:
        # Fallback: straight-line sign split
        cross = (xx - a_roi[0]) * v[1] - (yy - a_roi[1]) * v[0]
        side_pos = cross > 0
        half_pos = mroi & side_pos
        half_neg = mroi & (~side_pos)

    # Determine which side is actually LEFT/RIGHT by mean x coordinate
    def _mean_x(mask: np.ndarray) -> float:
        ys, xs = np.where(mask)
        if xs.size == 0:
            return float("inf")
        return float(xs.mean())

    pos_mx = _mean_x(half_pos)
    neg_mx = _mean_x(half_neg)

    if pos_mx <= neg_mx:
        left_mask = half_pos
        right_mask = half_neg
        left_is_pos = True
    else:
        left_mask = half_neg
        right_mask = half_pos
        left_is_pos = False

    area_left_px = int(left_mask.sum())
    area_right_px = int(right_mask.sum())

    # --- perimeter per half, excluding the midline ---
    # We measure the perimeter on the ORIGINAL brain contour only, then assign contour segments to sides.
    m_u8 = (mroi.astype(np.uint8) * 255)
    cnts, _hier = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    per_pos = 0.0
    per_neg = 0.0

    if cnts:
        for c in cnts:
            if c.shape[0] < 2:
                continue
            pts = c[:, 0, :].astype(np.float32)  # (N,2) xy
            n = pts.shape[0]
            for i in range(n):
                p = pts[i]
                q = pts[(i + 1) % n]
                mid = 0.5 * (p + q)
                # segment length in pixels
                seg = float(np.hypot(q[0] - p[0], q[1] - p[1]))
                # classify by side in ROI coords
                cr = (mid[0] - a_roi[0]) * v[1] - (mid[1] - a_roi[1]) * v[0]
                if cr > 0:
                    per_pos += seg
                else:
                    per_neg += seg

    if left_is_pos:
        perimeter_left_px = float(per_pos)
        perimeter_right_px = float(per_neg)
    else:
        perimeter_left_px = float(per_neg)
        perimeter_right_px = float(per_pos)

    midline_pts = [[float(p[0]), float(p[1])] for p in pts_full]

    return {
        "midline_pts": json.dumps(midline_pts, ensure_ascii=False),
        "area_left_px": area_left_px,
        "area_right_px": area_right_px,
        "perimeter_left_px": perimeter_left_px,
        "perimeter_right_px": perimeter_right_px,
        "crop_x0": int(x0),
        "crop_y0": int(y0),
        "crop_x1": int(x1),
        "crop_y1": int(y1),
        "pad": int(pad),
    }