from __future__ import annotations
import numpy as np
import cv2
import json


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
    a = clamp(a); b = clamp(b)

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
        "poly_mode": False,
        "pts": [a.copy(), b.copy()],
    }

    def redraw():
        d = disp_base.copy()
        if state["poly_mode"]:
            pts = [p.copy() for p in state["pts"]]
            # sort by Y for a stable up->down ordering
            pts.sort(key=lambda p: float(p[1]))
            if len(pts) >= 2:
                poly = np.round(np.stack(pts, axis=0)).astype(np.int32)
                cv2.polylines(d, [poly], isClosed=False, color=line_color_bgr, thickness=line_thickness, lineType=cv2.LINE_AA)
            for p in pts:
                pp = tuple(np.round(p).astype(int))
                cv2.circle(d, pp, 7, (255, 255, 255), -1, cv2.LINE_AA)
        else:
            aa = tuple(np.round(state["a"]).astype(int))
            bb = tuple(np.round(state["b"]).astype(int))
            cv2.line(d, aa, bb, line_color_bgr, line_thickness, cv2.LINE_AA)
            cv2.circle(d, aa, 7, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(d, bb, 7, (255, 255, 255), -1, cv2.LINE_AA)

        # label box
        txt = "MIDLINE MODE  |  P: polyline  |  click/drag points  |  ENTER: accept  R: reset  ESC: cancel"
        font_scale = 0.8
        font_thickness = 3
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.rectangle(d, (8, 8), (8 + tw + 14, 8 + th + 18), (0, 0, 0), -1)
        cv2.putText(d, txt, (15, 8 + th + 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), font_thickness, cv2.LINE_AA)
        return d

    def pick_handle(x: int, y: int, p: np.ndarray, r: int = 14) -> bool:
        return (x - float(p[0])) ** 2 + (y - float(p[1])) ** 2 <= float(r * r)

    def _nearest_point_index(x: int, y: int) -> int | None:
        pts = state["pts"]
        if not pts:
            return None
        d2 = [float((x - p[0]) ** 2 + (y - p[1]) ** 2) for p in pts]
        j = int(np.argmin(d2))
        if d2[j] <= 14.0 * 14.0:
            return j
        return None

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if state["poly_mode"]:
                j = _nearest_point_index(x, y)
                if j is not None:
                    state["drag"] = j
                    return
                p = clamp(np.array([x, y], dtype=np.float32))
                state["pts"].append(p)
                state["drag"] = len(state["pts"]) - 1
                return

            # straight-line mode: drag endpoints only
            if pick_handle(x, y, state["a"]):
                state["drag"] = "a"
            elif pick_handle(x, y, state["b"]):
                state["drag"] = "b"

        elif event == cv2.EVENT_MOUSEMOVE and state["drag"] is not None:
            p = clamp(np.array([x, y], dtype=np.float32))
            if state["poly_mode"]:
                if isinstance(state["drag"], int):
                    state["pts"][state["drag"]] = p
            else:
                state[state["drag"]] = p

        elif event == cv2.EVENT_LBUTTONUP:
            state["drag"] = None

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    a0, b0 = state["a"].copy(), state["b"].copy()
    pts0 = [state["a"].copy(), state["b"].copy()]

    while True:
        cv2.imshow(window, redraw())
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10):  # ENTER
            break
        if k == 27:  # ESC
            cv2.destroyWindow(window)
            return None
        if k in (ord("p"), ord("P")):
            state["poly_mode"] = not bool(state["poly_mode"])
            if state["poly_mode"]:
                # initialize polyline from current endpoints
                state["pts"] = [state["a"].copy(), state["b"].copy()]
            else:
                # collapse back to endpoints (top/bottom by Y)
                pts = [p.copy() for p in state["pts"]] if state.get("pts") else [state["a"].copy(), state["b"].copy()]
                pts.sort(key=lambda p: float(p[1]))
                state["a"], state["b"] = pts[0].copy(), pts[-1].copy()

        if k in (ord("r"), ord("R")):
            state["a"] = a0.copy()
            state["b"] = b0.copy()
            state["pts"] = [p.copy() for p in pts0]
            state["poly_mode"] = False

    cv2.destroyWindow(window)

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