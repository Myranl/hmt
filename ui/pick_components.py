import numpy as np
import cv2
from typing import Tuple, Dict, Any
import json

# Helper: select components on a background RGB image
def select_components_on_background(
    sketch_u8_roi: np.ndarray,
    bg_rgb_roi: np.ndarray,
    *,
    window: str,
    init_selected: np.ndarray | None = None,
    init_cuts=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Click to toggle connected components (non-gray) while viewing them on top of a brain background.

    - sketch_u8_roi: uint8 {0,127,255}
    - bg_rgb_roi: RGB image (same HxW) shown as background

    Keys: ENTER=done, U=undo, R=reset, ESC=cancel (clears selection).
    Returns (uint8 mask (0/1) in ROI coordinates, sketch_u8 with cuts applied).

    If `init_selected` is provided (same shape as ROI), it is used as the starting selection (useful for "edit" mode).

    Note: `init_cuts` is accepted for API compatibility but not used here (cuts are already baked into sketch_u8_roi).
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

    # Modes: pick components, or edit the sketch by drawing strokes
    mode: str = "pick"  # "pick" | "cut" | "add"
    pending_pt: tuple[int, int] | None = None

    cut_thickness = 7
    add_thickness = 7

    cuts: list[tuple[tuple[int, int], tuple[int, int]]] = []      # draw value 127 (barrier)
    adds: list[tuple[tuple[int, int], tuple[int, int]]] = []      # draw value 255 (white reinforcement)

    # Unified undo stack: items are (kind, payload)
    # kind == "stroke": payload = ("cut"|"add")
    # kind == "sel": payload = (rr, cc, prev_vals)
    undo_stack: list[tuple[str, object]] = []

    def recompute_labels() -> tuple[np.ndarray, np.ndarray]:
        # apply edits onto a fresh copy
        nonlocal base
        base = base0.copy()

        # cuts first (barriers), then adds (white lines)
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

    # if we restored a previous selection, drop pixels that are no longer selectable (now gray)
    selected[(base == 127)] = 0

    # precompute edge overlay from the CURRENT base (updated after cuts)
    def compute_edges() -> np.ndarray:
        return cv2.Canny(base, 50, 150)

    edges = compute_edges()

    def redraw() -> np.ndarray:
        disp = bg_bgr0.copy()
        disp = (0.85 * disp).astype(np.uint8)

        # draw sketch edges in white
        disp[edges > 0] = (255, 255, 255)

        # selection overlay in semi-transparent green
        m = selected > 0
        if np.any(m):
            alpha = 0.4  # transparency: 0 = invisible, 1 = solid
            green = np.zeros_like(disp)
            green[:] = (0, 255, 0)
            disp[m] = (alpha * green[m] + (1 - alpha) * disp[m]).astype(np.uint8)

        if mode == "cut":
            status = "MODE: CUT"
            help2 = "CUT: click 2 points to draw a gray break line"
        elif mode == "add":
            status = "MODE: ADD"
            help2 = "ADD: click 2 points to draw a white reinforcement line"
        else:
            status = "MODE: PICK"
            help2 = "PICK: click regions to toggle selection"

        help1 = "C: CUT | A: ADD | X: clear strokes | ENTER: done | U: undo | R: reset sel | ESC: cancel"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0
        thickness = 2
        margin = 10
        pad = 8

        txt1 = f"{status} | {help1}"
        (w1, h1), _ = cv2.getTextSize(txt1, font, scale, thickness)
        (w2, h2), _ = cv2.getTextSize(help2, font, scale, thickness)

        box_w = max(w1, w2) + pad * 2
        box_h = h1 + h2 + pad * 3

        # black background box (top-left corner)
        cv2.rectangle(
            disp,
            (margin, margin),
            (margin + box_w, margin + box_h),
            (0, 0, 0),
            -1,
        )

        y1 = margin + pad + h1
        y2 = y1 + pad + h2

        cv2.putText(disp, txt1, (margin + pad, y1), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(disp, help2, (margin + pad, y2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        if mode in ("cut", "add"):
            txt = "CUT MODE" if mode == "cut" else "ADD MODE"
            font2 = cv2.FONT_HERSHEY_SIMPLEX
            scale2 = 1.6
            thickness2 = 3
            (tw, th), _ = cv2.getTextSize(txt, font2, scale2, thickness2)

            # top-right corner with margin
            margin2 = 15
            x = disp.shape[1] - tw - margin2
            y = th + margin2

            # draw black background rectangle
            pad2 = 10
            cv2.rectangle(
                disp,
                (x - pad2, y - th - pad2),
                (x + tw + pad2, y + pad2),
                (0, 0, 0),
                -1,
            )

            # draw yellow text on top
            cv2.putText(disp, txt, (x, y), font2, scale2, (0, 255, 255), thickness2, cv2.LINE_AA)

        # show pending point
        if mode in ("cut", "add") and pending_pt is not None:
            cv2.circle(disp, (int(pending_pt[0]), int(pending_pt[1])), 7, (0, 255, 255), -1)

        return disp

    disp = redraw()

    def on_mouse(event, x, y, flags, param):
        nonlocal disp, selected, pending_pt, lab, fg, edges

        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if x < 0 or y < 0 or x >= lab.shape[1] or y >= lab.shape[0]:
            return

        if mode in ("cut", "add"):
            # define a stroke segment by two clicks
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

                # recompute connectivity after edits
                lab, fg = recompute_labels()
                edges = compute_edges()

                # selection may now include pixels that became gray; clean it
                selected[(base == 127)] = 0

            disp = redraw()
            return

        # PICK mode: toggle connected component
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

        # push undo for this toggle
        undo_stack.append(("sel", (rr, cc, prev_vals)))

        # keep old history list for compatibility (not used for undo anymore)
        if not np.any(prev_vals):
            history.append(idx)

        disp = redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        cv2.imshow(window, disp)
        k = cv2.waitKey(20) & 0xFF

        if k in (13, 10):
            break
        if k == 27:
            selected[:] = 0
            break

        if k in (ord('c'), ord('C')):
            mode = "pick" if mode == "cut" else "cut"
            pending_pt = None
            disp = redraw()

        if k in (ord('a'), ord('A')):
            mode = "pick" if mode == "add" else "add"
            pending_pt = None
            disp = redraw()

        if k in (ord('x'), ord('X')):
            # clear all strokes (cuts/adds) and recompute
            cuts.clear()
            adds.clear()
            pending_pt = None
            lab, fg = recompute_labels()
            edges = compute_edges()
            selected[(base == 127)] = 0
            disp = redraw()

        if k in (ord('u'), ord('U')):
            if undo_stack:
                kind, payload = undo_stack.pop()

                if kind == "stroke":
                    stroke_kind = str(payload)
                    if stroke_kind == "cut" and cuts:
                        cuts.pop()
                    elif stroke_kind == "add" and adds:
                        adds.pop()

                    pending_pt = None
                    lab, fg = recompute_labels()
                    edges = compute_edges()
                    selected[(base == 127)] = 0

                elif kind == "sel":
                    rr, cc, prev_vals = payload  # type: ignore
                    selected[rr, cc] = prev_vals

                disp = redraw()

        if k in (ord('r'), ord('R')):
            selected[:] = 0
            history.clear()
            disp = redraw()

    cv2.destroyWindow(window)
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