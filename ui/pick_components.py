import numpy as np
import cv2


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

    cut_mode = False
    cut_pending: tuple[int, int] | None = None
    cut_thickness = 7
    cuts: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def recompute_labels() -> tuple[np.ndarray, np.ndarray]:
        # apply cuts onto a fresh copy
        nonlocal base
        base = base0.copy()
        if cuts:
            for (x1, y1), (x2, y2) in cuts:
                cv2.line(base, (int(x1), int(y1)), (int(x2), int(y2)), 127, int(cut_thickness))

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

        status = "CUT ON" if cut_mode else "CUT OFF"
        help1 = "C: cut mode | X: clear cuts | ENTER: done | U: undo sel | R: reset sel | ESC: cancel"
        help2 = "CUT mode: click 2 points to draw a break line"

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

        if cut_mode:
            txt = "CUT MODE"
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

        # show pending cut point
        if cut_mode and cut_pending is not None:
            cv2.circle(disp, (int(cut_pending[0]), int(cut_pending[1])), 6, (0, 255, 255), -1)

        return disp

    disp = redraw()

    def on_mouse(event, x, y, flags, param):
        nonlocal disp, selected, cut_pending, lab, fg, edges

        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if x < 0 or y < 0 or x >= lab.shape[1] or y >= lab.shape[0]:
            return

        if cut_mode:
            # define a cut segment by two clicks
            if cut_pending is None:
                cut_pending = (x, y)
            else:
                cuts.append((cut_pending, (x, y)))
                cut_pending = None

                # recompute connectivity after cut
                lab, fg = recompute_labels()
                edges = compute_edges()

                # selection may now include pixels that became gray; clean it
                selected[(base == 127)] = 0

            disp = redraw()
            return

        # normal mode: toggle component
        idx = int(lab[y, x])
        if idx <= 0:
            return
        mask = (lab == idx)
        if np.any(selected[mask]):
            selected[mask] = 0
        else:
            selected[mask] = 1
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
            cut_mode = not cut_mode
            cut_pending = None
            disp = redraw()

        if k in (ord('x'), ord('X')):
            # clear cuts and recompute
            cuts.clear()
            cut_pending = None
            lab, fg = recompute_labels()
            edges = compute_edges()
            selected[(base == 127)] = 0
            disp = redraw()

        if k in (ord('u'), ord('U')):
            # undo last selection component by id (best-effort; id may be stale after cuts)
            if history:
                idx = history.pop()
                selected[lab == idx] = 0
                disp = redraw()

        if k in (ord('r'), ord('R')):
            selected[:] = 0
            history.clear()
            disp = redraw()

    cv2.destroyWindow(window)
    return selected, base