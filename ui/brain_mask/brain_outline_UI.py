import numpy as np
import cv2
from ui.brain_mask.mask_utils import _gray_to_u8, _odd, _put_text_box, smooth_contour
from ui.brain_mask.mask_morphology import _fill_holes, _largest_component, _convex_hull_mask, _apply_edit_layers
from ui.brain_mask.mask_compute import compute_mask

def _connected_component_from_seed(bin_u8: np.ndarray, x: int, y: int, *, search_r: int = 10) -> np.ndarray:
    """Given a binary 0/255 image and a seed (x,y), return a 0/255 mask of that CC.

    If the click lands on background, we search a small neighborhood for the nearest
    foreground pixel and use it as the seed.
    """
    h, w = bin_u8.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return np.zeros_like(bin_u8, dtype=np.uint8)

    fg = (bin_u8 > 0).astype(np.uint8)

    # If click is on background, search nearby for a foreground pixel.
    if fg[y, x] == 0 and search_r > 0:
        x0 = max(0, x - int(search_r))
        x1 = min(w, x + int(search_r) + 1)
        y0 = max(0, y - int(search_r))
        y1 = min(h, y + int(search_r) + 1)
        win = fg[y0:y1, x0:x1]
        ys, xs = np.where(win > 0)
        if ys.size == 0:
            return np.zeros_like(bin_u8, dtype=np.uint8)
        # pick nearest fg pixel
        dy = ys.astype(np.int32) + y0 - y
        dx = xs.astype(np.int32) + x0 - x
        i = int(np.argmin(dy * dy + dx * dx))
        y = int(ys[i] + y0)
        x = int(xs[i] + x0)

    if fg[y, x] == 0:
        return np.zeros_like(bin_u8, dtype=np.uint8)

    num, lab, _stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if num <= 1:
        return np.zeros_like(bin_u8, dtype=np.uint8)
    idx = int(lab[y, x])
    if idx <= 0:
        return np.zeros_like(bin_u8, dtype=np.uint8)
    return (lab == idx).astype(np.uint8) * 255


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
    """OpenCV UI to tune threshold + contour smoothing.

    Controls:
      - thr: threshold (0..255)
      - smooth: contour smoothing strength (odd kernel size)
      - close/open: morphology kernel sizes

    Keys:
      - ENTER: accept
      - ESC: cancel (returns empty mask)

    Returns:
      (mask_bool, params_dict)
    """
    img = img_rgb

    # downsample for speed
    h0, w0 = img.shape[:2]
    scale = min(1.0, downsample_max_side / float(max(h0, w0)))
    if scale < 1.0:
        new_w = int(round(w0 * scale))
        new_h = int(round(h0 * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # base grayscale (u8)
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    cv2.createTrackbar("thr", window, int(init_thr), 255, lambda _: None)
    cv2.createTrackbar("smooth", window, int(init_smooth), 101, lambda _: None)
    cv2.createTrackbar("close", window, int(init_close), 101, lambda _: None)
    cv2.createTrackbar("open", window, int(init_open), 101, lambda _: None)
    # used only for manual ERASE: how strong the opening is when detecting bumps/spurs
    cv2.createTrackbar("edit_open", window, 41, 151, lambda _: None)

    cur_thr = int(init_thr)
    cur_smk = int(init_smooth)
    cur_csz = int(init_close)
    cur_osz = int(init_open)

    edit_add_u8 = np.zeros(g.shape, dtype=np.uint8)
    edit_del_u8 = np.zeros(g.shape, dtype=np.uint8)
    mode = {"kind": "erase"}  # dict so inner callbacks can mutate

    # Undo stack for manual edits: store (add_layer, del_layer)
    undo_stack: list[tuple[np.ndarray, np.ndarray]] = []

    def _push_undo() -> None:
        # store copies of the edit layers
        undo_stack.append((edit_add_u8.copy(), edit_del_u8.copy()))
        # cap size to avoid unbounded RAM usage
        if len(undo_stack) > 50:
            undo_stack.pop(0)

    def _undo_last() -> None:
        if not undo_stack:
            return
        a, d = undo_stack.pop()
        edit_add_u8[:] = a
        edit_del_u8[:] = d

    state = {
        "m_u8": np.zeros(g.shape, dtype=np.uint8),
        "show_mask": True,
        "show_protrusions": True,
    }

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        m_current_u8 = state["m_u8"]
        ix = int(x)
        iy = int(y)
        if ix < 0 or iy < 0 or ix >= m_current_u8.shape[1] or iy >= m_current_u8.shape[0]:
            return
        if mode["kind"] == "erase":
            # Remove only outward bumps/spurs: pixels that disappear under a strong opening.
            # (Convex hull doesn't work: mask is always inside hull.)
            edit_open = cv2.getTrackbarPos("edit_open", window) if "window" in locals() else 41
            edit_open = int(edit_open)
            if edit_open < 3:
                edit_open = 3
            if edit_open % 2 == 0:
                edit_open += 1

            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edit_open, edit_open))
            base = cv2.morphologyEx(m_current_u8, cv2.MORPH_OPEN, k)

            protrusions = cv2.bitwise_and(m_current_u8, cv2.bitwise_not(base))
            cc = _connected_component_from_seed(protrusions, ix, iy, search_r=12)
            if cc.sum() > 0:
                _push_undo()
                edit_del_u8[:] = cv2.bitwise_or(edit_del_u8, cc)
        elif mode["kind"] == "add":
            hull = _convex_hull_mask(m_current_u8)
            indent = cv2.bitwise_and(hull, cv2.bitwise_not(m_current_u8))
            cc = _connected_component_from_seed(indent, ix, iy, search_r=12)
            if cc.sum() > 0:
                _push_undo()
                edit_add_u8[:] = cv2.bitwise_or(edit_add_u8, cc)

    cv2.setMouseCallback(window, on_mouse)

    last = None

    while True:
        thr = cv2.getTrackbarPos("thr", window)
        smk = cv2.getTrackbarPos("smooth", window)
        csz = cv2.getTrackbarPos("close", window)
        osz = cv2.getTrackbarPos("open", window)

        cur_thr = int(thr)
        cur_smk = int(smk)
        cur_csz = int(csz)
        cur_osz = int(osz)

        csz = _odd(max(1, csz))
        osz = _odd(max(1, osz))
        smk = _odd(max(1, smk))

        auto_m = (g < int(thr)).astype(np.uint8) * 255

        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (csz, csz))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (osz, osz))
        auto_m = cv2.morphologyEx(auto_m, cv2.MORPH_CLOSE, k_close)
        auto_m = cv2.morphologyEx(auto_m, cv2.MORPH_OPEN, k_open)

        auto_m = _largest_component(auto_m)
        if int((auto_m > 0).sum()) >= int(min_area):
            auto_m = _fill_holes(auto_m)

        m = _apply_edit_layers(auto_m, edit_add_u8, edit_del_u8)

        state["m_u8"] = m

        # contour
        cnts, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea) if cnts else None
        cnt_s = smooth_contour(cnt, smk) if cnt is not None else None

        # compute simple metrics on current mask
        area_px = int((m > 0).sum())
        cnt_use = cnt_s if cnt_s is not None else cnt
        perim_px = float(cv2.arcLength(cnt_use, True)) if cnt_use is not None else 0.0

        # Build display: RGB image + contours
        vis_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # main brain mask overlay (semi-transparent green)
        if state["show_mask"]:
            alpha = 0.28  # transparency
            green = np.zeros_like(vis_bgr, dtype=np.uint8)
            green[:, :, 1] = 255
            m_bool = (m > 0)
            # blend only where mask is true
            vis_bgr[m_bool] = cv2.addWeighted(vis_bgr[m_bool], 1.0 - alpha, green[m_bool], alpha, 0.0)

            # optional thin boundary (subtle) for readability
            if cnt_s is not None:
                cv2.drawContours(vis_bgr, [cnt_s], -1, (0, 200, 0), 2)

        # show what will be removed by ERASE: protrusions (muted red) based on edit_open
        if state["show_protrusions"]:
            edit_open = int(cv2.getTrackbarPos("edit_open", window))
            if edit_open < 3:
                edit_open = 3
            if edit_open % 2 == 0:
                edit_open += 1
            k_edit = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edit_open, edit_open))
            base = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_edit)
            protrusions = cv2.bitwise_and(m, cv2.bitwise_not(base))
            if int((protrusions > 0).sum()) > 0:
                p_cnts, _ = cv2.findContours((protrusions > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if p_cnts:
                    # muted brick-red and thinner
                    cv2.drawContours(vis_bgr, p_cnts, -1, (30, 30, 180), 1)

        # text overlay helpers

        mode_str = mode["kind"].upper()
        _put_text_box(vis_bgr, f"Mode: {mode_str} (E) / ADD (A) | U=undo | C=clear | M=mask | P=protrusions", (10, 30))
        _put_text_box(vis_bgr, f"thr={thr}  close={csz} open={osz} smooth={smk}  edit_open={cv2.getTrackbarPos('edit_open', window)}", (10, 62))
        _put_text_box(vis_bgr, f"area={area_px} px  perim={perim_px:.1f} px", (10, 94))

        cv2.imshow(window, vis_bgr)
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 10):
            last = (m > 0)
            break
        if k == 27:
            last = None
            break
        if k in (ord('e'), ord('E')):
            mode['kind'] = 'erase'
        if k in (ord('a'), ord('A')):
            mode['kind'] = 'add'
        if k in (ord('u'), ord('U')):
            _undo_last()
        if k in (ord('c'), ord('C')):
            _push_undo()
            edit_add_u8[:] = 0
            edit_del_u8[:] = 0
        if k in (ord('m'), ord('M')):
            state["show_mask"] = not state["show_mask"]
        if k in (ord('p'), ord('P')):
            state["show_protrusions"] = not state["show_protrusions"]

    cv2.destroyWindow(window)

    if last is None:
        return np.zeros((h0, w0), dtype=bool), {
            "accepted": False,
            "thr": int(cur_thr),
            "close": int(_odd(max(1, cur_csz))),
            "open": int(_odd(max(1, cur_osz))),
            "smooth": int(_odd(max(1, cur_smk))),
            "scale": float(scale),
            "area_px": 0,
            "perim_px": 0.0,
        }

    # upscale accepted mask back to original size
    if scale < 1.0:
        last_u8 = last.astype(np.uint8)
        last_u8 = cv2.resize(last_u8, (w0, h0), interpolation=cv2.INTER_NEAREST)
        last = last_u8.astype(bool)

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
        "thr": int(thr),
        "close": int(csz),
        "open": int(osz),
        "smooth": int(smk),
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