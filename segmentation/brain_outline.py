import numpy as np
import cv2


def _to_u8(gray: np.ndarray) -> np.ndarray:
    """gray can be float [0..1] or uint8."""
    if gray.dtype == np.uint8:
        return gray
    g = np.asarray(gray, dtype=np.float32)
    # robust clamp if someone passes weird ranges
    gmin, gmax = np.nanpercentile(g, [1, 99])
    if not np.isfinite(gmin) or not np.isfinite(gmax) or gmax <= gmin:
        gmin, gmax = float(np.nanmin(g)), float(np.nanmax(g))
        if not np.isfinite(gmin) or not np.isfinite(gmax) or gmax <= gmin:
            return np.zeros_like(g, dtype=np.uint8)
    g = (g - gmin) / (gmax - gmin)
    g = np.clip(g, 0.0, 1.0)
    return (g * 255.0 + 0.5).astype(np.uint8)


def _largest_component(mask_u8: np.ndarray) -> np.ndarray:
    """mask_u8 is 0/255. Return 0/255 largest CC."""
    num, lab, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return np.zeros_like(mask_u8, dtype=np.uint8)
    # stats[0] is background
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    return ((lab == idx).astype(np.uint8) * 255)


def _fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    """Fill holes in a 0/255 mask using flood fill."""
    m = (mask_u8 > 0).astype(np.uint8) * 255
    h, w = m.shape
    flood = m.copy()
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(m, flood_inv)


def brain_mask_from_threshold(
    img_rgb_or_gray: np.ndarray,
    *,
    thr: int,
    invert: bool = True,
    pre_close: int = 11,
    pre_open: int = 5,
    min_area: int = 20000,
    fill_holes: bool = True,
) -> np.ndarray:
    """Binary brain mask from a single threshold.

    - Assumes background is bright and tissue is darker.
    - If `invert=True`, we treat 'dark' as foreground by using (gray < thr).
    Returns boolean mask.
    """
    if img_rgb_or_gray.ndim == 3:
        g = cv2.cvtColor(img_rgb_or_gray, cv2.COLOR_RGB2GRAY)
    else:
        g = _to_u8(img_rgb_or_gray)

    if invert:
        m = (g < int(thr)).astype(np.uint8) * 255
    else:
        m = (g > int(thr)).astype(np.uint8) * 255

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(pre_close), int(pre_close)))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(pre_open), int(pre_open)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)

    m = _largest_component(m)
    if int((m > 0).sum()) < int(min_area):
        return np.zeros(m.shape, dtype=bool)

    if fill_holes:
        m = _fill_holes(m)

    return (m > 0)


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

    def _odd(x: int) -> int:
        x = int(x)
        if x < 1:
            return 1
        return x if (x % 2 == 1) else (x + 1)

    def smooth_contour(cnt: np.ndarray, k: int) -> np.ndarray:
        # cnt: (N,1,2)
        if cnt is None or len(cnt) < 5:
            return cnt
        k = _odd(k)
        pts = cnt[:, 0, :].astype(np.float32)
        if pts.shape[0] < k:
            return cnt
        # circular smoothing
        pad = k // 2
        pts_pad = np.vstack([pts[-pad:], pts, pts[:pad]])
        kernel = np.ones((k,), dtype=np.float32) / float(k)
        xs = np.convolve(pts_pad[:, 0], kernel, mode="valid")
        ys = np.convolve(pts_pad[:, 1], kernel, mode="valid")
        sm = np.stack([xs, ys], axis=1)
        sm = np.round(sm).astype(np.int32)
        sm = sm.reshape((-1, 1, 2))
        return sm

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    cv2.createTrackbar("thr", window, int(init_thr), 255, lambda _: None)
    cv2.createTrackbar("smooth", window, int(init_smooth), 101, lambda _: None)
    cv2.createTrackbar("close", window, int(init_close), 101, lambda _: None)
    cv2.createTrackbar("open", window, int(init_open), 101, lambda _: None)

    cur_thr = int(init_thr)
    cur_smk = int(init_smooth)
    cur_csz = int(init_close)
    cur_osz = int(init_open)

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

        m = (g < int(thr)).astype(np.uint8) * 255

        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (csz, csz))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (osz, osz))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)

        m = _largest_component(m)
        if int((m > 0).sum()) >= int(min_area):
            m = _fill_holes(m)

        # contour
        cnts, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea) if cnts else None
        cnt_s = smooth_contour(cnt, smk) if cnt is not None else None

        # preview: original + contour + binary side-by-side
        vis = img.copy()
        if cnt_s is not None:
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            cv2.drawContours(vis_bgr, [cnt_s], -1, (0, 255, 0), 4)
            vis = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

        m_rgb = cv2.cvtColor(m, cv2.COLOR_GRAY2RGB)

        top = vis
        bot = m_rgb

        # compute simple metrics on current mask
        area_px = int((m > 0).sum())
        cnt_use = cnt_s if cnt_s is not None else cnt
        perim_px = float(cv2.arcLength(cnt_use, True)) if cnt_use is not None else 0.0

        # annotate RIGHT panel (binary)
        bot_bgr = cv2.cvtColor(bot, cv2.COLOR_RGB2BGR)
        def _put_text_box_b(img_bgr, text: str, org: tuple[int, int], *, font_scale: float = 0.7, thickness: int = 2, pad: int = 6):
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), base = cv2.getTextSize(text, font, font_scale, thickness)
            x, y = int(org[0]), int(org[1])
            x0 = max(0, x - pad)
            y0 = max(0, y - th - pad)
            x1 = min(img_bgr.shape[1] - 1, x + tw + pad)
            y1 = min(img_bgr.shape[0] - 1, y + base + pad)
            cv2.rectangle(img_bgr, (x0, y0), (x1, y1), (0, 0, 0), -1)
            cv2.putText(img_bgr, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        _put_text_box_b(bot_bgr, f"area={area_px} px", (10, 30))
        _put_text_box_b(bot_bgr, f"perim={perim_px:.1f} px", (10, 62))
        bot = cv2.cvtColor(bot_bgr, cv2.COLOR_BGR2RGB)

        # put text with black background box for readability
        top_bgr = cv2.cvtColor(top, cv2.COLOR_RGB2BGR)

        def _put_text_box(img_bgr, text: str, org: tuple[int, int], *, font_scale: float = 0.7, thickness: int = 2, pad: int = 6):
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), base = cv2.getTextSize(text, font, font_scale, thickness)
            x, y = int(org[0]), int(org[1])
            # background rectangle (black)
            x0 = max(0, x - pad)
            y0 = max(0, y - th - pad)
            x1 = min(img_bgr.shape[1] - 1, x + tw + pad)
            y1 = min(img_bgr.shape[0] - 1, y + base + pad)
            cv2.rectangle(img_bgr, (x0, y0), (x1, y1), (0, 0, 0), -1)
            cv2.putText(img_bgr, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        _put_text_box(top_bgr, "ENTER=accept | ESC=cancel", (10, 30))
        _put_text_box(top_bgr, f"thr={thr}  close={csz} open={osz} smooth={smk}", (10, 62))

        top = cv2.cvtColor(top_bgr, cv2.COLOR_BGR2RGB)

        show = np.hstack([top, bot])
        show_bgr = cv2.cvtColor(show, cv2.COLOR_RGB2BGR)
        cv2.imshow(window, show_bgr)

        k = cv2.waitKey(30) & 0xFF
        if k in (13, 10):
            last = (m > 0)
            break
        if k == 27:
            last = None
            break

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