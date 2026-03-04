import numpy as np
from PIL import Image, ImageDraw
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk
from ui.pick_components import select_components_on_background
from segmentation.postprocess import smooth_fill_mask

def review_and_maybe_edit(
    *,
    img2_rgb: np.ndarray,
    sketch_u8_roi: np.ndarray,
    bg_roi: np.ndarray,
    roi: tuple[int, int, int, int],
    left_roi_sel: np.ndarray,
    right_roi_sel: np.ndarray,
    brain_mask_full: np.ndarray | None = None,
    brain_pad: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Show overlay with edit and save buttons.

    When re-picking, we seed the picker with the previous (left∪right) selection and reuse any existing CUT lines. The updated selection is then split into left/right automatically.

    Returns updated (left_roi_sel, right_roi_sel) in ROI coordinates.
    """

    x0, y0, x1, y1 = roi
    H, W = img2_rgb.shape[:2]

    # Display crop: brain mask bbox + pad (full-res). If mask not provided, show full image.
    crop_x0, crop_y0, crop_x1, crop_y1 = 0, 0, W, H
    if brain_mask_full is not None and brain_mask_full.shape[:2] == (H, W) and np.any(brain_mask_full):
        ys, xs = np.where(brain_mask_full > 0)
        pad = int(max(0, brain_pad))
        crop_y0 = max(int(ys.min()) - pad, 0)
        crop_y1 = min(int(ys.max()) + pad + 1, H)
        crop_x0 = max(int(xs.min()) - pad, 0)
        crop_x1 = min(int(xs.max()) + pad + 1, W)

    # mutable holders
    cur_left = left_roi_sel.copy()
    cur_right = right_roi_sel.copy()
    cur_sketch = sketch_u8_roi.copy()

    # keep track of CUT lines so re-pick doesn't forget them
    cur_cuts: list[tuple[tuple[int, int], tuple[int, int]]] = []

    root = tk.Tk()
    root.title("Review hippocampus")

    frm = ttk.Frame(root, padding=10)
    frm.grid(row=0, column=0, sticky="nsew")
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)

    # Top control row (always visible)
    btn_row = ttk.Frame(frm)
    btn_row.grid(row=0, column=0, sticky="w", pady=(0, 8))

    # Image below
    lbl = ttk.Label(frm)
    lbl.grid(row=1, column=0, sticky="nsew")

    frm.rowconfigure(1, weight=1)
    frm.columnconfigure(0, weight=1)

    state = {"photo": None}

    def make_overlay_u8() -> np.ndarray:
        # Build full-res overlays then crop for display
        left_full = np.zeros((H, W), dtype=np.uint8)
        right_full = np.zeros((H, W), dtype=np.uint8)

        # show FINAL polygons: connect/fill/smooth the selected segments
        left_poly = smooth_fill_mask(cur_left, close_ksize=25, open_ksize=7, blur_sigma=2.0)
        right_poly = smooth_fill_mask(cur_right, close_ksize=25, open_ksize=7, blur_sigma=2.0)

        left_full[y0:y1, x0:x1] = left_poly
        right_full[y0:y1, x0:x1] = right_poly

        # crop for display
        img_crop = img2_rgb[crop_y0:crop_y1, crop_x0:crop_x1].astype(np.float32).copy()
        left_crop = left_full[crop_y0:crop_y1, crop_x0:crop_x1].astype(bool)
        right_crop = right_full[crop_y0:crop_y1, crop_x0:crop_x1].astype(bool)

        out = img_crop
        alpha = 0.35

        # left = red-ish
        out[left_crop, 0] = (1 - alpha) * out[left_crop, 0] + alpha * 255
        out[left_crop, 1] = (1 - alpha) * out[left_crop, 1]
        out[left_crop, 2] = (1 - alpha) * out[left_crop, 2]

        # right = blue-ish
        out[right_crop, 2] = (1 - alpha) * out[right_crop, 2] + alpha * 255
        out[right_crop, 0] = (1 - alpha) * out[right_crop, 0]
        out[right_crop, 1] = (1 - alpha) * out[right_crop, 1]

        return np.clip(out, 0, 255).astype(np.uint8)

    def refresh() -> None:
        ov = make_overlay_u8()
        im = Image.fromarray(ov)

        # fit to screen-ish
        max_side = 1100
        w, h = im.size
        s = max(w, h)
        if s > max_side:
            scale = max_side / float(s)
            im = im.resize((int(round(w * scale)), int(round(h * scale))), resample=Image.Resampling.BILINEAR)

        state["photo"] = ImageTk.PhotoImage(im)
        lbl.configure(image=state["photo"])

    def edit_selection() -> None:
        nonlocal cur_left, cur_right, cur_sketch, cur_cuts

        # pick components once (user does not care about left/right here)
        init_union = (cur_left.astype(bool) | cur_right.astype(bool)).astype(np.uint8)
        sel_roi, cur_sketch = select_components_on_background(
            cur_sketch,
            bg_roi,
            window="Re-pick hippocampus (green)",
            init_selected=init_union,
            init_cuts=cur_cuts,
        )

        # split automatically by overlap with previous left/right (fallback: x-centroid)
        sel_roi = (sel_roi > 0).astype(np.uint8)
        cur_left_prev = cur_left.astype(bool)
        cur_right_prev = cur_right.astype(bool)

        num, lab = cv2.connectedComponents(sel_roi, connectivity=8)
        new_left = np.zeros_like(sel_roi, dtype=np.uint8)
        new_right = np.zeros_like(sel_roi, dtype=np.uint8)

        h_roi, w_roi = sel_roi.shape
        x_mid = w_roi / 2.0

        for idx in range(1, int(num)):
            comp = (lab == idx)
            if not np.any(comp):
                continue
            ol = int(np.sum(comp & cur_left_prev))
            orr = int(np.sum(comp & cur_right_prev))

            if ol == 0 and orr == 0:
                xs = np.where(comp)[1]
                cx = float(xs.mean()) if xs.size else x_mid
                to_right = cx >= x_mid
            else:
                to_right = orr > ol

            if to_right:
                new_right[comp] = 1
            else:
                new_left[comp] = 1

        cur_left = new_left
        cur_right = new_right
        refresh()

    def save_and_close() -> None:
        root.destroy()

    ttk.Button(btn_row, text="Edit", command=edit_selection).grid(row=0, column=0, padx=(0, 8))
    ttk.Button(btn_row, text="Save", command=save_and_close).grid(row=0, column=1)

    # Button-driven only (avoid accidental Enter)
    root.bind("<Escape>", lambda _e: save_and_close())

    refresh()
    root.mainloop()

    return cur_left, cur_right