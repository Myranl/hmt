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
) -> tuple[np.ndarray, np.ndarray]:
    """Show overall overlay with 3 buttons.

    When re-picking left/right, we seed the picker with the previous selection and reuse any existing CUT lines.

    Returns updated (left_roi_sel, right_roi_sel) in ROI coordinates.
    """

    x0, y0, x1, y1 = roi
    H, W = img2_rgb.shape[:2]

    # mutable holders
    cur_left = left_roi_sel.copy()
    cur_right = right_roi_sel.copy()
    cur_sketch = sketch_u8_roi.copy()

    # keep track of CUT lines so re-pick doesn't forget them
    cur_cuts: list[tuple[tuple[int, int], tuple[int, int]]] = []

    root = tk.Tk()
    root.title("Review segmentation")

    frm = ttk.Frame(root, padding=10)
    frm.grid(row=0, column=0, sticky="nsew")
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)

    lbl = ttk.Label(frm)
    lbl.grid(row=0, column=0, columnspan=3, pady=(0, 10))

    state = {"photo": None}

    def make_overlay_u8() -> np.ndarray:
        left_ds = np.zeros((H, W), dtype=np.uint8)
        right_ds = np.zeros((H, W), dtype=np.uint8)

        # show FINAL polygons: connect/fill/smooth the selected segments
        left_poly = smooth_fill_mask(cur_left, close_ksize=25, open_ksize=7, blur_sigma=2.0)
        right_poly = smooth_fill_mask(cur_right, close_ksize=25, open_ksize=7, blur_sigma=2.0)

        left_ds[y0:y1, x0:x1] = left_poly
        right_ds[y0:y1, x0:x1] = right_poly

        out = img2_rgb.astype(np.float32).copy()
        alpha = 0.35

        mL = left_ds.astype(bool)
        mR = right_ds.astype(bool)

        # left = red-ish
        out[mL, 0] = (1 - alpha) * out[mL, 0] + alpha * 255
        out[mL, 1] = (1 - alpha) * out[mL, 1]
        out[mL, 2] = (1 - alpha) * out[mL, 2]

        # right = blue-ish
        out[mR, 2] = (1 - alpha) * out[mR, 2] + alpha * 255
        out[mR, 0] = (1 - alpha) * out[mR, 0]
        out[mR, 1] = (1 - alpha) * out[mR, 1]

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

    def change_left() -> None:
        nonlocal cur_left, cur_sketch, cur_cuts
        cur_left, cur_sketch = select_components_on_background(
            cur_sketch,
            bg_roi,
            window="Re-pick LEFT (green)",
            init_selected=cur_left,
            init_cuts=cur_cuts,
        )
        refresh()

    def change_right() -> None:
        nonlocal cur_right, cur_sketch, cur_cuts
        cur_right, cur_sketch = select_components_on_background(
            cur_sketch,
            bg_roi,
            window="Re-pick RIGHT (green)",
            init_selected=cur_right,
            init_cuts=cur_cuts,
        )
        refresh()

    def all_ok() -> None:
        root.destroy()

    ttk.Button(frm, text="all ok", command=all_ok).grid(row=1, column=0, padx=5)
    ttk.Button(frm, text="change left", command=change_left).grid(row=1, column=1, padx=5)
    ttk.Button(frm, text="change right", command=change_right).grid(row=1, column=2, padx=5)

    refresh()
    root.mainloop()

    return cur_left, cur_right