import numpy as np
from PIL import Image, ImageDraw
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk

from ui.pick_components import select_components_on_background
from ui.review_ui import review_and_maybe_edit  # type: ignore
from segmentation.postprocess import smooth_fill_mask  # type: ignore
from preproc.retina import downsample_rgb_cv2, enhance_contrast_and_smooth, retina_subtract_local_mean
from preproc.quantize import sketch_three_bins, small_components_to_gray
from analysis.overlay import _overlay_masks_on_original
from ui.tk_utils import to_photo_u8, overlay_grid_and_roi


def _left_panel_photo(
    img_rgb: np.ndarray,
    *,
    max_side: int,
    grid_on: bool,
    step: int,
    roi: tuple[int, int, int, int] | None,
) -> ImageTk.PhotoImage:
    """Render the LEFT panel as a PhotoImage using the ORIGINAL RGB image.

    Grid/ROI are drawn AFTER resizing for display, but grid positions/labels are computed
    in ORIGINAL coordinates so labels remain clean multiples (200, 400, ...).
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("_left_panel_photo expects RGB image")

    im0 = Image.fromarray(img_rgb)
    w0, h0 = im0.size
    s0 = max(w0, h0)

    if s0 > max_side:
        scale = max_side / float(s0)
        w = int(round(w0 * scale))
        h = int(round(h0 * scale))
        im = im0.resize((w, h), resample=Image.Resampling.NEAREST)
    else:
        scale = 1.0
        w, h = w0, h0
        im = im0.copy()

    if grid_on or (roi is not None):
        dr = ImageDraw.Draw(im)

        # helper: draw text with black background box for readability
        def _text_box(x: int, y: int, text: str) -> None:
            # small padding box
            tw = int(dr.textlength(text))
            th = 12
            pad = 3
            x0b = max(0, x - pad)
            y0b = max(0, y - pad)
            x1b = min(w - 1, int(x + tw) + pad)
            y1b = min(h - 1, y + th + pad)
            dr.rectangle([x0b, y0b, x1b, y1b], fill=0)
            dr.text((x, y), text, fill=(220, 220, 220))

        if grid_on:
            step0 = max(10, int(step))  # ORIGINAL-coordinate step

            # draw grid lines at ORIGINAL multiples, mapped to display
            for x0 in range(0, w0, step0):
                x = int(round(x0 * scale))
                dr.line([(x, 0), (x, h)], fill=(220, 220, 220), width=2)
            for y0 in range(0, h0, step0):
                y = int(round(y0 * scale))
                dr.line([(0, y), (w, y)], fill=(220, 220, 220), width=2)

            # labels every 1 step (0, 200, 400, ...)
            for x0 in range(0, w0, step0):
                x = int(round(x0 * scale))
                _text_box(x + 4, 4, str(x0))
            for y0 in range(0, h0, step0):
                y = int(round(y0 * scale))
                _text_box(4, y + 4, str(y0))

        if roi is not None:
            x0, y0, x1, y1 = roi
            x0 = int(round(x0 * scale))
            y0 = int(round(y0 * scale))
            x1 = int(round(x1 * scale))
            y1 = int(round(y1 * scale))
            dr.rectangle([x0, y0, x1, y1], outline=(125, 125, 255), width=3)

    return ImageTk.PhotoImage(im)


def run_ui_and_get_params(gray_used: np.ndarray, img2: np.ndarray, *, t1_init: float, t2_init: float) -> dict | None:
    """Two-step UI.

    Step 1: pick ROI (drag on original) + grid settings.
    Step 2: tune 3-bin sketch params on the fixed ROI.

    Returns a dict with keys:
      t1,t2,x0,y0,x1,y1,small_to_gray,small_N,grid_on,grid_step
    or None if cancelled.
    """

    def run_roi_ui(*, img_rgb: np.ndarray) -> dict | None:
        root = tk.Tk()
        root.title("Step 1/2: pick ROI")

        frm = ttk.Frame(root, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # controls (compact)
        ctrl = ttk.Frame(frm)
        ctrl.grid(row=0, column=0, sticky="ew")

        var_grid_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="grid", variable=var_grid_on).grid(row=0, column=0, padx=(0, 10), sticky="w")

        ttk.Label(ctrl, text="step").grid(row=0, column=1, padx=(0, 6), sticky="e")
        var_grid_step = tk.StringVar(value="200")
        ttk.Entry(ctrl, textvariable=var_grid_step, width=6).grid(row=0, column=2, padx=(0, 12), sticky="w")

        lbl_info = ttk.Label(ctrl, text="Drag ROI on image. Save to continue.")
        lbl_info.grid(row=0, column=3, sticky="w")

        # image
        panes = ttk.Frame(frm)
        panes.grid(row=1, column=0, pady=(10, 0), sticky="nsew")
        frm.rowconfigure(1, weight=1)
        frm.columnconfigure(0, weight=1)

        lbl_img = ttk.Label(panes)
        lbl_img.grid(row=0, column=0, sticky="n")

        # bottom bar
        bar = ttk.Frame(frm)
        bar.grid(row=2, column=0, pady=(12, 0), sticky="ew")
        bar.columnconfigure(0, weight=1)

        chosen: dict = {"done": False}
        state = {"ph": None, "scale": 1.0, "W": 0, "H": 0}
        roi_state = {"x0": 0, "y0": 0, "x1": int(img_rgb.shape[1]), "y1": int(img_rgb.shape[0])}
        drag = {"active": False, "x0": 0, "y0": 0}

        def _event_to_img_xy(ev) -> tuple[int, int]:
            sc = float(state.get("scale", 1.0))
            if sc <= 0:
                sc = 1.0
            x = int(round(ev.x / sc))
            y = int(round(ev.y / sc))
            W = int(img_rgb.shape[1])
            H = int(img_rgb.shape[0])
            x = max(0, min(W - 1, x))
            y = max(0, min(H - 1, y))
            return x, y

        def render() -> None:
            W = int(img_rgb.shape[1])
            H = int(img_rgb.shape[0])

            max_side = 900
            s0 = max(W, H)
            scale = 1.0 if s0 <= max_side else (max_side / float(s0))
            state["scale"] = float(scale)
            state["W"] = int(round(W * scale))
            state["H"] = int(round(H * scale))

            roi = (int(roi_state["x0"]), int(roi_state["y0"]), int(roi_state["x1"]), int(roi_state["y1"]))
            step = int(float(var_grid_step.get().strip()))

            ph = _left_panel_photo(
                img_rgb,
                max_side=max_side,
                grid_on=bool(var_grid_on.get()),
                step=step,
                roi=roi,
            )
            state["ph"] = ph
            lbl_img.configure(image=ph)
            lbl_img.image = ph

            lbl_info.configure(text=f"ROI=({roi[0]},{roi[1]},{roi[2]},{roi[3]})")

        def on_press(ev) -> None:
            drag["active"] = True
            x, y = _event_to_img_xy(ev)
            drag["x0"], drag["y0"] = x, y

        def on_release(ev) -> None:
            if not drag.get("active", False):
                return
            drag["active"] = False

            x0s, y0s = int(drag["x0"]), int(drag["y0"])
            x1s, y1s = _event_to_img_xy(ev)

            x0n = min(x0s, x1s)
            y0n = min(y0s, y1s)
            x1n = max(x0s, x1s)
            y1n = max(y0s, y1s)

            if x1n <= x0n:
                x1n = min(int(img_rgb.shape[1]), x0n + 1)
            if y1n <= y0n:
                y1n = min(int(img_rgb.shape[0]), y0n + 1)

            roi_state["x0"], roi_state["y0"], roi_state["x1"], roi_state["y1"] = int(x0n), int(y0n), int(x1n), int(y1n)
            render()

        def on_save() -> None:
            try:
                chosen["grid_on"] = bool(var_grid_on.get())
                chosen["grid_step"] = int(float(var_grid_step.get().strip()))
                chosen["x0"] = int(roi_state["x0"])
                chosen["y0"] = int(roi_state["y0"])
                chosen["x1"] = int(roi_state["x1"])
                chosen["y1"] = int(roi_state["y1"])
            except Exception as e:
                lbl_info.configure(text=f"error: {e}")
                return

            chosen["done"] = True
            root.after(10, root.destroy)

        def on_cancel() -> None:
            chosen["done"] = False
            root.after(10, root.destroy)

        ttk.Button(bar, text="Cancel", command=on_cancel).grid(row=0, column=0, sticky="w")
        ttk.Button(bar, text="Save ROI", command=on_save).grid(row=0, column=1, padx=(10, 0), sticky="e")

        lbl_img.bind("<ButtonPress-1>", on_press)
        lbl_img.bind("<ButtonRelease-1>", on_release)

        render()
        root.mainloop()

        if chosen.get("done", False):
            return chosen
        return None

    def run_bins_ui(*, gray: np.ndarray, img_rgb: np.ndarray, roi: tuple[int, int, int, int], grid_on: bool, grid_step: int) -> dict | None:
        # initial thresholds
        T1_INIT = float(t1_init)
        T2_INIT = float(t2_init)

        x0, y0, x1, y1 = roi

        root = tk.Tk()
        root.title("Step 2/2: 3-bin sketch controls")

        frm = ttk.Frame(root, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # controls
        ctrl = ttk.Frame(frm)
        ctrl.grid(row=0, column=0, sticky="ew")
        ctrl.columnconfigure(7, weight=1)

        ttk.Label(ctrl, text="t1 (0..1)").grid(row=0, column=0, padx=(0, 6))
        var_t1 = tk.StringVar(value=f"{T1_INIT:.2f}")
        ttk.Entry(ctrl, textvariable=var_t1, width=8).grid(row=0, column=1, padx=(0, 12))

        ttk.Label(ctrl, text="t2 (0..1)").grid(row=0, column=2, padx=(0, 6))
        var_t2 = tk.StringVar(value=f"{T2_INIT:.2f}")
        ttk.Entry(ctrl, textvariable=var_t2, width=8).grid(row=0, column=3, padx=(0, 12))

        var_small_to_gray = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="small→gray", variable=var_small_to_gray).grid(row=0, column=4, padx=(12, 6), sticky="w")

        ttk.Label(ctrl, text="N").grid(row=0, column=5, padx=(0, 6), sticky="e")
        var_small_N = tk.StringVar(value="900")
        ttk.Entry(ctrl, textvariable=var_small_N, width=6).grid(row=0, column=6, sticky="w")

        btn_update = ttk.Button(ctrl, text="Update")
        btn_update.grid(row=0, column=7, padx=(12, 0), sticky="e")

        lbl_status = ttk.Label(ctrl, text=f"ROI fixed: ({x0},{y0},{x1},{y1})")
        lbl_status.grid(row=1, column=0, columnspan=8, sticky="w", pady=(6, 0))

        # image panels
        panes = ttk.Frame(frm)
        panes.grid(row=1, column=0, pady=(10, 0), sticky="nsew")
        frm.rowconfigure(1, weight=1)
        frm.columnconfigure(0, weight=1)

        lbl_left = ttk.Label(panes)
        lbl_left.grid(row=0, column=0, rowspan=2, padx=(0, 10), sticky="n")

        lbl_right = ttk.Label(panes)
        lbl_right.grid(row=0, column=1, padx=(10, 0), sticky="n")

        lbl_right_overlay = ttk.Label(panes)
        lbl_right_overlay.grid(row=1, column=1, padx=(10, 0), pady=(8, 0), sticky="n")

        state = {"ph_left": None, "ph_right": None, "ph_right_overlay": None, "sketch_u8": None}
        chosen: dict = {"done": False}

        def _roi_overlay_photo(
            roi_rgb: np.ndarray,
            sketch_u8: np.ndarray,
            *,
            max_side: int = 750,
            alpha: float = 0.35,
        ) -> ImageTk.PhotoImage:
            if roi_rgb.ndim != 3 or roi_rgb.shape[2] != 3:
                raise ValueError("_roi_overlay_photo expects RGB ROI")

            im0 = Image.fromarray(roi_rgb)
            w0, h0 = im0.size
            s0 = max(w0, h0)
            if s0 > max_side:
                scale = max_side / float(s0)
                w = int(round(w0 * scale))
                h = int(round(h0 * scale))
                im = im0.resize((w, h), resample=Image.Resampling.BILINEAR)
                m = Image.fromarray(sketch_u8).resize((w, h), resample=Image.Resampling.NEAREST)
            else:
                im = im0
                m = Image.fromarray(sketch_u8)

            base = np.array(im).astype(np.float32)
            mm = np.array(m)

            m_black = (mm == 255).astype(np.uint8) * 255
            m_white = (mm == 0).astype(np.uint8) * 255

            e_black = cv2.Canny(m_black, 50, 150)
            e_white = cv2.Canny(m_white, 50, 150)

            k = np.ones((3, 3), np.uint8)
            e_black = cv2.dilate(e_black, k, iterations=1)
            e_white = cv2.dilate(e_white, k, iterations=1)

            overlay = base.copy()

            # black-bin contour: green
            overlay[e_black > 0, 0] *= 0.2
            overlay[e_black > 0, 2] *= 0.2
            overlay[e_black > 0, 1] = 255

            # white-bin contour: red
            overlay[e_white > 0, 1] *= 0.2
            overlay[e_white > 0, 2] *= 0.2
            overlay[e_white > 0, 0] = 255

            if (e_black > 0).any() or (e_white > 0).any():
                out = (1 - alpha) * base + alpha * overlay
            else:
                out = base

            out_u8 = np.clip(out, 0, 255).astype(np.uint8)
            return ImageTk.PhotoImage(Image.fromarray(out_u8))

        def render() -> None:
            try:
                t1 = float(var_t1.get().strip())
                t2 = float(var_t2.get().strip())
                if not (0.0 < t1 < t2 < 1.0):
                    raise ValueError("need 0 < t1 < t2 < 1")

                gray_roi = gray[y0:y1, x0:x1]
                _, sketch_u8 = sketch_three_bins(gray_roi, t1=float(t1), t2=float(t2))

                if var_small_to_gray.get():
                    n = int(float(var_small_N.get().strip()))
                    if n <= 0:
                        raise ValueError("N must be positive")
                    sketch_u8 = small_components_to_gray(sketch_u8, min_area=n)

                state["sketch_u8"] = sketch_u8

                ph_left = _left_panel_photo(
                    img_rgb,
                    max_side=750,
                    grid_on=grid_on,
                    step=grid_step,
                    roi=roi,
                )
                ph_right = to_photo_u8(sketch_u8, max_side=750)
                roi_rgb = img_rgb[y0:y1, x0:x1]
                ph_right_overlay = _roi_overlay_photo(roi_rgb, sketch_u8, max_side=750, alpha=0.35)

                state["ph_left"] = ph_left
                state["ph_right"] = ph_right
                state["ph_right_overlay"] = ph_right_overlay

                lbl_left.configure(image=ph_left)
                lbl_right.configure(image=ph_right)
                lbl_right_overlay.configure(image=ph_right_overlay)

                lbl_left.image = ph_left
                lbl_right.image = ph_right
                lbl_right_overlay.image = ph_right_overlay

                lbl_status.configure(text=f"updated | ROI fixed: ({x0},{y0},{x1},{y1})")

            except Exception as e:
                lbl_status.configure(text=f"error: {e}")

        def on_save() -> None:
            if state["sketch_u8"] is None:
                lbl_status.configure(text="error: nothing to save")
                return

            try:
                chosen["t1"] = float(var_t1.get().strip())
                chosen["t2"] = float(var_t2.get().strip())
                chosen["small_to_gray"] = bool(var_small_to_gray.get())
                chosen["small_N"] = int(float(var_small_N.get().strip()))
            except Exception as e:
                lbl_status.configure(text=f"error: {e}")
                return

            chosen["done"] = True
            root.after(10, root.destroy)

        def on_cancel() -> None:
            chosen["done"] = False
            root.after(10, root.destroy)

        btn_update.configure(command=render)

        bar = ttk.Frame(frm)
        bar.grid(row=2, column=0, pady=(12, 0), sticky="ew")
        bar.columnconfigure(0, weight=1)

        ttk.Button(bar, text="Cancel", command=on_cancel).grid(row=0, column=0, sticky="w")
        ttk.Button(bar, text="Save", command=on_save).grid(row=0, column=1, padx=(10, 0), sticky="e")

        render()
        root.mainloop()

        if chosen.get("done", False):
            return chosen
        return None

    # ---- run step 1 ----
    roi_res = run_roi_ui(img_rgb=img2)
    if roi_res is None:
        return None

    roi = (int(roi_res["x0"]), int(roi_res["y0"]), int(roi_res["x1"]), int(roi_res["y1"]))
    grid_on = bool(roi_res["grid_on"])
    grid_step = int(roi_res["grid_step"])

    # ---- run step 2 ----
    bins_res = run_bins_ui(gray=gray_used, img_rgb=img2, roi=roi, grid_on=grid_on, grid_step=grid_step)
    if bins_res is None:
        return None

    out = {
        "t1": float(bins_res["t1"]),
        "t2": float(bins_res["t2"]),
        "x0": int(roi[0]),
        "y0": int(roi[1]),
        "x1": int(roi[2]),
        "y1": int(roi[3]),
        "small_to_gray": bool(bins_res["small_to_gray"]),
        "small_N": int(bins_res["small_N"]),
        "grid_on": bool(grid_on),
        "grid_step": int(grid_step),
    }
    return out
