import numpy as np
from PIL import Image
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk
from preproc.quantize import sketch_three_bins, small_components_to_gray
from ui.common.tk_utils import to_photo_u8

def run_bins_ui(*, gray: np.ndarray, img_rgb: np.ndarray, roi: tuple[int, int, int, int], grid_on: bool,
                grid_step: int, t1_init: float = 0.33, t2_init: float = 0.66) -> dict | None:
    # initial thresholds
    T1_INIT = float(t1_init)
    T2_INIT = float(t2_init)

    x0, y0, x1, y1 = roi

    root = tk.Tk()
    root.title("3-bin sketch (thresholds)")

    frm = ttk.Frame(root, padding=10)
    frm.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    frm.columnconfigure(1, weight=1)
    frm.rowconfigure(1, weight=1)

    # left: controls
    ctrl = ttk.Frame(frm)
    ctrl.grid(row=0, column=0, rowspan=2, padx=(0, 12), sticky="nw")

    lf_thresh = ttk.LabelFrame(ctrl, text="Пороги t1 / t2")
    lf_thresh.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
    lf_thresh.columnconfigure(1, weight=1)

    var_t1 = tk.DoubleVar(value=float(T1_INIT))
    lbl_t1 = ttk.Label(lf_thresh, text=f"{var_t1.get():.2f}", width=4)
    ttk.Label(lf_thresh, text="t1").grid(row=0, column=0, padx=(0, 4), sticky="w")
    lbl_t1.grid(row=0, column=1, padx=(0, 6), sticky="e")
    s_t1 = ttk.Scale(lf_thresh, from_=0.0, to=1.0, orient="horizontal", variable=var_t1)
    s_t1.grid(row=1, column=0, columnspan=2, padx=(0, 0), pady=(2, 6), sticky="ew")

    var_t2 = tk.DoubleVar(value=float(T2_INIT))
    lbl_t2 = ttk.Label(lf_thresh, text=f"{var_t2.get():.2f}", width=4)
    ttk.Label(lf_thresh, text="t2").grid(row=2, column=0, padx=(0, 4), sticky="w")
    lbl_t2.grid(row=2, column=1, padx=(0, 6), sticky="e")
    s_t2 = ttk.Scale(lf_thresh, from_=0.0, to=1.0, orient="horizontal", variable=var_t2)
    s_t2.grid(row=3, column=0, columnspan=2, padx=(0, 0), pady=(2, 0), sticky="ew")

    var_small_to_gray = tk.BooleanVar(value=True)
    ttk.Checkbutton(ctrl, text="small→gray", variable=var_small_to_gray).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 2))
    ttk.Label(ctrl, text="N").grid(row=2, column=0, padx=(0, 4), sticky="w")
    var_small_N = tk.StringVar(value="900")
    ttk.Entry(ctrl, textvariable=var_small_N, width=6).grid(row=2, column=1, sticky="w")
    btn_update = ttk.Button(ctrl, text="Update")
    btn_update.grid(row=3, column=0, sticky="w", pady=(8, 4))

    lbl_status = ttk.Label(ctrl, text=f"ROI: ({x0},{y0},{x1},{y1})")
    lbl_status.grid(row=4, column=0, sticky="w", pady=(4, 0))

    btn_cancel = ttk.Button(ctrl, text="Cancel")
    btn_cancel.grid(row=5, column=0, sticky="w", pady=(12, 2))
    btn_save = ttk.Button(ctrl, text="Save")
    btn_save.grid(row=6, column=0, sticky="w", pady=(0, 0))

    # right: image panels (only two)
    panes = ttk.Frame(frm)
    panes.grid(row=1, column=1, pady=(10, 0), sticky="nsew")

    lbl_right = ttk.Label(panes)
    lbl_right.grid(row=0, column=0, sticky="n")

    lbl_right_overlay = ttk.Label(panes)
    lbl_right_overlay.grid(row=1, column=0, pady=(8, 0), sticky="n")

    state = {"ph_right": None, "ph_right_overlay": None, "sketch_u8": None}
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
            t1 = float(var_t1.get())
            t2 = float(var_t2.get())
            if not (0.0 < t1 < t2 < 1.0):
                raise ValueError("need 0 < t1 < t2 < 1")

            lbl_t1.configure(text=f"{t1:.2f}")
            lbl_t2.configure(text=f"{t2:.2f}")

            gray_roi = gray[y0:y1, x0:x1]
            _, sketch_u8 = sketch_three_bins(gray_roi, t1=float(t1), t2=float(t2))

            if var_small_to_gray.get():
                n = int(float(var_small_N.get().strip()))
                if n <= 0:
                    raise ValueError("N must be positive")
                sketch_u8 = small_components_to_gray(sketch_u8, min_area=n)

            state["sketch_u8"] = sketch_u8

            try:
                screen_w = root.winfo_screenwidth()
                screen_h = root.winfo_screenheight()
                max_side = int(min(screen_w * 0.5, screen_h * 0.75))
            except Exception:
                max_side = 750

            ph_right = to_photo_u8(sketch_u8, max_side=max_side)
            roi_rgb = img_rgb[y0:y1, x0:x1]
            ph_right_overlay = _roi_overlay_photo(roi_rgb, sketch_u8, max_side=max_side, alpha=0.35)

            state["ph_right"] = ph_right
            state["ph_right_overlay"] = ph_right_overlay

            lbl_right.configure(image=ph_right)
            lbl_right_overlay.configure(image=ph_right_overlay)

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
            chosen["t1"] = float(var_t1.get())
            chosen["t2"] = float(var_t2.get())
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

    _pending = {"id": None}

    def _schedule_render() -> None:
        # throttle to avoid hammering render while dragging
        if _pending["id"] is not None:
            try:
                root.after_cancel(_pending["id"])
            except Exception:
                pass
        _pending["id"] = root.after(80, lambda: (render(), _pending.__setitem__("id", None)))

    def _on_t1_change(_val: str) -> None:
        try:
            lbl_t1.configure(text=f"{float(var_t1.get()):.2f}")
        except Exception:
            pass
        _schedule_render()

    def _on_t2_change(_val: str) -> None:
        try:
            lbl_t2.configure(text=f"{float(var_t2.get()):.2f}")
        except Exception:
            pass
        _schedule_render()

    s_t1.configure(command=_on_t1_change)
    s_t2.configure(command=_on_t2_change)

    btn_update.configure(command=render)
    btn_cancel.configure(command=on_cancel)
    btn_save.configure(command=on_save)

    root.bind("<Return>", lambda _e: on_save())
    root.bind("<Escape>", lambda _e: on_cancel())
    render()
    root.mainloop()

    if chosen.get("done", False):
        return chosen
    return None