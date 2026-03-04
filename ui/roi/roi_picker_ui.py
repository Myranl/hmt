import numpy as np
import tkinter as tk
from tkinter import ttk
from ui.review_ui import review_and_maybe_edit  # type: ignore
from segmentation.postprocess import smooth_fill_mask  # type: ignore
from ui.tk_utils import left_panel_photo
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

        ph = left_panel_photo(
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