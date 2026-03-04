import numpy as np
import tkinter as tk
from tkinter import ttk
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
    roi_state = {"x0": 0, "y0": 0, "x1": 0, "y1": 0, "set": False}
    drag = {
        "active": False,
        "mode": "new",  # new|move|resize_*
        "x0": 0,
        "y0": 0,
        "roi0": (0, 0, 0, 0),
        "hit": None,
    }
    HIT_R = 12  # hit radius in image pixels for grabbing edges/corners

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

    def _norm_roi(x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int]:
        W = int(img_rgb.shape[1])
        H = int(img_rgb.shape[0])
        x0 = max(0, min(W - 1, int(x0)))
        x1 = max(0, min(W - 1, int(x1)))
        y0 = max(0, min(H - 1, int(y0)))
        y1 = max(0, min(H - 1, int(y1)))
        if x1 <= x0:
            x1 = min(W - 1, x0 + 1)
        if y1 <= y0:
            y1 = min(H - 1, y0 + 1)
        return int(x0), int(y0), int(x1), int(y1)

    def _hit_test_roi(x: int, y: int, roi: tuple[int, int, int, int]) -> str | None:
        x0, y0, x1, y1 = roi
        # corners
        if abs(x - x0) <= HIT_R and abs(y - y0) <= HIT_R:
            return "resize_tl"
        if abs(x - x1) <= HIT_R and abs(y - y0) <= HIT_R:
            return "resize_tr"
        if abs(x - x0) <= HIT_R and abs(y - y1) <= HIT_R:
            return "resize_bl"
        if abs(x - x1) <= HIT_R and abs(y - y1) <= HIT_R:
            return "resize_br"
        # edges
        if abs(x - x0) <= HIT_R and (y0 - HIT_R) <= y <= (y1 + HIT_R):
            return "resize_l"
        if abs(x - x1) <= HIT_R and (y0 - HIT_R) <= y <= (y1 + HIT_R):
            return "resize_r"
        if abs(y - y0) <= HIT_R and (x0 - HIT_R) <= x <= (x1 + HIT_R):
            return "resize_t"
        if abs(y - y1) <= HIT_R and (x0 - HIT_R) <= x <= (x1 + HIT_R):
            return "resize_b"
        # inside
        if x0 <= x <= x1 and y0 <= y <= y1:
            return "move"
        return None

    def render() -> None:
        W = int(img_rgb.shape[1])
        H = int(img_rgb.shape[0])

        max_side = 900
        s0 = max(W, H)
        scale = 1.0 if s0 <= max_side else (max_side / float(s0))
        state["scale"] = float(scale)
        state["W"] = int(round(W * scale))
        state["H"] = int(round(H * scale))

        roi = None
        if bool(roi_state.get("set", False)):
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

        if roi is None:
            lbl_info.configure(text="No ROI. Drag on image to create. Enter = Save")
        else:
            lbl_info.configure(text=f"ROI=({roi[0]},{roi[1]},{roi[2]},{roi[3]}) | Enter = Save")

    def on_press(ev) -> None:
        x, y = _event_to_img_xy(ev)
        hit = None
        if bool(roi_state.get("set", False)):
            roi = (int(roi_state["x0"]), int(roi_state["y0"]), int(roi_state["x1"]), int(roi_state["y1"]))
            hit = _hit_test_roi(x, y, roi)
        else:
            roi = (int(x), int(y), int(x) + 1, int(y) + 1)

        drag["active"] = True
        drag["x0"], drag["y0"] = int(x), int(y)
        drag["roi0"] = roi
        drag["hit"] = hit

        if hit is None:
            drag["mode"] = "new"
            # start a fresh tiny roi at click; will expand on motion
            roi_state["x0"], roi_state["y0"], roi_state["x1"], roi_state["y1"] = int(x), int(y), int(x) + 1, int(y) + 1
            roi_state["set"] = True
        else:
            drag["mode"] = hit
            roi_state["set"] = True

        render()

    def on_motion(ev) -> None:
        if not drag.get("active", False):
            return
        x, y = _event_to_img_xy(ev)
        x = int(x)
        y = int(y)

        mode = str(drag.get("mode", "new"))
        fallback_roi = (roi_state["x0"], roi_state["y0"], roi_state["x1"], roi_state["y1"])
        x0, y0, x1, y1 = map(int, drag.get("roi0", fallback_roi))
        dx = x - int(drag.get("x0", x))
        dy = y - int(drag.get("y0", y))

        if mode == "new":
            nx0 = min(int(drag["x0"]), x)
            ny0 = min(int(drag["y0"]), y)
            nx1 = max(int(drag["x0"]), x)
            ny1 = max(int(drag["y0"]), y)
            nx0, ny0, nx1, ny1 = _norm_roi(nx0, ny0, nx1, ny1)
        elif mode == "move":
            nx0, ny0, nx1, ny1 = _norm_roi(x0 + dx, y0 + dy, x1 + dx, y1 + dy)
        elif mode == "resize_l":
            nx0, ny0, nx1, ny1 = _norm_roi(x, y0, x1, y1)
        elif mode == "resize_r":
            nx0, ny0, nx1, ny1 = _norm_roi(x0, y0, x, y1)
        elif mode == "resize_t":
            nx0, ny0, nx1, ny1 = _norm_roi(x0, y, x1, y1)
        elif mode == "resize_b":
            nx0, ny0, nx1, ny1 = _norm_roi(x0, y0, x1, y)
        elif mode == "resize_tl":
            nx0, ny0, nx1, ny1 = _norm_roi(x, y, x1, y1)
        elif mode == "resize_tr":
            nx0, ny0, nx1, ny1 = _norm_roi(x0, y, x, y1)
        elif mode == "resize_bl":
            nx0, ny0, nx1, ny1 = _norm_roi(x, y0, x1, y)
        elif mode == "resize_br":
            nx0, ny0, nx1, ny1 = _norm_roi(x0, y0, x, y)
        else:
            nx0, ny0, nx1, ny1 = _norm_roi(roi_state["x0"], roi_state["y0"], roi_state["x1"], roi_state["y1"])

        roi_state["x0"], roi_state["y0"], roi_state["x1"], roi_state["y1"] = int(nx0), int(ny0), int(nx1), int(ny1)
        render()

    def on_release(ev) -> None:
        if not drag.get("active", False):
            return
        drag["active"] = False
        # finalize one last time
        on_motion(ev)

    def on_save() -> None:
        if not bool(roi_state.get("set", False)):
            lbl_info.configure(text="No ROI selected. Drag to create ROI first.")
            return
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
    lbl_img.bind("<B1-Motion>", on_motion)

    root.bind("<Return>", lambda _e: on_save())
    root.bind("<Escape>", lambda _e: on_cancel())

    render()
    root.mainloop()

    if chosen.get("done", False):
        return chosen
    return None