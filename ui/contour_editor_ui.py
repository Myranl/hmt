import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


def edit_contour_ui(
    brain_mask: np.ndarray,
    original_image_rgb: np.ndarray,
    *,
    window: str = "Contour Editor",
) -> tuple[np.ndarray, dict]:
    """UI for editing non-complete brain contours. Opens a Tk window; blocks until user accepts or skips."""

    h, w = brain_mask.shape[:2]
    # Build display: image + green mask outline
    vis_bgr = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR)
    m_u8 = (brain_mask.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_bgr, cnts, -1, (0, 255, 0), 2)
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

    result_mask = brain_mask.copy()
    result_params: dict = {"accepted": False, "edited": False, "correction_type": "none"}

    # Use Tk() so the window appears even after the previous UI (brain_outline) destroyed its root
    root = tk.Tk()
    root.title(window)
    root.minsize(800, 600)
    root.geometry("1100x750")

    frm = ttk.Frame(root, padding=8)
    frm.pack(fill="both", expand=True)
    frm.columnconfigure(0, weight=1)
    frm.rowconfigure(0, weight=1)

    frm_canvas = ttk.Frame(frm)
    frm_canvas.grid(row=0, column=0, sticky="nsew")
    scroll_y = ttk.Scrollbar(frm_canvas)
    scroll_x = ttk.Scrollbar(frm_canvas, orient=tk.HORIZONTAL)
    canvas = tk.Canvas(frm_canvas, highlightthickness=0, bg="#111")
    canvas.grid(row=0, column=0, sticky="nsew")
    scroll_y.grid(row=0, column=1, sticky="ns")
    scroll_x.grid(row=1, column=0, sticky="ew")
    canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
    scroll_y.configure(command=canvas.yview)
    scroll_x.configure(command=canvas.xview)
    frm_canvas.columnconfigure(0, weight=1)
    frm_canvas.rowconfigure(0, weight=1)

    ctrl = ttk.Frame(frm, width=260)
    ctrl.grid(row=0, column=1, sticky="ns", padx=(12, 0))
    ctrl.grid_propagate(False)

    ttk.Label(ctrl, text="Contour Editor", font=("TkDefaultFont", 14, "bold")).grid(
        row=0, column=0, sticky="w", pady=(0, 6)
    )
    ttk.Label(
        ctrl,
        text="Correct non-complete brain contour (Cut break / Bridge gap).\nFor now: Accept to continue, Skip to keep current mask.",
        justify="left",
    ).grid(row=1, column=0, sticky="w", pady=(0, 12))

    def do_accept() -> None:
        result_params["accepted"] = True
        root.destroy()

    def do_skip() -> None:
        result_params["accepted"] = True  # continue pipeline
        result_params["edited"] = False
        root.destroy()

    ttk.Button(ctrl, text="Accept (continue)", command=do_accept).grid(row=2, column=0, sticky="ew", pady=(0, 6))
    ttk.Button(ctrl, text="Skip (keep mask)", command=do_skip).grid(row=3, column=0, sticky="ew")

    max_canvas_w, max_canvas_h = 900, 700
    scale = min(1.0, max_canvas_w / float(w), max_canvas_h / float(h))
    disp_w = int(round(w * scale))
    disp_h = int(round(h * scale))
    if scale < 1.0:
        vis_disp = cv2.resize(vis_rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    else:
        vis_disp = vis_rgb
    pil = Image.fromarray(vis_disp)
    tk_img = ImageTk.PhotoImage(pil)
    canvas.configure(width=min(disp_w, max_canvas_w), height=min(disp_h, max_canvas_h))
    canvas.configure(scrollregion=(0, 0, disp_w, disp_h))
    canvas.create_image(0, 0, anchor="nw", image=tk_img)
    # Keep reference so image is not garbage-collected
    canvas._photo = tk_img

    root.wait_window()
    return result_mask, result_params
