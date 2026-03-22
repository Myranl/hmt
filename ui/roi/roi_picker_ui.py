from __future__ import annotations

import sys
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

import customtkinter as ctk  # type: ignore[import-untyped]

from ui.common.theme import setup_theme, get_base_font, get_small_muted_font
from ui.common.screen_layout import layout_screen_wh
from ui.common.widgets import (
    create_card_frame,
    create_primary_button,
    create_secondary_button,
)


def _render_roi_display_rgb(
    img_rgb: np.ndarray,
    out_w: int,
    out_h: int,
    *,
    grid_on: bool,
    step: int,
    roi: tuple[int, int, int, int] | None,
) -> np.ndarray:
    """Resize RGB image to out_w×out_h and draw grid / ROI like ``left_panel_photo``."""
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("img_rgb must be H×W×3 uint8 RGB")
    h0, w0 = int(img_rgb.shape[0]), int(img_rgb.shape[1])
    out_w = max(1, int(out_w))
    out_h = max(1, int(out_h))
    im0 = Image.fromarray(img_rgb)
    im = im0.resize((out_w, out_h), resample=Image.Resampling.BILINEAR)
    scale_x = out_w / float(w0)
    scale_y = out_h / float(h0)

    if grid_on or roi is not None:
        dr = ImageDraw.Draw(im)

        def _text_box(x: int, y: int, text: str) -> None:
            tw = int(dr.textlength(text))
            th = 12
            pad = 3
            x0b = max(0, x - pad)
            y0b = max(0, y - pad)
            x1b = min(out_w - 1, int(x + tw) + pad)
            y1b = min(out_h - 1, y + th + pad)
            dr.rectangle([x0b, y0b, x1b, y1b], fill=0)
            dr.text((x, y), text, fill=(220, 220, 220))

        if grid_on:
            step0 = max(10, int(step))
            for x0 in range(0, w0, step0):
                x = int(round(x0 * scale_x))
                dr.line([(x, 0), (x, out_h)], fill=(220, 220, 220), width=2)
            for y0 in range(0, h0, step0):
                y = int(round(y0 * scale_y))
                dr.line([(0, y), (out_w, y)], fill=(220, 220, 220), width=2)
            for x0 in range(0, w0, step0):
                x = int(round(x0 * scale_x))
                _text_box(x + 4, 4, str(x0))
            for y0 in range(0, h0, step0):
                y = int(round(y0 * scale_y))
                _text_box(4, y + 4, str(y0))

        if roi is not None:
            x0, y0, x1, y1 = roi
            xa = int(round(x0 * scale_x))
            ya = int(round(y0 * scale_y))
            xb = int(round(x1 * scale_x))
            yb = int(round(y1 * scale_y))
            dr.rectangle([xa, ya, xb, yb], outline=(125, 125, 255), width=3)

    return np.asarray(im, dtype=np.uint8)


def run_roi_ui(*, img_rgb: np.ndarray) -> dict | None:
    setup_theme()
    root = ctk.CTk()
    root.title("Step 1/2: pick ROI")
    if sys.platform != "win32":
        try:
            root.grab_set()
        except Exception:
            pass

    root.configure(fg_color="white")
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=0)

    img_h, img_w = int(img_rgb.shape[0]), int(img_rgb.shape[1])
    screen_w, screen_h = layout_screen_wh(root)

    HDR_H = 56
    FTR_ROW = 44
    GRID_PAD_X = 36
    GRID_PAD_Y = 32
    max_win_w = min(1400, int(screen_w * 0.92))
    max_body_h = int(screen_h * 0.86) - HDR_H - FTR_ROW - GRID_PAD_Y
    target_canvas_w = max(480, max_win_w - GRID_PAD_X)
    fit_scale = target_canvas_w / float(max(img_w, 1))
    nat_canvas_h = int(round(img_h * fit_scale))
    SCROLL_X_H = 22
    initial_canvas_h = max(80, min(max_body_h, nat_canvas_h))
    viewport_w = int(target_canvas_w)
    canvas_viewport_h_max = int(initial_canvas_h)

    zoom_mul = [1.0]
    disp_scale = [fit_scale * zoom_mul[0]]
    disp_w = [max(1, int(round(img_w * disp_scale[0])))]
    disp_h = [max(1, int(round(img_h * disp_scale[0])))]

    var_grid_on = tk.BooleanVar(value=True)
    var_grid_step = tk.StringVar(value="200")

    chosen: dict = {"done": False}
    roi_state = {"x0": 0, "y0": 0, "x1": 0, "y1": 0, "set": False}
    drag = {
        "active": False,
        "mode": "new",
        "x0": 0,
        "y0": 0,
        "roi0": (0, 0, 0, 0),
        "hit": None,
    }
    HIT_R = 12
    space_held = [False]
    pan_drag = [False]
    tk_img_ref: dict[str, ImageTk.PhotoImage] = {}
    canvas_img_id: list[int] = []

    def _norm_roi(x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int]:
        W, H = img_w, img_h
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
        if abs(x - x0) <= HIT_R and abs(y - y0) <= HIT_R:
            return "resize_tl"
        if abs(x - x1) <= HIT_R and abs(y - y0) <= HIT_R:
            return "resize_tr"
        if abs(x - x0) <= HIT_R and abs(y - y1) <= HIT_R:
            return "resize_bl"
        if abs(x - x1) <= HIT_R and abs(y - y1) <= HIT_R:
            return "resize_br"
        if abs(x - x0) <= HIT_R and (y0 - HIT_R) <= y <= (y1 + HIT_R):
            return "resize_l"
        if abs(x - x1) <= HIT_R and (y0 - HIT_R) <= y <= (y1 + HIT_R):
            return "resize_r"
        if abs(y - y0) <= HIT_R and (x0 - HIT_R) <= x <= (x1 + HIT_R):
            return "resize_t"
        if abs(y - y1) <= HIT_R and (x0 - HIT_R) <= x <= (x1 + HIT_R):
            return "resize_b"
        if x0 <= x <= x1 and y0 <= y <= y1:
            return "move"
        return None

    # -------- Header (one row: tools + single status line, like hippocampus) --------
    header = ctk.CTkFrame(root, fg_color="transparent")
    header.grid(row=0, column=0, sticky="ew", padx=18, pady=(12, 6))
    header.grid_columnconfigure(5, weight=1)

    ctk.CTkLabel(
        header,
        text="Pick ROI",
        font=ctk.CTkFont(size=17, weight="bold"),
    ).grid(row=0, column=0, sticky="w")

    ctk.CTkLabel(
        header,
        text="ROI",
        font=ctk.CTkFont(size=12, weight="bold"),
        corner_radius=8,
        fg_color=("#cfe8f6", "#2b4a5e"),
        text_color=("gray20", "gray90"),
        width=56,
        height=28,
    ).grid(row=0, column=1, sticky="w", padx=(10, 0))

    cb_grid = ctk.CTkCheckBox(
        header,
        text="Grid",
        variable=var_grid_on,
        font=get_base_font(),
        command=lambda: _refresh(),
    )
    cb_grid.grid(row=0, column=2, sticky="w", padx=(14, 0))

    ctk.CTkLabel(header, text="step", font=get_small_muted_font()).grid(
        row=0, column=3, sticky="e", padx=(12, 4)
    )
    ent_step = ctk.CTkEntry(header, textvariable=var_grid_step, width=64, height=28)
    ent_step.grid(row=0, column=4, sticky="w")

    hdr_status = ctk.CTkLabel(
        header,
        text="",
        font=get_base_font(),
        text_color=("gray35", "gray70"),
        anchor="e",
        justify="right",
    )
    hdr_status.grid(row=0, column=5, sticky="e")

    # -------- Canvas (dark card) --------
    img_card = create_card_frame(root, fg_color=("#2a2a2a", "#1a1a1a"))
    img_card.grid(row=1, column=0, sticky="new", padx=18, pady=(0, 0))
    img_card.columnconfigure(0, weight=1)
    img_card.rowconfigure(0, weight=0)

    canvas_holder = tk.Frame(img_card, bg="#2a2a2a")
    canvas_holder.grid(row=0, column=0, sticky="new", padx=2, pady=2)
    canvas_holder.columnconfigure(0, weight=1)
    canvas_holder.rowconfigure(0, weight=0)
    canvas_holder.rowconfigure(1, weight=0)

    scroll_y = ttk.Scrollbar(canvas_holder)
    scroll_x = ttk.Scrollbar(canvas_holder, orient=tk.HORIZONTAL)
    canvas = tk.Canvas(canvas_holder, highlightthickness=0, bg="#252525")
    canvas.grid(row=0, column=0, sticky="nsew")
    scroll_y.grid(row=0, column=1, sticky="ns")
    scroll_x.grid(row=1, column=0, sticky="ew")
    canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
    scroll_y.configure(command=canvas.yview)
    scroll_x.configure(command=canvas.xview)

    def _on_canvas_holder_configure(_ev) -> None:
        # Resize canvas to fill its container (canvas_holder)
        cw = canvas_holder.winfo_width()
        ch = canvas_holder.winfo_height() - SCROLL_X_H
        if cw > 1 and ch > 1:
            canvas.configure(width=cw, height=ch)
            _refresh()

    canvas_holder.bind("<Configure>", _on_canvas_holder_configure)

    # -------- Footer --------
    footer = ctk.CTkFrame(root, fg_color="transparent")
    footer.grid(row=2, column=0, sticky="ew", padx=18, pady=(8, 12))
    footer.grid_columnconfigure(2, weight=1)
    zoom_var = tk.StringVar(value="Zoom: 100%")
    ctk.CTkLabel(footer, textvariable=zoom_var, font=get_small_muted_font(), text_color="gray50").grid(
        row=0, column=0, sticky="w"
    )
    ctk.CTkLabel(
        footer,
        text="Scroll to zoom · Space + drag to pan · Esc = skip image",
        font=get_small_muted_font(),
        text_color="gray50",
    ).grid(row=0, column=3, sticky="e", padx=(0, 6))

    skip_w, save_w, action_h = 118, 100, 30
    skip_btn = create_secondary_button(
        footer, text="Skip Image", command=lambda: None, width=skip_w, height=action_h
    )
    skip_btn.grid(row=0, column=1, sticky="w", padx=(10, 0))
    save_btn = create_primary_button(
        footer, text="Save ROI", command=lambda: None, width=save_w, height=action_h
    )
    save_btn.grid(row=0, column=4, sticky="e")

    def _recompute_scale_from_zoom() -> None:
        disp_scale[0] = fit_scale * zoom_mul[0]
        disp_w[0] = max(1, int(round(img_w * disp_scale[0])))
        disp_h[0] = max(1, int(round(img_h * disp_scale[0])))
        pct = int(round(zoom_mul[0] * 100))
        zoom_var.set(f"Zoom: {pct}%")

    def _header_text() -> str:
        g = "on" if var_grid_on.get() else "off"
        try:
            st = int(float(var_grid_step.get().strip()))
        except Exception:
            st = 200
        tail = " · Drag to draw · handles · Enter = save"
        if not bool(roi_state.get("set", False)):
            return f"No selection · grid {g} · step {st}{tail}"
        r = (int(roi_state["x0"]), int(roi_state["y0"]), int(roi_state["x1"]), int(roi_state["y1"]))
        return f"({r[0]}, {r[1]}) → ({r[2]}, {r[3]}) · grid {g} · step {st}{tail}"

    def _refresh() -> None:
        _recompute_scale_from_zoom()
        dw, dh = disp_w[0], disp_h[0]
        roi = None
        if bool(roi_state.get("set", False)):
            roi = (int(roi_state["x0"]), int(roi_state["y0"]), int(roi_state["x1"]), int(roi_state["y1"]))
        try:
            step = int(float(var_grid_step.get().strip()))
        except Exception:
            step = 200
        rgb = _render_roi_display_rgb(
            img_rgb,
            dw,
            dh,
            grid_on=bool(var_grid_on.get()),
            step=step,
            roi=roi,
        )
        hdr_status.configure(text=_header_text())
        cw = int(min(viewport_w, max(1, dw)))
        ch = int(min(canvas_viewport_h_max, max(1, dh)))
        canvas.configure(width=cw, height=ch)
        pil = Image.fromarray(rgb)
        tk_img = ImageTk.PhotoImage(pil, master=root)
        tk_img_ref["img"] = tk_img
        if not canvas_img_id:
            canvas_img_id.append(canvas.create_image(0, 0, anchor="nw", image=tk_img))
        else:
            canvas.itemconfigure(canvas_img_id[0], image=tk_img)
        canvas.configure(scrollregion=(0, 0, dw, dh))

    def _canvas_xy(ev) -> tuple[int, int] | None:
        cx, cy = canvas.canvasx(ev.x), canvas.canvasy(ev.y)
        dw, dh = disp_w[0], disp_h[0]
        if cx < 0 or cy < 0 or cx >= dw or cy >= dh:
            return None
        ix = int(round(cx * img_w / float(dw)))
        iy = int(round(cy * img_h / float(dh)))
        return min(max(ix, 0), img_w - 1), min(max(iy, 0), img_h - 1)

    def _apply_zoom_at(ev, direction: int) -> None:
        canvas.update_idletasks()
        mx = canvas.canvasx(ev.x)
        my = canvas.canvasy(ev.y)
        dw0, dh0 = disp_w[0], disp_h[0]
        if dw0 <= 0 or dh0 <= 0:
            return
        fx = mx / float(dw0)
        fy = my / float(dh0)
        vis_w = max(1, canvas.winfo_width())
        vis_h = max(1, canvas.winfo_height())
        step = 1.12 if direction > 0 else 1.0 / 1.12
        zoom_mul[0] = min(8.0, max(0.2, zoom_mul[0] * step))
        _refresh()
        dw1, dh1 = disp_w[0], disp_h[0]
        n_scroll_x = fx * dw1 - ev.x
        n_scroll_y = fy * dh1 - ev.y
        sx1 = max(0.0, min(max(0.0, dw1 - vis_w), n_scroll_x))
        sy1 = max(0.0, min(max(0.0, dh1 - vis_h), n_scroll_y))
        frac_x = sx1 / max(1e-6, dw1 - vis_w) if dw1 > vis_w else 0.0
        frac_y = sy1 / max(1e-6, dh1 - vis_h) if dh1 > vis_h else 0.0
        canvas.xview_moveto(frac_x)
        canvas.yview_moveto(frac_y)

    def _on_mousewheel(ev) -> None:
        d = int(getattr(ev, "delta", 0) or 0)
        if d == 0:
            return
        direction = 1 if d > 0 else -1
        _apply_zoom_at(ev, direction)

    def _on_wheel_linux_up(_ev) -> None:
        class E:
            x, y, delta = _ev.x, _ev.y, 120

        _on_mousewheel(E())

    def _on_wheel_linux_down(_ev) -> None:
        class E:
            x, y, delta = _ev.x, _ev.y, -120

        _on_mousewheel(E())

    canvas.bind("<MouseWheel>", _on_mousewheel)
    canvas.bind("<Button-4>", _on_wheel_linux_up)
    canvas.bind("<Button-5>", _on_wheel_linux_down)
    canvas.bind("<Enter>", lambda _e: canvas.focus_set())

    def on_press(ev) -> None:
        if space_held[0]:
            pan_drag[0] = True
            canvas.scan_mark(int(ev.x), int(ev.y))
            return
        xy = _canvas_xy(ev)
        if xy is None:
            return
        x, y = xy
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
            roi_state["x0"], roi_state["y0"], roi_state["x1"], roi_state["y1"] = int(x), int(y), int(x) + 1, int(y) + 1
            roi_state["set"] = True
        else:
            drag["mode"] = hit
            roi_state["set"] = True
        _refresh()

    def on_motion(ev) -> None:
        if pan_drag[0]:
            canvas.scan_dragto(int(ev.x), int(ev.y), gain=1)
            return
        if not drag.get("active", False):
            return
        xy = _canvas_xy(ev)
        if xy is None:
            return
        x, y = int(xy[0]), int(xy[1])
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
        _refresh()

    def on_release(ev) -> None:
        if pan_drag[0]:
            pan_drag[0] = False
            return
        if not drag.get("active", False):
            return
        drag["active"] = False
        on_motion(ev)

    def on_save() -> None:
        if not bool(roi_state.get("set", False)):
            hdr_status.configure(text="No ROI — drag on the image to create one first.")
            root.after(2200, _refresh)
            return
        try:
            chosen["grid_on"] = bool(var_grid_on.get())
            chosen["grid_step"] = int(float(var_grid_step.get().strip()))
            chosen["x0"] = int(roi_state["x0"])
            chosen["y0"] = int(roi_state["y0"])
            chosen["x1"] = int(roi_state["x1"])
            chosen["y1"] = int(roi_state["y1"])
        except Exception as e:
            hdr_status.configure(text=f"Error: {e}")
            root.after(2200, _refresh)
            return
        chosen["done"] = True
        try:
            root.grab_release()
        except Exception:
            pass
        root.after(10, root.destroy)

    def on_cancel() -> None:
        chosen["done"] = False
        try:
            root.grab_release()
        except Exception:
            pass
        root.after(10, root.destroy)

    skip_btn.configure(command=on_cancel)
    save_btn.configure(command=on_save)

    def on_key(ev) -> None:
        k = (ev.keysym or "").lower()
        if k in ("return", "kp_enter"):
            on_save()

    def on_space_press(_ev) -> None:
        space_held[0] = True
        try:
            canvas.configure(cursor="fleur")
        except Exception:
            pass

    def on_space_release(_ev) -> None:
        space_held[0] = False
        pan_drag[0] = False
        try:
            canvas.configure(cursor="")
        except Exception:
            pass

    ent_step.bind("<FocusOut>", lambda _e: _refresh())

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<ButtonRelease-1>", on_release)
    canvas.bind("<B1-Motion>", on_motion)

    root.bind("<Key>", on_key)
    root.bind("<Escape>", lambda _e: on_cancel())
    root.bind("<KeyPress-space>", on_space_press)
    root.bind("<KeyRelease-space>", on_space_release)

    total_w = viewport_w + GRID_PAD_X
    total_h = HDR_H + initial_canvas_h + SCROLL_X_H + FTR_ROW + GRID_PAD_Y + 12
    total_w = min(total_w, int(screen_w * 0.98))
    total_h = min(total_h, int(screen_h * 0.94))
    root.geometry(f"{total_w}x{total_h}")
    root.update_idletasks()
    rw = root.winfo_width()
    rh = root.winfo_height()
    # For debugging display issues on Windows: open at top-right instead of centered.
    x0 = max(0, screen_w - rw - 50) # 50px offset from right edge
    y0 = max(0, 50) # 50px offset from top edge
    root.geometry(f"{rw}x{rh}+{x0}+{y0}")

    root.minsize(880, 520)

    _refresh()
    root.update_idletasks()
    root.deiconify()
    root.lift()
    try:
        root.focus_force()
    except Exception:
        pass
    root.mainloop()

    if chosen.get("done", False):
        return chosen
    return None
