from __future__ import annotations

import tkinter as tk

import cv2
import customtkinter as ctk  # type: ignore[import-untyped]
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from preproc.quantize import sketch_three_bins, small_components_to_gray
from ui.common.theme import setup_theme, get_base_font, get_small_muted_font
from ui.common.widgets import (
    create_card_frame,
    create_primary_button,
    create_secondary_button,
    create_toolbar_button,
)


def _fit_roi_to_box(roi_w: int, roi_h: int, box_w: int, box_h: int, *, max_upscale: float = 5.0) -> tuple[int, int]:
    """Integer size that fits ROI aspect into box_w×box_h (may upscale up to max_upscale)."""
    rw = max(1, int(roi_w))
    rh = max(1, int(roi_h))
    bw = max(32, int(box_w))
    bh = max(32, int(box_h))
    scale = min(bw / float(rw), bh / float(rh))
    scale = min(scale, max_upscale)
    scale = max(scale, 1e-6)
    tw = max(1, int(round(rw * scale)))
    th = max(1, int(round(rh * scale)))
    return tw, th


def _blend_edge(base: np.ndarray, mask: np.ndarray, rgb: tuple[float, float, float], alpha: float) -> None:
    """In-place: where mask>0, blend base toward rgb."""
    m = mask.astype(bool)
    if not np.any(m):
        return
    r, g, b = rgb
    base[m, 0] = (1 - alpha) * base[m, 0] + alpha * r
    base[m, 1] = (1 - alpha) * base[m, 1] + alpha * g
    base[m, 2] = (1 - alpha) * base[m, 2] + alpha * b


def _annotate_preview(
    img_u8: np.ndarray,
    *,
    title: str,
    legend_lines: list[tuple[str, tuple[int, int, int]]] | None = None,
) -> np.ndarray:
    """Draw top-left title pill and optional bottom-right legend on RGB uint8 image."""
    im = Image.fromarray(img_u8.copy()).convert("RGBA")
    dr = ImageDraw.Draw(im, "RGBA")
    w, h = im.size
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
        font_sm = ImageFont.truetype("DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    pad = 8
    tw, th = dr.textbbox((0, 0), title, font=font)[2:4]
    bx0, by0 = 10, 10
    bx1, by1 = bx0 + tw + pad * 2, by0 + th + pad * 2
    dr.rounded_rectangle((bx0, by0, bx1, by1), radius=8, fill=(40, 40, 40, 220))
    dr.text((bx0 + pad, by0 + pad), title, fill=(235, 235, 235, 255), font=font)

    if legend_lines:
        lines = [text for text, _ in legend_lines]
        max_w = max(dr.textbbox((0, 0), ln, font=font_sm)[2] for ln in lines)
        line_h = dr.textbbox((0, 0), "Hg", font=font_sm)[3] + 4
        box_h = line_h * len(lines) + pad * 2
        box_w = max_w + pad * 2 + 22
        lx0 = w - box_w - 10
        ly0 = h - box_h - 10
        dr.rounded_rectangle((lx0, ly0, w - 10, h - 10), radius=8, fill=(30, 30, 30, 230))
        y = ly0 + pad
        for text, col in legend_lines:
            dr.rectangle((lx0 + pad, y + 2, lx0 + pad + 10, y + 12), fill=col + (255,))
            dr.text((lx0 + pad + 14, y), text, fill=(240, 240, 240, 255), font=font_sm)
            y += line_h

    return np.array(im.convert("RGB"))


def run_bins_ui(
    *,
    gray: np.ndarray,
    img_rgb: np.ndarray,
    roi: tuple[int, int, int, int],
    grid_on: bool,
    grid_step: int,
    t1_init: float = 0.33,
    t2_init: float = 0.66,
) -> dict | None:
    """Tune 3-bin thresholds on a fixed ROI. Layout aligned with design mock (dual range, legends)."""
    T1_INIT = float(t1_init)
    T2_INIT = float(t2_init)
    x0, y0, x1, y1 = roi

    setup_theme()
    parent = tk._default_root
    if parent is not None and isinstance(parent, ctk.CTk):
        root = ctk.CTkToplevel(parent)
        root.transient(parent)
    else:
        root = ctk.CTk()
    root.title("3-bin sketch")
    root.configure(fg_color="#f5f5f2")
    root.minsize(980, 640)
    root.grid_columnconfigure(1, weight=1)
    root.grid_rowconfigure(1, weight=1)

    try:
        sw = int(root.winfo_screenwidth())
        sh = int(root.winfo_screenheight())
    except Exception:
        sw, sh = 1400, 900
    rw = max(980, min(sw - 80, 1320))
    rh = max(640, min(sh - 100, 900))
    rw = min(rw, sw - 24)
    rh = min(rh, sh - 24)
    root.geometry(f"{rw}x{rh}+{(sw - rw) // 2}+{(sh - rh) // 2}")

    base_font = get_base_font()
    small_font = get_small_muted_font()

    # --- Header ---
    header = ctk.CTkFrame(root, fg_color="transparent")
    header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=(14, 6))
    header.grid_columnconfigure(1, weight=1)
    ctk.CTkLabel(
        header,
        text="3-bin sketch",
        font=ctk.CTkFont(size=18, weight="bold"),
        text_color=("gray15", "gray90"),
    ).grid(row=0, column=0, sticky="w")
    grid_txt = f"Grid: {'on' if grid_on else 'off'}"
    if grid_on:
        grid_txt += f" · step {grid_step}"
    meta = f"ROI ({x0}, {y0}) → ({x1}, {y1})    {grid_txt}"
    ctk.CTkLabel(
        header,
        text=meta,
        font=small_font,
        text_color="gray45",
    ).grid(row=0, column=1, sticky="e")

    CTRL_W = 328
    ctrl = create_card_frame(root)
    ctrl.grid(row=1, column=0, sticky="ns", padx=(20, 12), pady=(0, 16))
    ctrl.grid_propagate(False)
    ctrl.configure(width=CTRL_W)
    ctrl.columnconfigure(0, weight=1)

    t_vals: list[float] = [T1_INIT, T2_INIT]
    var_t1_str = tk.StringVar(master=root, value=f"{T1_INIT:.2f}")
    var_t2_str = tk.StringVar(master=root, value=f"{T2_INIT:.2f}")

    r = 0
    ctk.CTkLabel(
        ctrl,
        text="THRESHOLDS",
        font=ctk.CTkFont(size=12, weight="bold"),
        text_color="gray35",
    ).grid(row=r, column=0, sticky="w", padx=16, pady=(16, 8))
    r += 1

    range_holder = ctk.CTkFrame(ctrl, fg_color="transparent")
    range_holder.grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 6))
    range_bar = tk.Canvas(
        range_holder,
        height=36,
        width=284,
        highlightthickness=0,
        bg="#f5f5f2",
        bd=0,
    )
    range_bar.pack(anchor="w")

    drag_state: dict[str, int | None] = {"which": None}

    def _x_to_t(xp: int) -> float:
        w = max(1, range_bar.winfo_width() or 284)
        pad = 14
        tw = max(1, w - 2 * pad)
        return float(np.clip((xp - pad) / tw, 0.02, 0.98))

    def _t_to_x(tv: float) -> tuple[int, int]:
        w = max(1, range_bar.winfo_width() or 284)
        pad = 14
        tw = max(1, w - 2 * pad)
        return int(round(pad + tv * tw)), w

    def redraw_range_bar() -> None:
        range_bar.delete("all")
        w = max(1, range_bar.winfo_width() or 284)
        h = 36
        pad = 14
        tw = w - 2 * pad
        mid_y = h // 2
        range_bar.create_rectangle(pad, mid_y - 4, pad + tw, mid_y + 4, fill="#dcdcd8", outline="#c8c8c4", width=1)
        x1, _ = _t_to_x(t_vals[0])
        x2, _ = _t_to_x(t_vals[1])
        if x2 > x1:
            range_bar.create_rectangle(x1, mid_y - 5, x2, mid_y + 5, fill="#3cb371", outline="")
        rh = 9
        for xi in (x1, x2):
            range_bar.create_oval(
                xi - rh,
                mid_y - rh,
                xi + rh,
                mid_y + rh,
                fill="#2d9d4f",
                outline="#ffffff",
                width=2,
            )

    def sync_strings() -> None:
        var_t1_str.set(f"{t_vals[0]:.2f}")
        var_t2_str.set(f"{t_vals[1]:.2f}")

    def on_range_press(ev: tk.Event) -> None:
        x1, _ = _t_to_x(t_vals[0])
        x2, _ = _t_to_x(t_vals[1])
        d1, d2 = abs(ev.x - x1), abs(ev.x - x2)
        drag_state["which"] = 0 if d1 <= d2 else 1

    def on_range_motion(ev: tk.Event) -> None:
        if drag_state["which"] is None:
            return
        v = _x_to_t(ev.x)
        eps = 0.02
        if drag_state["which"] == 0:
            t_vals[0] = min(v, t_vals[1] - eps)
        else:
            t_vals[1] = max(v, t_vals[0] + eps)
        sync_strings()
        redraw_range_bar()
        _schedule_render()

    def on_range_release(_ev: tk.Event) -> None:
        drag_state["which"] = None

    range_bar.bind("<ButtonPress-1>", on_range_press)
    range_bar.bind("<B1-Motion>", on_range_motion)
    range_bar.bind("<ButtonRelease-1>", on_range_release)

    r += 1
    legend_fr = ctk.CTkFrame(ctrl, fg_color="transparent")
    legend_fr.grid(row=r, column=0, sticky="w", padx=16, pady=(0, 10))
    for txt, col in (
        ("■  low (black)", "#1a1a1a"),
        ("■  mid (gray)", "#888888"),
        ("□  high (white)", "#cccccc"),
    ):
        lb = ctk.CTkLabel(legend_fr, text=txt, font=ctk.CTkFont(size=11), text_color=col)
        lb.pack(side="left", padx=(0, 12))
    r += 1

    row_vals = ctk.CTkFrame(ctrl, fg_color="transparent")
    row_vals.grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 12))
    row_vals.columnconfigure(0, weight=1)
    row_vals.columnconfigure(1, weight=1)

    ctk.CTkLabel(row_vals, text="t1 low/mid", font=ctk.CTkFont(size=11), text_color="gray40").grid(
        row=0, column=0, sticky="w"
    )
    ctk.CTkLabel(row_vals, text="t2 mid/high", font=ctk.CTkFont(size=11), text_color="gray40").grid(
        row=0, column=1, sticky="w", padx=(10, 0)
    )
    ent_t1 = ctk.CTkEntry(
        row_vals,
        textvariable=var_t1_str,
        width=120,
        height=32,
        font=base_font,
        text_color=("#1565c0", "#64b5f6"),
        justify="center",
    )
    ent_t1.grid(row=1, column=0, sticky="ew", pady=(4, 0))
    ent_t2 = ctk.CTkEntry(
        row_vals,
        textvariable=var_t2_str,
        width=120,
        height=32,
        font=base_font,
        text_color=("#c2410c", "#fb923c"),
        justify="center",
    )
    ent_t2.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=(4, 0))
    r += 1

    ctk.CTkLabel(
        ctrl,
        text="SMALL REGIONS",
        font=ctk.CTkFont(size=12, weight="bold"),
        text_color="gray35",
    ).grid(row=r, column=0, sticky="w", padx=16, pady=(10, 8))
    r += 1

    var_small_to_gray = tk.BooleanVar(master=root, value=True)
    ctk.CTkCheckBox(
        ctrl,
        text="Merge small -> gray",
        variable=var_small_to_gray,
        font=base_font,
    ).grid(row=r, column=0, sticky="w", padx=16, pady=(0, 8))
    r += 1

    row_n = ctk.CTkFrame(ctrl, fg_color="transparent")
    row_n.grid(row=r, column=0, sticky="w", padx=16, pady=(0, 12))
    ctk.CTkLabel(row_n, text="Min area (px)", font=base_font).pack(side="left", padx=(0, 10))
    var_small_N = tk.StringVar(master=root, value="900")
    ctk.CTkEntry(row_n, textvariable=var_small_N, width=100, height=30).pack(side="left")
    r += 1

    btn_refresh = create_toolbar_button(ctrl, text="Update preview", width=260, height=34)
    btn_refresh.grid(row=r, column=0, sticky="ew", padx=16, pady=(4, 10))
    r += 1

    status_row = ctk.CTkFrame(ctrl, fg_color="transparent")
    status_row.grid(row=r, column=0, sticky="w", padx=16, pady=(0, 16))
    dot = ctk.CTkLabel(status_row, text="●", font=ctk.CTkFont(size=14), text_color="#22c55e")
    dot.pack(side="left", padx=(0, 6))
    status_var = tk.StringVar(master=root, value="Preview updated")
    ctk.CTkLabel(
        status_row,
        textvariable=status_var,
        font=small_font,
        text_color="gray45",
        anchor="w",
    ).pack(side="left")
    r += 1

    img_card = create_card_frame(root, fg_color=("#2a2a2a", "#1a1a1a"))
    img_card.grid(row=1, column=1, sticky="nsew", padx=(0, 20), pady=(0, 16))
    img_card.rowconfigure(0, weight=1)
    img_card.columnconfigure(0, weight=1)

    inner = ctk.CTkFrame(img_card, fg_color="transparent")
    inner.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
    inner.rowconfigure(0, weight=1)
    inner.rowconfigure(1, weight=1)
    inner.columnconfigure(0, weight=1)

    pane_sk = tk.Frame(inner, bg="#1e1e1e", highlightthickness=0)
    pane_sk.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
    pane_ov = tk.Frame(inner, bg="#1e1e1e", highlightthickness=0)
    pane_ov.grid(row=1, column=0, sticky="nsew")

    lbl_right = tk.Label(pane_sk, bg="#1e1e1e", bd=0, highlightthickness=0)
    lbl_right.place(relx=0.5, rely=0.5, anchor="center")
    lbl_right_overlay = tk.Label(pane_ov, bg="#1e1e1e", bd=0, highlightthickness=0)
    lbl_right_overlay.place(relx=0.5, rely=0.5, anchor="center")

    footer = ctk.CTkFrame(root, fg_color="transparent")
    footer.grid(row=2, column=0, columnspan=2, sticky="ew", padx=20, pady=(0, 14))
    footer.grid_columnconfigure(0, weight=1)

    state = {"ph_right": None, "ph_right_overlay": None, "sketch_u8": None}
    chosen: dict = {"done": False}
    _resize_after: list[str | None] = [None]

    def _preview_target_wh() -> tuple[int, int]:
        """Size (tw, th) for both previews from current pane geometry."""
        root.update_idletasks()
        roi_w = max(1, x1 - x0)
        roi_h = max(1, y1 - y0)
        try:
            pw = int(pane_sk.winfo_width())
            ph = int(pane_sk.winfo_height())
        except Exception:
            pw, ph = 0, 0
        if pw < 80 or ph < 80:
            try:
                iw = max(300, int(inner.winfo_width()) - 8)
                ih = max(200, (int(inner.winfo_height()) - 16) // 2)
            except Exception:
                iw, ih = 520, 320
            bw = max(120, iw - 16)
            bh = max(120, ih - 16)
        else:
            bw = max(60, pw - 16)
            bh = max(60, ph - 16)
        return _fit_roi_to_box(roi_w, roi_h, bw, bh, max_upscale=5.0)

    def _read_t_from_entries() -> tuple[float, float]:
        t1 = float(var_t1_str.get().strip().replace(",", "."))
        t2 = float(var_t2_str.get().strip().replace(",", "."))
        if not (0.0 < t1 < t2 < 1.0):
            raise ValueError("Need 0 < t1 < t2 < 1")
        return t1, t2

    def apply_entries_to_state(*_a: object) -> None:
        try:
            t1, t2 = _read_t_from_entries()
            t_vals[0], t_vals[1] = t1, t2
            redraw_range_bar()
            render()
        except Exception as e:
            status_var.set(f"Error: {e}")

    ent_t1.bind("<Return>", apply_entries_to_state)
    ent_t1.bind("<FocusOut>", apply_entries_to_state)
    ent_t2.bind("<Return>", apply_entries_to_state)
    ent_t2.bind("<FocusOut>", apply_entries_to_state)

    def _roi_overlay_photo(
        roi_rgb: np.ndarray,
        sketch_u8: np.ndarray,
        *,
        target_wh: tuple[int, int],
        alpha: float = 0.42,
    ) -> ImageTk.PhotoImage:
        if roi_rgb.ndim != 3 or roi_rgb.shape[2] != 3:
            raise ValueError("_roi_overlay_photo expects RGB ROI")

        tw, th = target_wh
        tw, th = max(1, tw), max(1, th)
        im0 = Image.fromarray(roi_rgb)
        im = im0.resize((tw, th), resample=Image.Resampling.BILINEAR)
        m = Image.fromarray(sketch_u8).resize((tw, th), resample=Image.Resampling.NEAREST)

        base = np.array(im).astype(np.float32)
        mm = np.array(m)

        m_black = (mm == 255).astype(np.uint8) * 255
        m_white = (mm == 0).astype(np.uint8) * 255

        e_t1 = cv2.Canny(m_black, 50, 150)
        e_t2 = cv2.Canny(m_white, 50, 150)
        k = np.ones((3, 3), np.uint8)
        e_t1 = cv2.dilate(e_t1, k, iterations=1)
        e_t2 = cv2.dilate(e_t2, k, iterations=1)

        overlay = base.copy()
        # t1 boundary ~ black/mid (blue), t2 ~ mid/white (orange)
        _blend_edge(overlay, e_t1, (60.0, 120.0, 255.0), alpha)
        _blend_edge(overlay, e_t2, (255.0, 140.0, 40.0), alpha * 0.95)

        out_u8 = np.clip(overlay, 0, 255).astype(np.uint8)
        out_u8 = _annotate_preview(
            out_u8,
            title="Contours on photo",
            legend_lines=[
                ("t1 boundary", (70, 130, 255)),
                ("t2 boundary", (255, 150, 50)),
            ],
        )
        return ImageTk.PhotoImage(Image.fromarray(out_u8), master=root)

    def render() -> None:
        try:
            t1, t2 = float(t_vals[0]), float(t_vals[1])
            if not (0.0 < t1 < t2 < 1.0):
                raise ValueError("Need 0 < t1 < t2 < 1")

            gray_roi = gray[y0:y1, x0:x1]
            _, sketch_u8 = sketch_three_bins(gray_roi, t1=float(t1), t2=float(t2))

            if var_small_to_gray.get():
                n = int(float(var_small_N.get().strip()))
                if n <= 0:
                    raise ValueError("N must be positive")
                sketch_u8 = small_components_to_gray(sketch_u8, min_area=n)

            state["sketch_u8"] = sketch_u8

            tw, th = _preview_target_wh()
            im_sk = Image.fromarray(sketch_u8).resize((tw, th), resample=Image.Resampling.NEAREST)
            rgb = np.array(im_sk.convert("RGB"))
            rgb = _annotate_preview(rgb, title="Quantized sketch")
            ph_sk = ImageTk.PhotoImage(Image.fromarray(rgb), master=root)

            roi_rgb = img_rgb[y0:y1, x0:x1]
            ph_right_overlay = _roi_overlay_photo(roi_rgb, sketch_u8, target_wh=(tw, th), alpha=0.42)

            state["ph_right"] = ph_sk
            state["ph_right_overlay"] = ph_right_overlay

            lbl_right.configure(image=ph_sk)
            lbl_right_overlay.configure(image=ph_right_overlay)
            lbl_right.image = ph_sk
            lbl_right_overlay.image = ph_right_overlay

            status_var.set("Preview updated")
        except Exception as e:
            status_var.set(f"Error: {e}")

    def on_save() -> None:
        if state["sketch_u8"] is None:
            status_var.set("Nothing to save — fix errors or wait for preview.")
            return
        try:
            t1, t2 = _read_t_from_entries()
            chosen["t1"] = t1
            chosen["t2"] = t2
            chosen["small_to_gray"] = bool(var_small_to_gray.get())
            chosen["small_N"] = int(float(var_small_N.get().strip()))
        except Exception as e:
            status_var.set(f"Error: {e}")
            return
        chosen["done"] = True
        root.after(10, root.destroy)

    def on_cancel() -> None:
        chosen["done"] = False
        root.after(10, root.destroy)

    _pending: dict[str, str | None] = {"id": None}

    def _schedule_render() -> None:
        if _pending["id"] is not None:
            try:
                root.after_cancel(_pending["id"])
            except Exception:
                pass
        _pending["id"] = root.after(75, _do_render_clear)

    def _do_render_clear() -> None:
        render()
        _pending["id"] = None

    btn_refresh.configure(command=render)

    save_btn = create_primary_button(footer, text="Save", command=on_save, width=100, height=32)
    save_btn.grid(row=0, column=2, sticky="e")
    cancel_btn = create_secondary_button(footer, text="Cancel", command=on_cancel, width=100, height=32)
    cancel_btn.grid(row=0, column=1, sticky="e", padx=(0, 8))

    root.bind("<Return>", lambda _e: on_save())
    root.bind("<Escape>", lambda _e: on_cancel())

    def _after_map(_ev: object | None = None) -> None:
        redraw_range_bar()

    range_bar.bind("<Configure>", lambda _e: redraw_range_bar())
    root.after(50, _after_map)

    def _on_preview_resize(_ev: object | None = None) -> None:
        if state["sketch_u8"] is None:
            return
        if _resize_after[0] is not None:
            try:
                root.after_cancel(_resize_after[0])
            except Exception:
                pass
        _resize_after[0] = root.after(120, _apply_resize_render)

    def _apply_resize_render() -> None:
        _resize_after[0] = None
        try:
            render()
        except Exception:
            pass

    pane_sk.bind("<Configure>", lambda _e: _on_preview_resize())
    pane_ov.bind("<Configure>", lambda _e: _on_preview_resize())

    sync_strings()
    render()
    root.wait_window(root)

    if chosen.get("done", False):
        return chosen
    return None
