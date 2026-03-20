from __future__ import annotations

from pathlib import Path
import csv
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from ui.file_selection.reorganise_result import reorganise_results_to_ok_csv
from ui.file_selection.settings import load_folder_choices, persist_graph_units_settings


CANONICAL_HEADERS = [
    "overlay_path",
    "img_name",
    "accepted",
    "contour_version",
    "brain_area_px",
    "brain_perim_px",
    "midline_area_left_px",
    "midline_area_right_px",
    "midline_perimeter_left_px",
    "midline_perimeter_right_px",
    "non_complete_contour",
    "hipp_area_left_px",
    "hipp_area_right_px",
    "hipp_perimeter_left_px",
    "hipp_perimeter_right_px",
]

NUMERIC_COLUMNS = [
    "brain_area_px",
    "brain_perim_px",
    "midline_area_left_px",
    "midline_area_right_px",
    "midline_perimeter_left_px",
    "midline_perimeter_right_px",
    "hipp_area_left_px",
    "hipp_area_right_px",
    "hipp_perimeter_left_px",
    "hipp_perimeter_right_px",
]

ALLOWED_SCATTER_PAIRS = {
    ("hipp_area_left_px", "hipp_area_right_px"),
    ("hipp_perimeter_left_px", "hipp_perimeter_right_px"),
    ("midline_area_left_px", "midline_area_right_px"),
    ("midline_perimeter_left_px", "midline_perimeter_right_px"),
}


def _to_bool(v: object) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes")


def _to_float(v: object) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _read_rows(out_dir: str) -> list[dict[str, str]]:
    p = Path(out_dir).expanduser().resolve()
    candidates = [p / "result_ok.csv", p / "results.csv"]
    src = next((x for x in candidates if x.exists()), None)
    if src is None:
        return []

    rows: list[dict[str, str]] = []
    with src.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            rows.append({k: str(v or "") for k, v in row.items()})
    return rows


def _apply_dataset_mode(rows: list[dict[str, str]], mode: str) -> list[dict[str, str]]:
    ok_rows = [r for r in rows if str(r.get("accepted", "")).strip().lower() == "ok"]

    if mode == "include_both":
        return ok_rows

    grouped: dict[str, list[dict[str, str]]] = {}
    for r in ok_rows:
        key = str(r.get("overlay_path", "")).strip() or str(r.get("img_name", "")).strip()
        if not key:
            continue
        grouped.setdefault(key, []).append(r)

    out: list[dict[str, str]] = []
    for _, rr in grouped.items():
        rr_nc = [x for x in rr if _to_bool(x.get("non_complete_contour", ""))]
        rr_single = [x for x in rr if not _to_bool(x.get("non_complete_contour", ""))]

        if mode == "exclude_non_complete":
            if rr_single:
                out.append(rr_single[-1])
            continue

        if rr_single:
            out.append(rr_single[-1])
            continue

        if not rr_nc:
            continue

        if mode == "prefer_corrected":
            corr = [x for x in rr_nc if str(x.get("contour_version", "")).strip().lower() == "corrected"]
            out.append((corr[-1] if corr else rr_nc[-1]))
        elif mode == "use_original":
            raw = [x for x in rr_nc if str(x.get("contour_version", "")).strip().lower() == "raw"]
            out.append((raw[-1] if raw else rr_nc[-1]))
        else:
            out.append(rr_nc[-1])
    return out


def _stats_text(arr: np.ndarray, title: str) -> str:
    if arr.size == 0:
        return f"{title}\nN=0"
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    return (
        f"{title}\n"
        f"N={arr.size}\n"
        f"mean={float(np.mean(arr)):.4g}\n"
        f"std={float(np.std(arr, ddof=1)):.4g}\n"
        f"median={float(np.median(arr)):.4g}\n"
        f"IQR={q1:.4g}..{q3:.4g}\n"
        f"min={float(np.min(arr)):.4g} max={float(np.max(arr)):.4g}"
    )


def _extract_numeric(rows: list[dict[str, str]], col: str) -> np.ndarray:
    vals = np.array([_to_float(r.get(col, "")) for r in rows], dtype=float)
    return vals[np.isfinite(vals)]


def show_graphs_ui(parent: tk.Misc, out_dir: str) -> None:
    settings = load_folder_choices()
    init_convert = bool(settings.get("graphs_convert_enabled", False))
    init_unit = str(settings.get("graphs_unit", "mm")).strip().lower()
    if init_unit not in ("mm", "cm"):
        init_unit = "mm"
    try:
        init_ppu = float(settings.get("graphs_pixels_per_unit", 100.0))
        if not np.isfinite(init_ppu) or init_ppu <= 0:
            init_ppu = 100.0
    except Exception:
        init_ppu = 100.0

    try:
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    except Exception:
        messagebox.showwarning(
            "Show graphs",
            "matplotlib is not installed.\nInstall it (pip install matplotlib) to use graphs.",
            parent=parent,
        )
        return

    win = tk.Toplevel(parent)
    win.title("Show graphs")
    # Keep a reasonable default size; user can resize if needed.
    win.geometry("1050x680")
    win.rowconfigure(0, weight=1)
    win.columnconfigure(1, weight=1)

    left = ttk.Frame(win, padding=10)
    left.grid(row=0, column=0, sticky="ns")
    right = ttk.Frame(win, padding=10)
    right.grid(row=0, column=1, sticky="nsew")
    right.rowconfigure(1, weight=1)
    right.columnconfigure(0, weight=1)

    ttk.Label(left, text="Dataset mode").grid(row=0, column=0, sticky="w")
    var_mode = tk.StringVar(value="prefer_corrected")
    mode_map = {
        "Prefer corrected": "prefer_corrected",
        "Use original": "use_original",
        "Exclude non-complete": "exclude_non_complete",
        "Include both": "include_both",
    }
    cmb_mode = ttk.Combobox(left, state="readonly", values=list(mode_map.keys()))
    cmb_mode.set("Prefer corrected")
    cmb_mode.grid(row=1, column=0, sticky="ew", pady=(2, 10))

    # Optional conversion from pixels to mm/cm for plotting/statistics.
    conv = ttk.LabelFrame(left, text="Units", padding=6)
    conv.grid(row=2, column=0, sticky="ew", pady=(0, 8))
    conv.columnconfigure(1, weight=1)
    var_convert_units = tk.BooleanVar(value=init_convert)
    chk_convert = ttk.Checkbutton(conv, text="pxl to mm/cm", variable=var_convert_units)
    chk_convert.grid(row=0, column=0, columnspan=2, sticky="w")
    ttk.Label(conv, text="Unit").grid(row=1, column=0, sticky="w", pady=(4, 0))
    var_unit = tk.StringVar(value=init_unit)
    cmb_unit = ttk.Combobox(conv, state="readonly", values=["mm", "cm"], textvariable=var_unit, width=8)
    cmb_unit.grid(row=1, column=1, sticky="w", pady=(4, 0))
    ttk.Label(conv, text="Pixels per unit").grid(row=2, column=0, sticky="w", pady=(4, 0))
    var_px_per_unit = tk.DoubleVar(value=init_ppu)
    ent_px_per_unit = ttk.Entry(conv, textvariable=var_px_per_unit, width=12)
    ent_px_per_unit.grid(row=2, column=1, sticky="w", pady=(4, 0))

    ttk.Label(left, text="Graph type").grid(row=3, column=0, sticky="w")
    graph_map = {
        "Scatter": "scatter",
        "Histogram": "hist",
        "Boxplot": "boxplot",
    }
    cmb_graph = ttk.Combobox(left, state="readonly", values=list(graph_map.keys()))
    cmb_graph.set("Scatter")
    cmb_graph.grid(row=4, column=0, sticky="ew", pady=(2, 10))

    # Dynamic options area: controls depend on graph type
    opts = ttk.Frame(left)
    opts.grid(row=5, column=0, sticky="ew")
    opts.columnconfigure(0, weight=1)

    # Histogram options
    lbl_hist_col = ttk.Label(opts, text="Histogram column")
    cmb_hist_col = ttk.Combobox(opts, state="readonly", values=NUMERIC_COLUMNS)
    cmb_hist_col.set("hipp_area_left_px")
    lbl_bins = ttk.Label(opts, text="Bins")
    var_bins = tk.IntVar(value=30)
    ent_bins = ttk.Entry(opts, textvariable=var_bins, width=8)

    # Scatter options
    lbl_sc_x = ttk.Label(opts, text="X column")
    cmb_sc_x = ttk.Combobox(opts, state="readonly", values=NUMERIC_COLUMNS)
    cmb_sc_x.set("hipp_area_left_px")
    lbl_sc_y = ttk.Label(opts, text="Y column")
    cmb_sc_y = ttk.Combobox(opts, state="readonly", values=NUMERIC_COLUMNS)
    cmb_sc_y.set("hipp_area_right_px")
    var_sc_refline = tk.BooleanVar(value=False)
    chk_sc_refline = ttk.Checkbutton(opts, text="Show y=x reference line", variable=var_sc_refline)

    # Boxplot options
    lbl_box_group = ttk.Label(opts, text="Boxplot group")
    box_group_map = {
        "Hippocampus": "hipp",
        "Midline": "midline",
    }
    cmb_box_group = ttk.Combobox(opts, state="readonly", values=list(box_group_map.keys()))
    cmb_box_group.set("Hippocampus")
    lbl_box_metric = ttk.Label(opts, text="Metric")
    box_metric_map = {
        "Area": "area",
        "Perimeter": "perimeter",
    }
    cmb_box_metric = ttk.Combobox(opts, state="readonly", values=list(box_metric_map.keys()))
    cmb_box_metric.set("Area")

    btn_draw = ttk.Button(left, text="Draw")
    btn_draw.grid(row=6, column=0, sticky="ew", pady=(8, 4))
    btns_save = ttk.Frame(left)
    btns_save.grid(row=7, column=0, sticky="ew", pady=(4, 4))
    btns_save.columnconfigure(0, weight=1)
    btns_save.columnconfigure(1, weight=1)
    btns_save.columnconfigure(2, weight=1)
    btn_save_current = ttk.Button(btns_save, text="Save current graph")
    btn_save_all = ttk.Button(btns_save, text="Save all graphs")
    btn_copy_stats = ttk.Button(btns_save, text="Copy stats")
    btn_save_current.grid(row=0, column=0, sticky="ew", padx=(0, 4))
    btn_save_all.grid(row=0, column=1, sticky="ew", padx=(4, 0))
    btn_copy_stats.grid(row=0, column=2, sticky="ew", padx=(4, 0))

    stats_txt = tk.Text(left, width=40, height=24, wrap="word")
    stats_txt.grid(row=8, column=0, sticky="nsew", pady=(8, 0))
    stats_txt.configure(state="disabled")

    # NOTE: In TkAgg, canvas pixel size ~= figsize * dpi.
    # Keep display DPI moderate so the UI doesn't become huge.
    fig = Figure(figsize=(7.4, 5.4), dpi=110)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=right)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=1, column=0, sticky="nsew")

    def _set_stats(text: str) -> None:
        stats_txt.configure(state="normal")
        stats_txt.delete("1.0", tk.END)
        stats_txt.insert(tk.END, text)
        stats_txt.configure(state="disabled")

    def _copy_stats_selection(_ev=None):
        try:
            sel = stats_txt.selection_get()
        except Exception:
            return "break"
        try:
            win.clipboard_clear()
            win.clipboard_append(sel)
            win.update_idletasks()
        except Exception:
            pass
        return "break"

    def _copy_stats_all(_ev=None):
        try:
            txt = stats_txt.get("1.0", "end-1c")
            if not txt.strip():
                return "break"
            win.clipboard_clear()
            win.clipboard_append(txt)
            win.update_idletasks()
        except Exception:
            pass
        return "break"

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 2 or b.size < 2:
            return float("nan")
        if np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    def _pixels_per_unit() -> float:
        try:
            v = float(var_px_per_unit.get())
            return v if v > 0 else 1.0
        except Exception:
            return 1.0

    def _persist_units_settings() -> None:
        persist_graph_units_settings(
            convert_enabled=bool(var_convert_units.get()),
            unit=var_unit.get().strip().lower() or "mm",
            pixels_per_unit=_pixels_per_unit(),
        )

    def _col_kind(col: str) -> str:
        c = col.lower()
        if "area" in c:
            return "area"
        if "perim" in c:
            return "perimeter"
        return "linear"

    def _convert_vals(vals: np.ndarray, col: str) -> np.ndarray:
        if not bool(var_convert_units.get()):
            return vals
        k = _col_kind(col)
        ppu = _pixels_per_unit()
        if k == "area":
            return vals / (ppu * ppu)
        return vals / ppu

    def _label(col: str) -> str:
        if not bool(var_convert_units.get()):
            return col
        unit = var_unit.get().strip() or "unit"
        k = _col_kind(col)
        if k == "area":
            return f"{col} ({unit}^2)"
        return f"{col} ({unit})"

    def _show_hist_opts() -> None:
        lbl_hist_col.grid(row=0, column=0, sticky="w")
        cmb_hist_col.grid(row=1, column=0, sticky="ew", pady=(2, 8))
        lbl_bins.grid(row=2, column=0, sticky="w")
        ent_bins.grid(row=3, column=0, sticky="w", pady=(2, 8))

    def _hide_hist_opts() -> None:
        lbl_hist_col.grid_remove()
        cmb_hist_col.grid_remove()
        lbl_bins.grid_remove()
        ent_bins.grid_remove()

    def _show_scatter_opts() -> None:
        lbl_sc_x.grid(row=0, column=0, sticky="w")
        cmb_sc_x.grid(row=1, column=0, sticky="ew", pady=(2, 8))
        lbl_sc_y.grid(row=2, column=0, sticky="w")
        cmb_sc_y.grid(row=3, column=0, sticky="ew", pady=(2, 8))
        chk_sc_refline.grid(row=4, column=0, sticky="w", pady=(2, 8))

    def _hide_scatter_opts() -> None:
        lbl_sc_x.grid_remove()
        cmb_sc_x.grid_remove()
        lbl_sc_y.grid_remove()
        cmb_sc_y.grid_remove()
        chk_sc_refline.grid_remove()

    def _show_box_opts() -> None:
        lbl_box_group.grid(row=0, column=0, sticky="w")
        cmb_box_group.grid(row=1, column=0, sticky="ew", pady=(2, 8))
        lbl_box_metric.grid(row=2, column=0, sticky="w")
        cmb_box_metric.grid(row=3, column=0, sticky="ew", pady=(2, 8))

    def _hide_box_opts() -> None:
        lbl_box_group.grid_remove()
        cmb_box_group.grid_remove()
        lbl_box_metric.grid_remove()
        cmb_box_metric.grid_remove()

    def _refresh_dynamic_controls() -> None:
        _hide_hist_opts()
        _hide_scatter_opts()
        _hide_box_opts()
        graph_key = graph_map.get(cmb_graph.get(), "scatter")
        if graph_key == "hist":
            _show_hist_opts()
        elif graph_key == "boxplot":
            _show_box_opts()
        else:
            _show_scatter_opts()

    def _draw() -> None:
        # Always refresh normalized dataset before plotting.
        ok_save, _out_csv, _rows_written = reorganise_results_to_ok_csv(out_dir)
        if not ok_save:
            _set_stats("Failed to reorganise results before plotting.\nCheck terminal output.")
            ax.clear()
            canvas.draw_idle()
            return
        rows = _read_rows(out_dir)
        if not rows:
            _set_stats("No rows found after reorganisation.")
            ax.clear()
            canvas.draw_idle()
            return

        mode_key = mode_map.get(cmb_mode.get(), "prefer_corrected")
        graph_key = graph_map.get(cmb_graph.get(), "scatter")
        r = _apply_dataset_mode(rows, mode_key)
        ax.clear()

        if graph_key == "scatter":
            xcol = cmb_sc_x.get().strip()
            ycol = cmb_sc_y.get().strip()
            pairs: list[tuple[float, float]] = []
            for row in r:
                xv = _to_float(row.get(xcol, ""))
                yv = _to_float(row.get(ycol, ""))
                if np.isfinite(xv) and np.isfinite(yv):
                    pairs.append((xv, yv))
            if not pairs:
                _set_stats(f"No valid paired values for scatter:\nX={xcol}\nY={ycol}")
            else:
                x = np.array([p[0] for p in pairs], dtype=float)
                y = np.array([p[1] for p in pairs], dtype=float)
                x = _convert_vals(x, xcol)
                y = _convert_vals(y, ycol)
                n = x.size
                ax.scatter(x, y, alpha=0.75, s=40, linewidths=0.3)
                if bool(var_sc_refline.get()):
                    # Draw y=x only within overlapping X/Y range.
                    lo = max(float(np.min(x)), float(np.min(y)))
                    hi = min(float(np.max(x)), float(np.max(y)))
                    if hi > lo:
                        ax.plot([lo, hi], [lo, hi], "--", linewidth=1.2, alpha=0.75)
                ax.set_xlabel(_label(xcol))
                ax.set_ylabel(_label(ycol))
                ax.set_title(f"Scatter: {_label(xcol)} vs {_label(ycol)}")
                _set_stats(
                    _stats_text(x, f"X ({_label(xcol)})")
                    + "\n\n"
                    + _stats_text(y, f"Y ({_label(ycol)})")
                    + f"\n\nN pairs={n}\ncorr={_corr(x, y):.4g}\nmean(Y-X)={float(np.mean(y-x)):.4g}"
                )

        elif graph_key == "boxplot":
            grp = box_group_map.get(cmb_box_group.get(), "hipp")
            metric = box_metric_map.get(cmb_box_metric.get(), "area")
            if grp == "hipp":
                lcol = "hipp_area_left_px" if metric == "area" else "hipp_perimeter_left_px"
                rcol = "hipp_area_right_px" if metric == "area" else "hipp_perimeter_right_px"
                title = f"Hippocampus {metric}: left vs right"
            else:
                lcol = "midline_area_left_px" if metric == "area" else "midline_perimeter_left_px"
                rcol = "midline_area_right_px" if metric == "area" else "midline_perimeter_right_px"
                title = f"Midline {metric}: left vs right"

            lvals = _convert_vals(_extract_numeric(r, lcol), lcol)
            rvals = _convert_vals(_extract_numeric(r, rcol), rcol)
            if lvals.size == 0 and rvals.size == 0:
                _set_stats(f"No valid values for boxplot:\n{lcol}\n{rcol}")
            else:
                ax.boxplot([lvals, rvals], labels=["Left", "Right"], showfliers=True)
                ax.set_title(title + (f" [{var_unit.get()}]" if bool(var_convert_units.get()) else " [px]"))
                y_unit = f"{var_unit.get()}^2" if _col_kind(lcol) == "area" and bool(var_convert_units.get()) else (var_unit.get() if bool(var_convert_units.get()) else "px")
                ax.set_ylabel(y_unit)
                _set_stats(
                    _stats_text(lvals, f"Left ({_label(lcol)})")
                    + "\n\n"
                    + _stats_text(rvals, f"Right ({_label(rcol)})")
                )

        else:  # histogram
            col = cmb_hist_col.get().strip()
            vals = _convert_vals(_extract_numeric(r, col), col)
            bins = var_bins.get() if isinstance(var_bins.get(), int) else 30
            bins = max(2, min(400, int(bins)))
            if vals.size == 0:
                _set_stats(f"No valid values for {col}.")
            else:
                ax.hist(vals, bins=bins, alpha=0.8, edgecolor="black")
                ax.set_title(f"Histogram: {_label(col)}")
                ax.set_xlabel(_label(col))
                ax.set_ylabel("Count")
                _set_stats(_stats_text(vals, f"{_label(col)}"))

        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        canvas.draw_idle()

    def _graphs_dir() -> Path:
        gd = Path(out_dir).expanduser().resolve() / "result_graphs"
        gd.mkdir(parents=True, exist_ok=True)
        return gd

    def _safe_name(s: str) -> str:
        out = []
        for ch in s.lower().strip():
            if ch.isalnum() or ch in ("_", "-"):
                out.append(ch)
            elif ch in (" ", ".", "/"):
                out.append("_")
        return "".join(out).strip("_") or "plot"

    def _current_plot_name() -> str:
        graph_key = graph_map.get(cmb_graph.get(), "scatter")
        mode_key = mode_map.get(cmb_mode.get(), "prefer_corrected")
        if graph_key == "scatter":
            core = f"scatter_{cmb_sc_x.get().strip()}_vs_{cmb_sc_y.get().strip()}"
        elif graph_key == "hist":
            core = f"hist_{cmb_hist_col.get().strip()}_bins{max(2, min(400, int(var_bins.get())))}"
        else:
            grp = box_group_map.get(cmb_box_group.get(), "hipp")
            met = box_metric_map.get(cmb_box_metric.get(), "area")
            core = f"box_{grp}_{met}_left_vs_right"
        return f"{_safe_name(mode_key)}__{_safe_name(core)}.png"

    def _is_allowed_scatter_pair(xcol: str, ycol: str) -> bool:
        pair = (xcol, ycol)
        pair_rev = (ycol, xcol)
        return (pair in ALLOWED_SCATTER_PAIRS) or (pair_rev in ALLOWED_SCATTER_PAIRS)

    def _save_current_graph() -> None:
        # For scatter, save only semantically valid paired comparisons.
        graph_key = graph_map.get(cmb_graph.get(), "scatter")
        if graph_key == "scatter":
            xcol = cmb_sc_x.get().strip()
            ycol = cmb_sc_y.get().strip()
            if not _is_allowed_scatter_pair(xcol, ycol):
                messagebox.showwarning(
                    "Show graphs",
                    "For scatter saving, use only paired comparisons:\n"
                    "- hipp_area_left_px vs hipp_area_right_px\n"
                    "- hipp_perimeter_left_px vs hipp_perimeter_right_px\n"
                    "- midline_area_left_px vs midline_area_right_px\n"
                    "- midline_perimeter_left_px vs midline_perimeter_right_px",
                    parent=win,
                )
                return

        _draw()
        out_path = _graphs_dir() / _current_plot_name()
        try:
            fig.savefig(out_path, dpi=220, bbox_inches="tight")
            messagebox.showinfo("Show graphs", f"Saved:\n{out_path}", parent=win)
        except Exception as exc:
            messagebox.showwarning("Show graphs", f"Failed to save graph:\n{exc}", parent=win)

    def _save_all_graphs() -> None:
        prev = {
            "mode": cmb_mode.get(),
            "graph": cmb_graph.get(),
            "scx": cmb_sc_x.get(),
            "scy": cmb_sc_y.get(),
            "hist_col": cmb_hist_col.get(),
            "bins": var_bins.get(),
            "box_group": cmb_box_group.get(),
            "box_metric": cmb_box_metric.get(),
            "ref": bool(var_sc_refline.get()),
        }

        out_dir_graphs = _graphs_dir()
        saved = 0
        try:
            # Use only currently selected Dataset mode.
            # Scatter presets: L/R area + L/R perimeter for both groups.
            cmb_graph.set("Scatter")
            _refresh_dynamic_controls()
            scatter_pairs = [
                ("hipp_area_left_px", "hipp_area_right_px"),
                ("hipp_perimeter_left_px", "hipp_perimeter_right_px"),
                ("midline_area_left_px", "midline_area_right_px"),
                ("midline_perimeter_left_px", "midline_perimeter_right_px"),
            ]
            for xcol, ycol in scatter_pairs:
                cmb_sc_x.set(xcol)
                cmb_sc_y.set(ycol)
                var_sc_refline.set(False)
                _draw()
                out_path = out_dir_graphs / _current_plot_name()
                fig.savefig(out_path, dpi=220, bbox_inches="tight")
                saved += 1

            # Boxplot presets.
            cmb_graph.set("Boxplot")
            _refresh_dynamic_controls()
            for g in list(box_group_map.keys()):
                for met in list(box_metric_map.keys()):
                    cmb_box_group.set(g)
                    cmb_box_metric.set(met)
                    _draw()
                    out_path = out_dir_graphs / _current_plot_name()
                    fig.savefig(out_path, dpi=220, bbox_inches="tight")
                    saved += 1

            # Histogram presets: all numeric columns with current bins.
            cmb_graph.set("Histogram")
            _refresh_dynamic_controls()
            for col in NUMERIC_COLUMNS:
                cmb_hist_col.set(col)
                _draw()
                out_path = out_dir_graphs / _current_plot_name()
                fig.savefig(out_path, dpi=220, bbox_inches="tight")
                saved += 1
        except Exception as exc:
            messagebox.showwarning("Show graphs", f"Failed while saving all graphs:\n{exc}", parent=win)
        finally:
            cmb_mode.set(prev["mode"])
            cmb_graph.set(prev["graph"])
            _refresh_dynamic_controls()
            cmb_sc_x.set(prev["scx"])
            cmb_sc_y.set(prev["scy"])
            cmb_hist_col.set(prev["hist_col"])
            var_bins.set(prev["bins"])
            cmb_box_group.set(prev["box_group"])
            cmb_box_metric.set(prev["box_metric"])
            var_sc_refline.set(prev["ref"])
            _draw()

        if saved > 0:
            messagebox.showinfo("Show graphs", f"Saved {saved} graph(s) to:\n{out_dir_graphs}", parent=win)

    btn_draw.configure(command=_draw)
    btn_save_current.configure(command=_save_current_graph)
    btn_save_all.configure(command=_save_all_graphs)
    btn_copy_stats.configure(command=_copy_stats_all)
    cmb_mode.bind("<<ComboboxSelected>>", lambda _e: _draw())
    cmb_graph.bind("<<ComboboxSelected>>", lambda _e: (_refresh_dynamic_controls(), _draw()))
    cmb_hist_col.bind("<<ComboboxSelected>>", lambda _e: _draw())
    cmb_sc_x.bind("<<ComboboxSelected>>", lambda _e: _draw())
    cmb_sc_y.bind("<<ComboboxSelected>>", lambda _e: _draw())
    chk_sc_refline.configure(command=_draw)
    chk_convert.configure(command=lambda: (_persist_units_settings(), _draw()))
    cmb_unit.bind("<<ComboboxSelected>>", lambda _e: (_persist_units_settings(), _draw()))
    ent_px_per_unit.bind("<Return>", lambda _e: (_persist_units_settings(), _draw()))
    ent_px_per_unit.bind("<FocusOut>", lambda _e: (_persist_units_settings(), _draw()))
    # Explicit copy bindings for macOS/Windows/Linux.
    stats_txt.bind("<Command-c>", _copy_stats_selection)
    stats_txt.bind("<Control-c>", _copy_stats_selection)
    # Global shortcut: always copy full stats text (even if canvas has focus).
    win.bind("<Command-Shift-C>", _copy_stats_all)
    win.bind("<Control-Shift-C>", _copy_stats_all)
    cmb_box_group.bind("<<ComboboxSelected>>", lambda _e: _draw())
    cmb_box_metric.bind("<<ComboboxSelected>>", lambda _e: _draw())
    _refresh_dynamic_controls()
    _draw()

