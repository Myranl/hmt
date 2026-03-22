from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import sys
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, messagebox

import numpy as np
from config import CATEGORIES_STORE_NAME
from core.categories import load_store, merge_row_with_categories
from core.categories.schema import CategoryStore
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

# Same token as category overview buckets for empty cells
UNASSIGNED_LABEL = "(unassigned)"


def _load_category_store(out_dir: str) -> CategoryStore | None:
    p = Path(out_dir).expanduser().resolve() / CATEGORIES_STORE_NAME
    return load_store(p) if p.is_file() else None


def _resolve_image_path_for_row(row: dict[str, str], store: CategoryStore | None) -> Path | None:
    """Best-effort path to the source image for category resolution (under input_root)."""
    if store is None:
        return None
    root = Path(store.input_root).expanduser().resolve()
    if not root.is_dir():
        return None
    ip = str(row.get("image_path", "")).strip()
    if ip:
        p = Path(ip).expanduser()
        if p.is_absolute():
            if p.is_file():
                return p.resolve()
        else:
            cand = (root / ip).resolve()
            if cand.is_file():
                return cand
    imn = str(row.get("img_name", "")).strip()
    if imn:
        cand = (root / imn).resolve()
        if cand.is_file():
            return cand
    return None


def _enrich_rows_with_categories(
    rows: list[dict[str, str]], store: CategoryStore | None
) -> list[dict[str, str]]:
    if store is None:
        return rows
    out: list[dict[str, str]] = []
    for row in rows:
        p = _resolve_image_path_for_row(row, store)
        merged = merge_row_with_categories(dict(row), p, store)
        out.append({k: str(v or "") for k, v in merged.items()})
    return out


def _cat_value(row: dict[str, str], col_id: str) -> str:
    v = str(row.get(col_id, "")).strip()
    return v if v else UNASSIGNED_LABEL


def _try_welch_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float] | None:
    """Returns (t_statistic, p_value) or None if not computable / scipy missing."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return None
    try:
        from scipy.stats import ttest_ind

        r = ttest_ind(a, b, equal_var=False, nan_policy="omit")
        return float(r.statistic), float(r.pvalue)
    except Exception:
        return None


def _distinct_colors(n: int):
    """RGBA rows for n categories (matplotlib tab10-like)."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(max(n, 1))]


def _to_bool(v: object) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes")


def _to_float(v: object) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _short_img_label(row: dict[str, str]) -> str:
    """Basename for tooltips: prefer img_name, then image_path, then overlay_path."""
    name = str(row.get("img_name", "")).strip()
    if name:
        return Path(name).name
    ip = str(row.get("image_path", "")).strip()
    if ip:
        return Path(ip).name
    op = str(row.get("overlay_path", "")).strip()
    if op:
        return Path(op).name
    return "?"


def _read_rows(out_dir: str) -> list[dict[str, str]]:
    p = Path(out_dir).expanduser().resolve()
    # Prefer results.csv: it includes category columns merged by the pipeline / refresh.
    # result_ok.csv is canonical-only (no sex/dose/…) — using it first hid categories in graphs.
    candidates = [p / "results.csv", p / "result_ok.csv"]
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
    win.geometry("1200x720")
    win.minsize(980, 620)
    win.rowconfigure(0, weight=1)
    win.columnconfigure(1, weight=1)
    win.columnconfigure(2, weight=0, minsize=300)

    # Light styling: calmer chrome, consistent padding (best effort per platform).
    _panel_bg = "#f2f4f8"
    try:
        style = ttk.Style(win)
        if sys.platform == "darwin":
            style.theme_use("aqua")
        else:
            style.theme_use("clam")
        style.configure("Graphs.TFrame", background=_panel_bg)
        style.configure("Graphs.TLabelframe", background=_panel_bg)
        style.configure("Graphs.TLabelframe.Label", background=_panel_bg, font=("TkDefaultFont", 10, "bold"))
        style.configure("Graphs.TLabel", background=_panel_bg)
        style.configure("Accent.TButton", padding=(12, 8))
        win.configure(bg=_panel_bg)
    except Exception:
        _panel_bg = None  # type: ignore[assignment]

    def _st(nm: str) -> dict[str, str]:
        return {"style": nm} if _panel_bg else {}

    left = ttk.Frame(win, padding=(14, 16, 10, 16), **_st("Graphs.TFrame"))
    left.grid(row=0, column=0, sticky="ns")
    left.columnconfigure(0, weight=1)

    plot_area = ttk.Frame(win, padding=(4, 16, 8, 16), **_st("Graphs.TFrame"))
    plot_area.grid(row=0, column=1, sticky="nsew")
    plot_area.rowconfigure(0, weight=1)
    plot_area.columnconfigure(0, weight=1)

    stats_lf = ttk.LabelFrame(
        win,
        text="Statistics",
        padding=(10, 10),
        **_st("Graphs.TLabelframe"),
    )
    stats_lf.grid(row=0, column=2, sticky="nsew", padx=(0, 14), pady=16)
    stats_lf.rowconfigure(0, weight=1)
    stats_lf.columnconfigure(0, weight=1)

    lf_mode = ttk.LabelFrame(left, text="Dataset", padding=(10, 8), **_st("Graphs.TLabelframe"))
    lf_mode.grid(row=0, column=0, sticky="ew", pady=(0, 10))
    lf_mode.columnconfigure(0, weight=1)
    var_mode = tk.StringVar(value="prefer_corrected")
    mode_map = {
        "Prefer corrected": "prefer_corrected",
        "Use original": "use_original",
        "Exclude non-complete": "exclude_non_complete",
        "Include both": "include_both",
    }
    cmb_mode = ttk.Combobox(lf_mode, state="readonly", values=list(mode_map.keys()))
    cmb_mode.set("Prefer corrected")
    cmb_mode.grid(row=0, column=0, sticky="ew")

    # Optional conversion from pixels to mm/cm for plotting/statistics.
    conv = ttk.LabelFrame(left, text="Units", padding=(10, 8), **_st("Graphs.TLabelframe"))
    conv.grid(row=1, column=0, sticky="ew", pady=(0, 10))
    conv.columnconfigure(1, weight=1)
    var_convert_units = tk.BooleanVar(value=init_convert)
    chk_convert = ttk.Checkbutton(conv, text="Convert pixels → mm/cm", variable=var_convert_units)
    chk_convert.grid(row=0, column=0, columnspan=2, sticky="w")
    ttk.Label(conv, text="Unit").grid(row=1, column=0, sticky="w", pady=(4, 0))
    var_unit = tk.StringVar(value=init_unit)
    cmb_unit = ttk.Combobox(conv, state="readonly", values=["mm", "cm"], textvariable=var_unit, width=8)
    cmb_unit.grid(row=1, column=1, sticky="w", pady=(4, 0))
    ttk.Label(conv, text="Pixels per unit").grid(row=2, column=0, sticky="w", pady=(4, 0))
    var_px_per_unit = tk.DoubleVar(value=init_ppu)
    ent_px_per_unit = ttk.Entry(conv, textvariable=var_px_per_unit, width=12)
    ent_px_per_unit.grid(row=2, column=1, sticky="w", pady=(4, 0))

    lf_chart = ttk.LabelFrame(left, text="Chart type", padding=(10, 8), **_st("Graphs.TLabelframe"))
    lf_chart.grid(row=2, column=0, sticky="ew", pady=(0, 10))
    lf_chart.columnconfigure(0, weight=1)
    graph_map = {
        "Scatter": "scatter",
        "Histogram": "hist",
        "Boxplot": "boxplot",
    }
    cmb_graph = ttk.Combobox(lf_chart, state="readonly", values=list(graph_map.keys()))
    cmb_graph.set("Scatter")
    cmb_graph.grid(row=0, column=0, sticky="ew")

    lf_cat = ttk.LabelFrame(left, text="Category", padding=(10, 8), **_st("Graphs.TLabelframe"))
    lf_cat.grid(row=3, column=0, sticky="ew", pady=(0, 10))
    lf_cat.columnconfigure(0, weight=1)
    ttk.Label(
        lf_cat,
        text="Color / grouping (category_assignments.json)",
        wraplength=240,
        **_st("Graphs.TLabel"),
    ).grid(row=0, column=0, sticky="w")
    cat_choice: dict[str, str] = {"(none)": ""}

    def _rebuild_cat_combo() -> None:
        st = _load_category_store(out_dir)
        cat_choice.clear()
        cat_choice["(none)"] = ""
        labels = ["(none)"]
        if st:
            for c in st.columns:
                if not c.id:
                    continue
                lab = f"{c.title} ({c.id})"
                labels.append(lab)
                cat_choice[lab] = c.id
        cmb_cat.configure(values=labels)
        cur = cmb_cat.get()
        if cur not in labels:
            cmb_cat.set("(none)")

    cmb_cat = ttk.Combobox(lf_cat, state="readonly", values=["(none)"])
    cmb_cat.set("(none)")
    cmb_cat.grid(row=1, column=0, sticky="ew", pady=(6, 0))

    # Dynamic options area: controls depend on graph type
    opts = ttk.LabelFrame(left, text="Plot options", padding=(10, 8), **_st("Graphs.TLabelframe"))
    opts.grid(row=4, column=0, sticky="ew", pady=(0, 10))
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

    btn_draw = ttk.Button(left, text="Redraw", **_st("Accent.TButton"))
    btn_draw.grid(row=5, column=0, sticky="ew", pady=(4, 8))
    btns_save = ttk.Frame(left, **_st("Graphs.TFrame"))
    btns_save.grid(row=6, column=0, sticky="ew", pady=(0, 0))
    btns_save.columnconfigure(0, weight=1)
    btns_save.columnconfigure(1, weight=1)
    btns_save.columnconfigure(2, weight=1)
    btn_save_current = ttk.Button(btns_save, text="Save current")
    btn_save_all = ttk.Button(btns_save, text="Save all")
    btn_copy_stats = ttk.Button(btns_save, text="Copy stats")
    btn_save_current.grid(row=0, column=0, sticky="ew", padx=(0, 4))
    btn_save_all.grid(row=0, column=1, sticky="ew", padx=(4, 0))
    btn_copy_stats.grid(row=0, column=2, sticky="ew", padx=(4, 0))

    _mono = tkfont.nametofont("TkFixedFont")
    stats_txt = tk.Text(
        stats_lf,
        width=34,
        height=32,
        wrap="word",
        font=_mono,
        relief="flat",
        borderwidth=0,
        padx=6,
        pady=8,
        highlightthickness=1,
        highlightbackground="#d0d4dc",
        highlightcolor="#a8b0c0",
        selectbackground="#c8d4f0",
    )
    stats_scroll = ttk.Scrollbar(stats_lf, orient=tk.VERTICAL, command=stats_txt.yview)
    stats_txt.configure(yscrollcommand=stats_scroll.set)
    stats_txt.grid(row=0, column=0, sticky="nsew")
    stats_scroll.grid(row=0, column=1, sticky="ns")
    stats_txt.configure(state="disabled")

    # NOTE: In TkAgg, canvas pixel size ~= figsize * dpi.
    # Keep display DPI moderate so the UI doesn't become huge.
    fig = Figure(figsize=(7.4, 5.4), dpi=110)
    fig.patch.set_facecolor("#fafafa")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#ffffff")
    canvas = FigureCanvasTkAgg(fig, master=plot_area)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=0, column=0, sticky="nsew")

    # Scatter hover: mpl_connect id + data (cleared on each _draw)
    scatter_hover_state: dict[str, Any] = {
        "cid": None,
        "sc": None,
        "annot": None,
        "xs": None,
        "ys": None,
        "labels": None,
        "active": False,
    }

    def _scatter_hover_teardown() -> None:
        if scatter_hover_state.get("cid") is not None:
            try:
                canvas.mpl_disconnect(scatter_hover_state["cid"])
            except Exception:
                pass
            scatter_hover_state["cid"] = None
        scatter_hover_state["active"] = False
        scatter_hover_state["sc"] = None
        scatter_hover_state["annot"] = None
        scatter_hover_state["xs"] = None
        scatter_hover_state["ys"] = None
        scatter_hover_state["labels"] = None

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

    def _active_cat_col() -> str | None:
        lab = cmb_cat.get()
        cid = cat_choice.get(lab, "")
        return cid if cid else None

    def _scatter_stats_by_category(
        x: np.ndarray,
        y: np.ndarray,
        cats: list[str],
        xlab: str,
        ylab: str,
    ) -> str:
        keys = sorted(set(cats))
        parts: list[str] = []
        for k in keys:
            idx = [i for i, c in enumerate(cats) if c == k]
            if not idx:
                continue
            xi = x[idx]
            yi = y[idx]
            parts.append(f"=== {k} ===\n" + _stats_text(xi, f"X ({xlab})") + "\n\n" + _stats_text(yi, f"Y ({ylab})"))
        parts.append(
            "--- All points ---\n"
            f"N pairs={x.size}\ncorr={_corr(x, y):.4g}\nmean(Y-X)={float(np.mean(y - x)):.4g}"
        )
        if len(keys) == 2:
            a, b = keys[0], keys[1]
            ia = np.array([i for i, c in enumerate(cats) if c == a], dtype=int)
            ib = np.array([i for i, c in enumerate(cats) if c == b], dtype=int)
            if ia.size >= 2 and ib.size >= 2:
                wx = _try_welch_ttest(x[ia], x[ib])
                wy = _try_welch_ttest(y[ia], y[ib])
                if wx:
                    parts.append(f"\nWelch t-test X ({a} vs {b}): t={wx[0]:.4g}, p={wx[1]:.4g}")
                if wy:
                    parts.append(f"Welch t-test Y ({a} vs {b}): t={wy[0]:.4g}, p={wy[1]:.4g}")
        return "\n\n".join(parts)

    def _draw() -> None:
        _rebuild_cat_combo()
        # Sync category columns into results.csv on disk (same as OK / Categories Close).
        try:
            from pipeline.batch import refresh_results_csv_categories

            refresh_results_csv_categories(out_dir)
        except Exception:
            pass
        # Always refresh normalized dataset before plotting.
        ok_save, _out_csv, _rows_written = reorganise_results_to_ok_csv(out_dir)
        if not ok_save:
            _scatter_hover_teardown()
            _set_stats("Failed to reorganise results before plotting.\nCheck terminal output.")
            ax.clear()
            canvas.draw_idle()
            return
        rows = _read_rows(out_dir)
        if not rows:
            _scatter_hover_teardown()
            _set_stats("No rows found after reorganisation.")
            ax.clear()
            canvas.draw_idle()
            return

        cat_store = _load_category_store(out_dir)
        mode_key = mode_map.get(cmb_mode.get(), "prefer_corrected")
        graph_key = graph_map.get(cmb_graph.get(), "scatter")
        _scatter_hover_teardown()
        r = _apply_dataset_mode(rows, mode_key)
        r = _enrich_rows_with_categories(r, cat_store)
        cat_col = _active_cat_col()
        ax.clear()
        ax.set_facecolor("#ffffff")
        fig.patch.set_facecolor("#fafafa")
        ax.grid(True, alpha=0.28, linestyle="--", linewidth=0.75, zorder=0)

        if graph_key == "scatter":
            xcol = cmb_sc_x.get().strip()
            ycol = cmb_sc_y.get().strip()
            pairs: list[tuple[float, float]] = []
            labels_sc: list[str] = []
            cats_sc: list[str] = []
            for row in r:
                xv = _to_float(row.get(xcol, ""))
                yv = _to_float(row.get(ycol, ""))
                if np.isfinite(xv) and np.isfinite(yv):
                    pairs.append((xv, yv))
                    labels_sc.append(_short_img_label(row))
                    cats_sc.append(_cat_value(row, cat_col) if cat_col else "")
            if not pairs:
                _set_stats(f"No valid paired values for scatter:\nX={xcol}\nY={ycol}")
            else:
                x = np.array([p[0] for p in pairs], dtype=float)
                y = np.array([p[1] for p in pairs], dtype=float)
                x = _convert_vals(x, xcol)
                y = _convert_vals(y, ycol)
                n = x.size
                if cat_col:
                    keys = sorted(set(cats_sc))
                    pal = _distinct_colors(len(keys))
                    kv = {k: pal[i] for i, k in enumerate(keys)}
                    face = [kv[c] for c in cats_sc]
                    sc = ax.scatter(
                        x,
                        y,
                        c=face,
                        alpha=0.78,
                        s=44,
                        linewidths=0.35,
                        edgecolors="black",
                        picker=True,
                        pickradius=12,
                    )
                    for i, k in enumerate(keys):
                        ax.scatter([], [], c=[pal[i]], label=k, s=44, edgecolors="black", linewidths=0.35)
                    ax.legend(loc="best", fontsize=8, framealpha=0.92)
                else:
                    sc = ax.scatter(
                        x,
                        y,
                        alpha=0.75,
                        s=40,
                        linewidths=0.3,
                        picker=True,
                        pickradius=12,
                    )
                annot = ax.annotate(
                    "",
                    xy=(0.0, 0.0),
                    xytext=(12, 12),
                    textcoords="offset points",
                    fontsize=9,
                    horizontalalignment="left",
                    verticalalignment="bottom",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="gray", alpha=0.92),
                    arrowprops=dict(arrowstyle="-", color="gray", linewidth=0.8),
                )
                annot.set_clip_on(False)
                annot.set_visible(False)

                def _scatter_smart_label_placement(xd: float, yd: float, text: str) -> None:
                    """Offset + ha/va so the box stays inside the axes when possible."""
                    xmin, xmax, ymin, ymax = ax.axis()
                    xr = float(xmax - xmin)
                    yr = float(ymax - ymin)
                    if not np.isfinite(xr) or xr <= 0:
                        xr = 1.0
                    if not np.isfinite(yr) or yr <= 0:
                        yr = 1.0
                    rx = float(np.clip((xd - xmin) / xr, 0.0, 1.0))
                    ry = float(np.clip((yd - ymin) / yr, 0.0, 1.0))
                    edge = 0.18
                    pad = 12.0
                    est_w_pt = max(44.0, min(190.0, 5.5 * max(len(text), 6)))
                    if rx >= 1.0 - edge:
                        dx, ha = -est_w_pt, "right"
                    elif rx <= edge:
                        dx, ha = pad, "left"
                    else:
                        dx, ha = (pad, "left") if rx < 0.5 else (-est_w_pt, "right")
                    if ry >= 1.0 - edge:
                        dy, va = -pad, "top"
                    elif ry <= edge:
                        dy, va = pad, "bottom"
                    else:
                        dy, va = (pad, "bottom") if ry < 0.5 else (-pad, "top")
                    annot.xytext = (dx, dy)
                    try:
                        annot.set_horizontalalignment(ha)
                        annot.set_verticalalignment(va)
                    except Exception:
                        pass

                def _on_scatter_hover(event) -> None:
                    if not scatter_hover_state.get("active"):
                        return
                    ann = scatter_hover_state.get("annot")
                    sc_art = scatter_hover_state.get("sc")
                    xs = scatter_hover_state.get("xs")
                    ys = scatter_hover_state.get("ys")
                    lbs = scatter_hover_state.get("labels")
                    if ann is None or sc_art is None or xs is None or ys is None or lbs is None:
                        return
                    if event.inaxes != ax:
                        if ann.get_visible():
                            ann.set_visible(False)
                            canvas.draw_idle()
                        return
                    try:
                        contained, props = sc_art.contains(event)
                    except Exception:
                        contained, props = False, {}
                    if contained and props is not None:
                        ind = props.get("ind")
                        if ind is not None and len(ind) > 0:
                            i = int(ind[0])
                            if 0 <= i < len(lbs):
                                txt = str(lbs[i])
                                if cat_col and i < len(cats_sc):
                                    txt = f"{cats_sc[i]} — {txt}"
                                ann.xy = (float(xs[i]), float(ys[i]))
                                ann.set_text(txt)
                                _scatter_smart_label_placement(float(xs[i]), float(ys[i]), txt)
                                ann.set_visible(True)
                                canvas.draw_idle()
                                return
                    if ann.get_visible():
                        ann.set_visible(False)
                        canvas.draw_idle()

                scatter_hover_state["active"] = True
                scatter_hover_state["sc"] = sc
                scatter_hover_state["annot"] = annot
                scatter_hover_state["xs"] = x
                scatter_hover_state["ys"] = y
                scatter_hover_state["labels"] = labels_sc
                scatter_hover_state["cid"] = canvas.mpl_connect("motion_notify_event", _on_scatter_hover)
                if bool(var_sc_refline.get()):
                    # Draw y=x only within overlapping X/Y range.
                    lo = max(float(np.min(x)), float(np.min(y)))
                    hi = min(float(np.max(x)), float(np.max(y)))
                    if hi > lo:
                        ax.plot([lo, hi], [lo, hi], "--", linewidth=1.2, alpha=0.75)
                ax.set_xlabel(_label(xcol))
                ax.set_ylabel(_label(ycol))
                ttl = f"Scatter: {_label(xcol)} vs {_label(ycol)}"
                if cat_col:
                    ttl += f" (by {cat_col})"
                ax.set_title(ttl)
                if cat_col:
                    _set_stats(
                        _scatter_stats_by_category(x, y, cats_sc, _label(xcol), _label(ycol))
                    )
                else:
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

            if not cat_col:
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
            else:
                # One pair of boxes (Left, Right) per category value.
                buckets: dict[str, tuple[list[float], list[float]]] = {}
                for row in r:
                    cat = _cat_value(row, cat_col)
                    xv = _to_float(row.get(lcol, ""))
                    yv = _to_float(row.get(rcol, ""))
                    if cat not in buckets:
                        buckets[cat] = ([], [])
                    if np.isfinite(xv):
                        buckets[cat][0].append(xv)
                    if np.isfinite(yv):
                        buckets[cat][1].append(yv)
                for cat in buckets:
                    lx, rx = buckets[cat]
                    buckets[cat] = (
                        _convert_vals(np.array(lx, dtype=float), lcol),
                        _convert_vals(np.array(rx, dtype=float), rcol),
                    )
                order = sorted(buckets.keys(), key=lambda x: (x != UNASSIGNED_LABEL, str(x).lower()))
                data: list[np.ndarray] = []
                positions: list[float] = []
                pos = 0.0
                step = 2.4
                w = 0.32
                for cat in order:
                    lv, rv = buckets[cat]
                    if lv.size == 0 and rv.size == 0:
                        continue
                    data.append(lv)
                    data.append(rv)
                    positions.extend([pos - w, pos + w])
                    pos += step
                if not data:
                    _set_stats(f"No valid values for boxplot (by category):\n{lcol}\n{rcol}")
                else:
                    bp = ax.boxplot(data, positions=positions, widths=w * 1.85, showfliers=True, patch_artist=True)
                    left_c = "#7eb6ff"
                    right_c = "#ffb37a"
                    for i, patch in enumerate(bp["boxes"]):
                        patch.set_facecolor(left_c if i % 2 == 0 else right_c)
                    tick_pos = [(positions[i] + positions[i + 1]) / 2 for i in range(0, len(positions), 2)]
                    tick_lbl: list[str] = []
                    for cat in order:
                        lv, rv = buckets[cat]
                        if lv.size == 0 and rv.size == 0:
                            continue
                        tick_lbl.append(cat)
                    ax.set_xticks(tick_pos[: len(tick_lbl)])
                    ax.set_xticklabels(tick_lbl, rotation=15, ha="right")
                    from matplotlib.patches import Patch

                    ax.legend(
                        handles=[Patch(facecolor=left_c, label="Left"), Patch(facecolor=right_c, label="Right")],
                        loc="upper right",
                        fontsize=8,
                    )
                    t2 = title + (f" [{var_unit.get()}]" if bool(var_convert_units.get()) else " [px]")
                    ax.set_title(t2 + f" — by {cat_col}")
                    y_unit = f"{var_unit.get()}^2" if _col_kind(lcol) == "area" and bool(var_convert_units.get()) else (var_unit.get() if bool(var_convert_units.get()) else "px")
                    ax.set_ylabel(y_unit)
                    st_parts: list[str] = []
                    for cat in order:
                        lv, rv = buckets[cat]
                        if lv.size == 0 and rv.size == 0:
                            continue
                        st_parts.append(
                            f"=== {cat} ===\n"
                            + _stats_text(lv, f"Left ({_label(lcol)})")
                            + "\n\n"
                            + _stats_text(rv, f"Right ({_label(rcol)})")
                        )
                    _set_stats("\n\n".join(st_parts))

        else:  # histogram
            col = cmb_hist_col.get().strip()
            bins = var_bins.get() if isinstance(var_bins.get(), int) else 30
            bins = max(2, min(400, int(bins)))
            if not cat_col:
                vals = _convert_vals(_extract_numeric(r, col), col)
                if vals.size == 0:
                    _set_stats(f"No valid values for {col}.")
                else:
                    ax.hist(vals, bins=bins, alpha=0.8, edgecolor="black")
                    ax.set_title(f"Histogram: {_label(col)}")
                    ax.set_xlabel(_label(col))
                    ax.set_ylabel("Count")
                    _set_stats(_stats_text(vals, f"{_label(col)}"))
            else:
                groups: dict[str, list[float]] = {}
                for row in r:
                    v = _to_float(row.get(col, ""))
                    if not np.isfinite(v):
                        continue
                    cat = _cat_value(row, cat_col)
                    groups.setdefault(cat, []).append(v)
                for cat in groups:
                    arr = np.array(groups[cat], dtype=float)
                    groups[cat] = list(_convert_vals(arr, col))
                order = sorted(groups.keys(), key=lambda x: (x != UNASSIGNED_LABEL, str(x).lower()))
                combined = np.array([x for cat in order for x in groups[cat]], dtype=float)
                if combined.size == 0:
                    _set_stats(f"No valid values for {col}.")
                else:
                    edges = np.histogram_bin_edges(combined, bins=bins)
                    pal = _distinct_colors(len(order))
                    for i, cat in enumerate(order):
                        vals = np.array(groups[cat], dtype=float)
                        if vals.size == 0:
                            continue
                        ax.hist(
                            vals,
                            bins=edges,
                            alpha=0.55,
                            label=str(cat),
                            color=pal[i % len(pal)],
                            edgecolor="black",
                            linewidth=0.35,
                        )
                    ax.legend(loc="best", fontsize=8)
                    ax.set_title(f"Histogram: {_label(col)} — by {cat_col}")
                    ax.set_xlabel(_label(col))
                    ax.set_ylabel("Count")
                    st_parts = []
                    for cat in order:
                        vals = np.array(groups[cat], dtype=float)
                        if vals.size == 0:
                            continue
                        st_parts.append(_stats_text(vals, f"{cat} ({_label(col)})"))
                    _set_stats("\n\n".join(st_parts))

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
        cc = _active_cat_col()
        if cc:
            core += f"_by_{cc}"
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
            "cat": cmb_cat.get(),
        }

        out_dir_graphs = _graphs_dir()
        saved = 0
        try:
            cmb_cat.set("(none)")
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
            if prev.get("cat") in cmb_cat.cget("values"):
                cmb_cat.set(prev["cat"])
            else:
                cmb_cat.set("(none)")
            _draw()

        if saved > 0:
            messagebox.showinfo("Show graphs", f"Saved {saved} graph(s) to:\n{out_dir_graphs}", parent=win)

    btn_draw.configure(command=_draw)
    btn_save_current.configure(command=_save_current_graph)
    btn_save_all.configure(command=_save_all_graphs)
    btn_copy_stats.configure(command=_copy_stats_all)
    cmb_mode.bind("<<ComboboxSelected>>", lambda _e: _draw())
    cmb_cat.bind("<<ComboboxSelected>>", lambda _e: _draw())
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

