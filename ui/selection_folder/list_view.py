from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
import tkinter as tk
from tkinter import ttk


ThumbMaker = Callable[[Path], "tk.PhotoImage | None"]


@dataclass
class ListViewCtx:
    root: tk.Tk
    inner: ttk.Frame
    on_inner_config: Callable[[], None]
    make_thumb: ThumbMaker
    lbl_count: ttk.Label
    status_var: tk.StringVar
    btn_run: ttk.Button
    var_in: tk.StringVar
    var_show_processed: tk.BooleanVar
    after_ids: list[str]

    # dynamic data
    img_paths: list[Path]
    processed: set[str]

    # row state
    thumbs: list[tk.PhotoImage]
    vars_sel: list[tk.BooleanVar]
    meta: list[dict]


_CTX: Optional[ListViewCtx] = None


def init_list_view(
    *,
    root: tk.Tk,
    inner: ttk.Frame,
    on_inner_config: Callable[[], None],
    make_thumb: ThumbMaker,
    lbl_count: ttk.Label,
    status_var: tk.StringVar,
    btn_run: ttk.Button,
    var_in: tk.StringVar,
    var_show_processed: tk.BooleanVar,
    after_ids: list[str],
) -> None:
    """Must be called once by the UI that owns the widgets."""
    global _CTX
    _CTX = ListViewCtx(
        root=root,
        inner=inner,
        on_inner_config=on_inner_config,
        make_thumb=make_thumb,
        lbl_count=lbl_count,
        status_var=status_var,
        btn_run=btn_run,
        var_in=var_in,
        var_show_processed=var_show_processed,
        after_ids=after_ids,
        img_paths=[],
        processed=set(),
        thumbs=[],
        vars_sel=[],
        meta=[],
    )


def set_data(*, img_paths: list[Path], processed: set[str]) -> None:
    """Update the list view data (called after scanning input/output)."""
    if _CTX is None:
        raise RuntimeError("list_view.init_list_view() must be called before set_data().")
    _CTX.img_paths = list(img_paths)
    _CTX.processed = set(processed)


def get_selection_state() -> tuple[list[tk.BooleanVar], list[dict]]:
    """Expose internal selection lists for wiring btn_run handler."""
    if _CTX is None:
        raise RuntimeError("list_view.init_list_view() must be called before get_selection_state().")
    return _CTX.vars_sel, _CTX.meta


def update_count() -> None:
    if _CTX is None:
        return
    total = len(_CTX.vars_sel)
    sel = sum(1 for v in _CTX.vars_sel if v.get())
    _CTX.lbl_count.configure(text=f"selected {sel}/{total}")


def clear_list() -> None:
    if _CTX is None:
        return
    for w in _CTX.inner.winfo_children():
        w.destroy()
    _CTX.thumbs.clear()
    _CTX.vars_sel.clear()
    _CTX.meta.clear()
    update_count()
    _CTX.on_inner_config()


def is_processed(p: Path) -> bool:
    if _CTX is None:
        return False
    return str(p.expanduser().resolve()) in _CTX.processed


def set_all(val: bool) -> None:
    if _CTX is None:
        return
    for v in _CTX.vars_sel:
        v.set(bool(val))
    update_count()


def invert() -> None:
    if _CTX is None:
        return
    for v in _CTX.vars_sel:
        v.set(not v.get())
    update_count()


def rebuild_list() -> None:
    """Called when show_processed toggles OR after we scan folders."""
    if _CTX is None:
        return
    if not _CTX.img_paths:
        return
    show_proc = bool(_CTX.var_show_processed.get())
    paths_to_show = [p for p in _CTX.img_paths if (show_proc or not is_processed(p))]
    build_rows_incremental(paths_to_show)


def build_rows_incremental(paths_to_show: list[Path]) -> None:
    if _CTX is None:
        return

    clear_list()
    total = len(paths_to_show)
    if total == 0:
        _CTX.status_var.set("No images found in the selected folder.")
        _CTX.btn_run.configure(state="disabled")
        return

    _CTX.status_var.set(f"Loading thumbnails 0/{total}…")

    row_idx = 0

    def add_chunk(start: int) -> None:
        nonlocal row_idx

        end = min(start + 40, total)
        for p in paths_to_show[start:end]:
            proc = is_processed(p)
            v = tk.BooleanVar(value=(not proc))
            _CTX.vars_sel.append(v)
            _CTX.meta.append({"path": p, "processed": proc})

            ph = _CTX.make_thumb(p)
            if ph is not None:
                _CTX.thumbs.append(ph)

            line = ttk.Frame(_CTX.inner)
            line.grid(row=row_idx, column=0, sticky="ew", pady=2)
            line.columnconfigure(3, weight=1)

            ttk.Checkbutton(line, variable=v).grid(row=0, column=0, padx=(0, 6), sticky="w")

            if ph is not None:
                lbl_img = ttk.Label(line, image=ph)
                lbl_img.image = ph
                lbl_img.grid(row=0, column=1, padx=(0, 8), sticky="w")
            else:
                ttk.Label(line, text="(no preview)").grid(row=0, column=1, padx=(0, 8), sticky="w")

            try:
                name = p.relative_to(Path(_CTX.var_in.get()).expanduser().resolve())
            except Exception:
                name = p.name

            ttk.Label(line, text=f"{name}").grid(row=0, column=2, sticky="w")
            ttk.Label(line, text=("processed" if proc else "new")).grid(row=0, column=3, sticky="e")

            row_idx += 1

        _CTX.status_var.set(f"Loading thumbnails {end}/{total}…")
        update_count()
        _CTX.root.update_idletasks()
        _CTX.on_inner_config()

        if end < total:
            aid = _CTX.root.after(1, lambda: add_chunk(end))
            _CTX.after_ids.append(str(aid))
        else:
            _CTX.status_var.set("Ready.")
            _CTX.btn_run.configure(state="normal")

    aid0 = _CTX.root.after(1, lambda: add_chunk(0))
    _CTX.after_ids.append(str(aid0))
