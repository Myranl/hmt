from __future__ import annotations
from pathlib import Path
import json as _json
from PIL import Image, ImageTk
import csv
import os
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk  # type: ignore[import-untyped]

from config import SETTINGS_PATH, RESULTS_SCHEMA_VERSION, RESULTS_META_NAME
from ui.file_selection.actions import browse_in, browse_out, apply_create_inside_state, make_on_cancel_any, make_on_run
from core.validation import _is_subpath, _can_write_dir, _dir_is_empty, _write_results_meta
from ui.file_selection.validation_ui import validate_paths_ui
from ui.file_selection.settings import load_folder_choices, persist_folder_choices
from ui.file_selection.reorganise_result import (
    debug_print_results_head,
    reorganise_results_to_ok_csv,
)
from pipeline.input_scan import ProcessedIndex, load_processed
from ui.file_selection.show_graphs_ui import show_graphs_ui
from ui.categories.category_editor_ui import run_category_editor_ui
from ui.common.tk_after import make_on_destroy
from ui.common.theme import setup_theme, get_base_font, get_small_muted_font
from ui.common.widgets import (
    create_card_frame,
    create_main_panel,
    create_primary_button,
    create_secondary_button,
    create_toolbar_button,
    create_status_label,
)


def run_folder_and_selection_ui(
    *,
    title: str = "Select input folder and images",
) -> tuple[str, str, list[str]] | None:
    """Pick input folder + output folder, then select images via thumbnails.
    Uses CustomTkinter for the UI. Returns (input_dir, out_dir, selected_abs_paths) or None if cancelled.
    """

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    # Global theme (colors / appearance) configured once here
    setup_theme()

    root = ctk.CTk()
    root.title(title)
    

    w = 1280
    h = 720

    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()

    x = (screen_w - w) // 2
    y = (screen_h - h) // 2

    root.geometry(f"{w}x{h}+{x}+{y}")

    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)

    root.configure(fg_color="white")

    # Fonts must be created after root exists
    base_font = get_base_font()
    small_muted_font = get_small_muted_font()

    prev = load_folder_choices()
    prev_in = str(prev.get("input_dir", ""))
    prev_out = str(prev.get("output_dir", ""))
    prev_inside = bool(prev.get("create_output_inside", False))
    prev_subfolders = bool(prev.get("process_subfolders", True))

    var_in = tk.StringVar(master=root, value=prev_in)
    var_out = tk.StringVar(master=root, value=prev_out)
    var_create_inside = tk.BooleanVar(master=root, value=prev_inside)
    var_process_subfolders = tk.BooleanVar(master=root, value=prev_subfolders)

    # Top: folder selection
    folder_frame = create_card_frame(root)
    folder_frame.grid(row=0, column=0, sticky="ew", padx=124, pady=(18, 10))
    folder_frame.columnconfigure(1, weight=1)

    lbl_input = ctk.CTkLabel(folder_frame, text="Input folder (images)", font=base_font)
    lbl_input.grid(row=0, column=0, sticky="w", pady=(8, 0))
    entry_in = ctk.CTkEntry(folder_frame, textvariable=var_in, width=420, state="disabled")
    entry_in.grid(row=0, column=1, sticky="ew", padx=(10, 8), pady=(8, 0))
    btn_browse_in = create_primary_button(folder_frame, text="Browse…")
    btn_browse_in.grid(row=0, column=2, sticky="e", pady=(8, 0))

    lbl_output = ctk.CTkLabel(folder_frame, text="Output folder (results)", font=base_font)
    lbl_output.grid(row=1, column=0, sticky="w", pady=(12, 0))
    entry_out = ctk.CTkEntry(folder_frame, textvariable=var_out, width=420, state="disabled")
    entry_out.grid(row=1, column=1, sticky="ew", padx=(10, 8), pady=(12, 0))
    btn_browse_out = create_primary_button(folder_frame, text="Browse…")
    btn_browse_out.grid(row=1, column=2, sticky="e", pady=(12, 0))

    chk_create_inside = ctk.CTkCheckBox(
        folder_frame,
        text="Create 'output' inside the input folder",
        variable=var_create_inside,
        font=base_font,
    )
    chk_create_inside.grid(row=2, column=0, columnspan=3, sticky="w", pady=(16, 0))

    chk_subfolders = ctk.CTkCheckBox(
        folder_frame,
        text="Process subfolders",
        variable=var_process_subfolders,
        font=base_font,
    )
    chk_subfolders.grid(row=3, column=0, columnspan=3, sticky="w", pady=(10, 0))

    create_status_label(
        folder_frame,
        text=(
            "The output folder will contain (or update) results.csv and last_selection.json. "
            "OK also refreshes result_ok.csv when results.csv is present."
        ),
        wraplength=540,
    ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(12, 0))

    create_status_label(
        folder_frame,
        text="Supported formats: .tif/.tiff, .png, .jpg/.jpeg, .bmp",
        wraplength=540,
    ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(4, 0))

    bar = ctk.CTkFrame(folder_frame, fg_color="transparent")
    bar.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(18, 10))
    bar.columnconfigure(0, weight=1)
    btn_cancel = create_secondary_button(bar, text="Cancel")
    btn_ok = create_primary_button(bar, text="OK", state="disabled")
    btn_cancel.grid(row=0, column=0, sticky="w")
    btn_ok.grid(row=0, column=1, sticky="e")

    # Main area: status + scrollable list + controls
    main_frame = create_main_panel(root)
    main_frame.grid(row=1, column=0, sticky="nsew", padx=124, pady=(0, 18))
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(1, weight=1)

    status_var = tk.StringVar(value="Select folders above, then press OK.")
    status_lbl = create_status_label(main_frame, textvariable=status_var)
    status_lbl.grid(row=0, column=0, sticky="w", pady=(10, 0))

    COLS_PER_ROW = 4
    scroll_list = ctk.CTkScrollableFrame(main_frame, width=600, height=280, fg_color=("white", "gray22"))
    scroll_list.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
    for c in range(COLS_PER_ROW):
        scroll_list.columnconfigure(c, weight=1)

    controls = ctk.CTkFrame(main_frame, fg_color="transparent")
    controls.grid(row=2, column=0, sticky="ew", pady=(12, 0))
    controls.columnconfigure(4, weight=1)

    var_show_processed = tk.BooleanVar(value=True)
    chk_show_processed = ctk.CTkCheckBox(
        controls,
        text="Show processed",
        variable=var_show_processed,
        font=base_font,
    )
    chk_show_processed.grid(row=0, column=0, sticky="w")
    btn_all = create_toolbar_button(controls, text="Select all")
    btn_none = create_toolbar_button(controls, text="Select none")
    btn_invert = create_toolbar_button(controls, text="Invert", width=80)
    btn_all.grid(row=0, column=1, padx=(16, 0))
    btn_none.grid(row=0, column=2, padx=(8, 0))
    btn_invert.grid(row=0, column=3, padx=(8, 0))
    lbl_count = ctk.CTkLabel(controls, text="", font=small_muted_font, text_color="gray40")
    lbl_count.grid(row=0, column=4, sticky="e")

    run_bar = ctk.CTkFrame(main_frame, fg_color="transparent")
    run_bar.grid(row=3, column=0, sticky="ew", pady=(12, 10))
    run_bar.columnconfigure(1, weight=1)
    btn_cancel2 = create_secondary_button(run_bar, text="Cancel")
    btn_categories = create_secondary_button(run_bar, text="Categories…", width=108)
    btn_show_graphs = create_secondary_button(run_bar, text="Show graphs")
    btn_run = create_primary_button(run_bar, text="Run selected", width=116, state="disabled")
    btn_cancel2.grid(row=0, column=0, sticky="w")
    btn_categories.grid(row=0, column=2, sticky="e", padx=(0, 8))
    btn_show_graphs.grid(row=0, column=3, sticky="e", padx=(0, 8))
    btn_run.grid(row=0, column=4, sticky="e")

    thumbs: list[ImageTk.PhotoImage] = []
    vars_sel: list[tk.BooleanVar] = []
    meta: list[dict] = []
    img_paths: list[Path] = []
    processed: ProcessedIndex = ProcessedIndex()
    csv_path: Path | None = None
    result = {"done": False, "selected": []}

    def apply_entry_state() -> None:
        if var_create_inside.get():
            entry_out.configure(state="disabled")
            btn_browse_out.configure(state="disabled")
            if var_in.get().strip():
                p = Path(var_in.get()).expanduser().resolve() / "output"
                var_out.set(str(p))
        else:
            entry_out.configure(state="disabled")
            btn_browse_out.configure(state="normal")
        validate_paths()

    btn_browse_in.configure(command=lambda: browse_in(root, var_in, var_out, var_create_inside, validate_paths))
    btn_browse_out.configure(command=lambda: browse_out(root, var_out, validate_paths))
    chk_create_inside.configure(command=lambda: apply_create_inside_state(var_create_inside, entry_out, btn_browse_out, var_in, var_out, validate_paths))

    def rebuild_list() -> None:
        for w in scroll_list.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
        show_proc = bool(var_show_processed.get())
        shown = 0
        checked = 0
        for i, m in enumerate(meta):
            is_proc = bool(m.get("processed", False))
            if (not show_proc) and is_proc:
                continue
            row_idx = shown // COLS_PER_ROW
            col_idx = shown % COLS_PER_ROW
            card = ctk.CTkFrame(scroll_list, fg_color=("gray96", "gray28"), corner_radius=8)
            card.grid(row=row_idx, column=col_idx, sticky="nsew", pady=4, padx=4)
            card.columnconfigure(0, weight=1)
            cb = ctk.CTkCheckBox(card, variable=vars_sel[i], text="", width=24)
            cb.grid(row=0, column=0, sticky="w", padx=(8, 6), pady=(6, 4))
            row_bg = "#3d3d3d" if ctk.get_appearance_mode() == "Dark" else "#c4c4c4"
            lbl_img = tk.Label(card, image=thumbs[i], bg=row_bg, bd=0)
            lbl_img.grid(row=1, column=0, sticky="n", pady=(0, 4))
            name = m.get("name") or Path(m["path"]).name
            suffix = " (processed)" if is_proc else ""
            ctk.CTkLabel(
                card,
                text=f"{name}{suffix}",
                anchor="center",
                font=base_font,
                wraplength=130,
                justify="center",
            ).grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 6))
            if vars_sel[i].get():
                checked += 1
            shown += 1
        lbl_count.configure(text=f"Selected: {checked} / Shown: {shown} / Total: {len(meta)}")
        btn_run.configure(state="normal" if checked > 0 else "disabled")

    def set_all_local(val: bool) -> None:
        show_proc = bool(var_show_processed.get())
        for i, m in enumerate(meta):
            if (not show_proc) and bool(m.get("processed", False)):
                continue
            vars_sel[i].set(bool(val))
        rebuild_list()

    def invert_local() -> None:
        show_proc = bool(var_show_processed.get())
        for i, m in enumerate(meta):
            if (not show_proc) and bool(m.get("processed", False)):
                continue
            vars_sel[i].set(not vars_sel[i].get())
        rebuild_list()

    btn_all.configure(command=lambda: set_all_local(True))
    btn_none.configure(command=lambda: set_all_local(False))
    btn_invert.configure(command=invert_local)
    var_show_processed.trace_add("write", lambda *_: rebuild_list())

    on_cancel_any = make_on_cancel_any(root=root, result=result)
    btn_cancel.configure(command=on_cancel_any)
    btn_cancel2.configure(command=on_cancel_any)

    def iter_images(in_path: Path, *, subfolders: bool) -> list[Path]:
        if subfolders:
            files = [p for p in in_path.rglob("*") if p.is_file() and p.suffix.lower() in exts]
        else:
            files = [p for p in in_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
        files.sort(key=lambda p: str(p).lower())
        return files

    def on_ok() -> None:
        nonlocal img_paths, processed, csv_path
        persist_folder_choices(
            input_dir=var_in.get().strip(),
            output_dir=var_out.get().strip(),
            create_output_inside=bool(var_create_inside.get()),
            process_subfolders=bool(var_process_subfolders.get()),
        )
        in_path = Path(var_in.get().strip()).expanduser().resolve()
        out_path = Path(var_out.get().strip()).expanduser().resolve()
        out_path.mkdir(parents=True, exist_ok=True)
        _write_results_meta(out_path, schema_version=RESULTS_SCHEMA_VERSION)
        res_csv = out_path / "results.csv"
        if res_csv.exists():
            debug_print_results_head(str(out_path), rows=5)
            ok_reorg, _reorg_path, _reorg_rows = reorganise_results_to_ok_csv(str(out_path))
            if not ok_reorg:
                messagebox.showwarning(
                    "Reorganise result",
                    "Could not save result_ok.csv.\n\n"
                    "Typical on Windows: the file is open in Excel (or another program) — close it.\n"
                    "Also check: folder not read-only; OneDrive/antivirus not locking the file.\n"
                    "Details are printed in the terminal.",
                    parent=root,
                )
        csv_path = out_path / "results.csv"
        processed = load_processed(csv_path)
        img_paths = iter_images(in_path, subfolders=bool(var_process_subfolders.get()))
        if not img_paths:
            status_var.set("No images found in the selected folder.")
            return
        btn_ok.configure(state="disabled")
        btn_browse_in.configure(state="disabled")
        btn_browse_out.configure(state="disabled")
        chk_create_inside.configure(state="disabled")
        status_var.set(f"Found {len(img_paths)} images. Loading thumbnails 0/{len(img_paths)}…")
        root.update_idletasks()
        thumbs.clear()
        vars_sel.clear()
        meta.clear()

        def make_thumb(p: Path, *, size: int = 120) -> ImageTk.PhotoImage:
            im = Image.open(p)
            im = im.convert("RGB")
            w, h = im.size
            s = max(w, h)
            if s > size:
                scale = size / float(s)
                im = im.resize((int(round(w * scale)), int(round(h * scale))), resample=Image.Resampling.BILINEAR)
            return ImageTk.PhotoImage(im)

        def load_one(i: int) -> None:
            if i >= len(img_paths):
                status_var.set(f"Loaded {len(img_paths)} thumbnails. Select images and press Run selected.")
                btn_browse_in.configure(state="normal")
                btn_browse_out.configure(state="normal")
                chk_create_inside.configure(state="normal")
                btn_ok.configure(state="normal")
                rebuild_list()
                return
            p = img_paths[i]
            p_res = str(p.expanduser().resolve())
            is_proc = processed.matches(p_res)
            try:
                thumbs.append(make_thumb(p))
            except Exception:
                ph = Image.new("RGB", (120, 120), (200, 200, 200))
                thumbs.append(ImageTk.PhotoImage(ph))
            v = tk.BooleanVar(value=(not is_proc))
            v.trace_add("write", lambda *_: rebuild_list())
            vars_sel.append(v)
            meta.append({"path": p_res, "name": p.name, "processed": is_proc})
            status_var.set(f"Found {len(img_paths)} images. Loading thumbnails {i+1}/{len(img_paths)}…")
            aid = root.after(1, lambda: load_one(i + 1))
            after_ids.append(aid)
            if (i % 20) == 0:
                rebuild_list()

        load_one(0)

    btn_ok.configure(command=on_ok)
    btn_run.configure(
        command=make_on_run(
            root=root,
            result=result,
            vars_sel=vars_sel,
            meta=meta,
            var_in=var_in,
            var_out=var_out,
            json_module=_json,
        )
    )

    def on_show_graphs() -> None:
        out_dir = var_out.get().strip()
        if not out_dir:
            messagebox.showwarning("Show graphs", "Please choose an output folder first.", parent=root)
            return
        show_graphs_ui(root, out_dir)

    btn_show_graphs.configure(command=on_show_graphs)

    def on_categories() -> None:
        in_dir = var_in.get().strip()
        out_dir = var_out.get().strip()
        if not in_dir or not out_dir:
            messagebox.showwarning(
                "Categories",
                "Please choose both input and output folders first.",
                parent=root,
            )
            return
        run_category_editor_ui(root, input_dir=in_dir, output_dir=out_dir)

    btn_categories.configure(command=on_categories)

    def validate_paths(*_args) -> bool:
        return validate_paths_ui(
            var_in=var_in,
            var_out=var_out,
            var_create_inside=var_create_inside,
            status_var=status_var,
            btn_ok=btn_ok,
            can_write_dir=_can_write_dir,
            is_subpath=_is_subpath,
        )

    var_in.trace_add("write", validate_paths)
    var_out.trace_add("write", validate_paths)
    var_create_inside.trace_add("write", validate_paths)

    after_ids: list[str] = []
    root.bind("<Destroy>", make_on_destroy(root=root, after_ids=after_ids))

    apply_create_inside_state(var_create_inside, entry_out, btn_browse_out, var_in, var_out, validate_paths)
    validate_paths()

    root.mainloop()

    if not result.get("done", False):
        return None
    return (var_in.get().strip(), var_out.get().strip(), list(result.get("selected", [])))
