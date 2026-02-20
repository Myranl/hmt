from __future__ import annotations
from pathlib import Path
import json as _json
from PIL import Image, ImageTk
import csv
import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from config import SETTINGS_PATH, RESULTS_SCHEMA_VERSION, RESULTS_META_NAME
from ui.selection_folder.actions import browse_in, browse_out, apply_create_inside_state, make_on_cancel_any, make_on_run
from core.validation import _is_subpath, _can_write_dir, _dir_is_empty, _write_results_meta
from ui.selection_folder.validation_ui import validate_paths_ui
from ui.common.tk_after import make_on_destroy
from ui.selection_folder.settings import load_folder_choices, persist_folder_choices

def run_folder_and_selection_ui(
    *,
    title: str = "Select input folder and images",
) -> tuple[str, str, list[str]] | None:
    """Pick input folder + output folder, then select images via thumbnails.
    Already-processed images (status=="ok" in <out_dir>/results.csv) are unchecked by default.
    Returns: (input_dir, out_dir, selected_abs_paths) or None if cancelled.
    """

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}



    # --- Single window UI ---
    root = tk.Tk()
    root.title(title)

    prev = load_folder_choices()

    prev_in = str(prev.get("input_dir", ""))
    prev_out = str(prev.get("output_dir", ""))
    prev_inside = bool(prev.get("create_output_inside", False))
    prev_subfolders = bool(prev.get("process_subfolders", True))

    var_in = tk.StringVar(master=root, value=prev_in)
    var_out = tk.StringVar(master=root, value=prev_out)
    var_create_inside = tk.BooleanVar(master=root, value=prev_inside)
    var_process_subfolders = tk.BooleanVar(master=root, value=prev_subfolders)

    # Top area: folder selection
    folder_frame = ttk.Frame(root, padding=10)
    folder_frame.grid(row=0, column=0, sticky="ew")
    root.columnconfigure(0, weight=1)

    # Middle/bottom area: selection list (disabled until folders chosen)
    main_frame = ttk.Frame(root, padding=10)
    main_frame.grid(row=1, column=0, sticky="nsew")
    root.rowconfigure(1, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(1, weight=1)

    # Status/placeholder
    status_var = tk.StringVar(value="Select folders above, then press OK.")
    status_lbl = ttk.Label(main_frame, textvariable=status_var, foreground="gray")
    status_lbl.grid(row=0, column=0, sticky="w")

    # Scrollable list
    canvas = tk.Canvas(main_frame, highlightthickness=0)
    canvas.grid(row=1, column=0, sticky="nsew", pady=(10, 0))

    vsb = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    vsb.grid(row=1, column=1, sticky="ns", pady=(10, 0))
    canvas.configure(yscrollcommand=vsb.set)

    inner = ttk.Frame(canvas)
    win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

    def on_inner_config(_ev=None):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def on_canvas_config(ev):
        canvas.itemconfigure(win_id, width=ev.width)

    inner.bind("<Configure>", on_inner_config)
    canvas.bind("<Configure>", on_canvas_config)

    # Controls above the list
    controls = ttk.Frame(main_frame)
    controls.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
    controls.columnconfigure(6, weight=1)

    lbl_in = ttk.Label(folder_frame, text="Input folder (images):")
    lbl_in.grid(row=0, column=0, sticky="w")
    entry_in = ttk.Entry(folder_frame, textvariable=var_in, state="readonly", width=60)
    entry_in.grid(row=0, column=1, sticky="ew", padx=(8, 8))
    btn_browse_in = ttk.Button(folder_frame, text="Browse…")
    btn_browse_in.grid(row=0, column=2, sticky="e")

    lbl_out = ttk.Label(folder_frame, text="Output folder (results):")
    lbl_out.grid(row=1, column=0, sticky="w", pady=(8, 0))
    entry_out = ttk.Entry(folder_frame, textvariable=var_out, state="readonly", width=60)
    entry_out.grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=(8, 0))
    btn_browse_out = ttk.Button(folder_frame, text="Browse…")
    btn_browse_out.grid(row=1, column=2, sticky="e", pady=(8, 0))

    chk_create_inside = ttk.Checkbutton(
        folder_frame,
        text="Create 'output' inside the input folder",
        variable=var_create_inside,
    )
    chk_create_inside.grid(row=2, column=0, columnspan=3, sticky="w", pady=(10, 0))

    chk_subfolders = ttk.Checkbutton(
        folder_frame,
        text="Process subfolders",
        variable=var_process_subfolders,
    )
    chk_subfolders.grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))

    help_lbl = ttk.Label(
        folder_frame,
        text="The output folder will contain (or update) results.csv and last_selection.json.",
        foreground="gray",
        wraplength=520,
        justify="left",
    )
    help_lbl.grid(row=4, column=0, columnspan=3, sticky="w", pady=(2, 0))

    fmt_lbl = ttk.Label(
        folder_frame,
        text="Supported formats: .tif/.tiff, .png, .jpg/.jpeg, .bmp",
        foreground="gray",
        wraplength=520,
        justify="left",
    )
    fmt_lbl.grid(row=5, column=0, columnspan=3, sticky="w", pady=(2, 0))

    bar = ttk.Frame(folder_frame)
    bar.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(10, 0))
    bar.columnconfigure(0, weight=1)

    btn_cancel = ttk.Button(bar, text="Cancel")
    btn_ok = ttk.Button(bar, text="OK", state="disabled")
    btn_cancel.grid(row=0, column=0, sticky="w")
    btn_ok.grid(row=0, column=1, sticky="e")

    folder_frame.columnconfigure(1, weight=1)

    var_show_processed = tk.BooleanVar(value=True)
    chk_show_processed = ttk.Checkbutton(controls, text="show processed", variable=var_show_processed)
    chk_show_processed.grid(row=0, column=0, sticky="w")

    btn_all = ttk.Button(controls, text="Select all")
    btn_none = ttk.Button(controls, text="Select none")
    btn_invert = ttk.Button(controls, text="Invert")
    btn_all.grid(row=0, column=1, padx=(12, 0))
    btn_none.grid(row=0, column=2, padx=(6, 0))
    btn_invert.grid(row=0, column=3, padx=(6, 0))

    lbl_count = ttk.Label(controls, text="")
    lbl_count.grid(row=0, column=6, sticky="e")

    run_bar = ttk.Frame(main_frame)
    run_bar.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
    run_bar.columnconfigure(0, weight=1)
    btn_cancel2 = ttk.Button(run_bar, text="Cancel")
    btn_run = ttk.Button(run_bar, text="Run selected", state="disabled")
    btn_cancel2.grid(row=0, column=0, sticky="w")
    btn_run.grid(row=0, column=1, sticky="e")

    # State
    thumbs: list[ImageTk.PhotoImage] = []
    vars_sel: list[tk.BooleanVar] = []
    meta: list[dict] = []
    img_paths: list[Path] = []
    processed: set[str] = set()
    csv_path: Path | None = None

    result = {"done": False, "selected": []}

    btn_browse_in.configure(command=lambda: browse_in(root, var_in, var_out, var_create_inside, validate_paths))
    btn_browse_out.configure(command=lambda: browse_out(root, var_out, validate_paths))
    chk_create_inside.configure(command=lambda: apply_create_inside_state(var_create_inside, entry_out, btn_browse_out, var_in, var_out, validate_paths))

    def rebuild_list() -> None:
        # Clear previous rows
        for w in inner.winfo_children():
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

            row = ttk.Frame(inner, padding=(2, 2))
            row.grid(row=shown, column=0, sticky="ew")
            row.columnconfigure(2, weight=1)

            cb = ttk.Checkbutton(row, variable=vars_sel[i])
            cb.grid(row=0, column=0, sticky="w")

            # thumb
            lbl_img = ttk.Label(row, image=thumbs[i])
            lbl_img.grid(row=0, column=1, sticky="w", padx=(8, 8))

            name = m.get("name") or Path(m["path"]).name
            suffix = " (processed)" if is_proc else ""
            lbl_txt = ttk.Label(row, text=f"{name}{suffix}")
            lbl_txt.grid(row=0, column=2, sticky="w")

            if vars_sel[i].get():
                checked += 1
            shown += 1

        # Count label
        lbl_count.configure(text=f"Selected: {checked} / Shown: {shown} / Total: {len(meta)}")

        # Enable run if anything is selected
        btn_run.configure(state=("normal" if checked > 0 else "disabled"))

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

    def load_processed(csv_path: Path) -> set[str]:
        if not csv_path.exists():
            return set()
        out: set[str] = set()
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                rd = csv.DictReader(f)
                for row in rd:
                    if (row.get("status") or "").strip().lower() == "ok":
                        p = (row.get("image_path") or "").strip()
                        if p:
                            out.add(str(Path(p).expanduser().resolve()))
        except Exception:
            return set()
        return out

    def on_ok() -> None:
        nonlocal img_paths, processed, csv_path

        # Persist last choice (best effort)
        persist_folder_choices(
            input_dir=var_in.get().strip(),
            output_dir=var_out.get().strip(),
            create_output_inside=bool(var_create_inside.get()),
            process_subfolders=bool(var_process_subfolders.get()),
        )

        in_path = Path(var_in.get().strip()).expanduser().resolve()
        out_path = Path(var_out.get().strip()).expanduser().resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        # results meta
        _write_results_meta(out_path, schema_version=RESULTS_SCHEMA_VERSION)

        # results csv
        csv_path = out_path / "results.csv"
        processed = load_processed(csv_path)

        # Scan images
        img_paths = iter_images(in_path, subfolders=bool(var_process_subfolders.get()))
        if not img_paths:
            status_var.set("No images found in the selected folder.")
            return

        # Disable top controls while loading
        btn_ok.configure(state="disabled")
        btn_browse_in.configure(state="disabled")
        btn_browse_out.configure(state="disabled")
        chk_create_inside.configure(state="disabled")

        status_var.set(f"Found {len(img_paths)} images. Loading thumbnails 0/{len(img_paths)}…")
        root.update_idletasks()

        # Reset list state
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
            is_proc = p_res in processed

            try:
                thumbs.append(make_thumb(p))
            except Exception:
                # fallback empty thumb
                ph = Image.new("RGB", (120, 120), (200, 200, 200))
                thumbs.append(ImageTk.PhotoImage(ph))

            v = tk.BooleanVar(value=(not is_proc))
            vars_sel.append(v)
            meta.append({"path": p_res, "name": p.name, "processed": is_proc})

            status_var.set(f"Found {len(img_paths)} images. Loading thumbnails {i+1}/{len(img_paths)}…")

            # Schedule next
            aid = root.after(1, lambda: load_one(i + 1))
            after_ids.append(aid)

            # Render progressively every ~20 items for responsiveness
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

    # Prevent "invalid command name ... after" errors by cancelling any scheduled work
    after_ids: list[str] = []
    root.bind("<Destroy>", make_on_destroy(root=root, after_ids=after_ids))


    apply_create_inside_state(var_create_inside, entry_out, btn_browse_out, var_in, var_out, validate_paths)
    validate_paths()

    root.mainloop()

    if not result.get("done", False):
        return None

    in_final = var_in.get().strip()
    out_final = var_out.get().strip()

    return (in_final, out_final, list(result.get("selected", [])))
