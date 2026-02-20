from __future__ import annotations
from pathlib import Path
import json as _jso
from PIL import Image, ImageTk
import csv
import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

SETTINGS_PATH = Path.home() / ".hipoca_histo_last_selection.json"

RESULTS_SCHEMA_VERSION = 1
RESULTS_META_NAME = "results.meta.json"

def run_folder_and_selection_ui(
    *,
    title: str = "Select input folder and images",
) -> tuple[str, str, list[str]] | None:
    """Pick input folder + output folder, then select images via thumbnails.
    Already-processed images (status=="ok" in <out_dir>/results.csv) are unchecked by default.
    Returns: (input_dir, out_dir, selected_abs_paths) or None if cancelled.
    """

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    def _is_subpath(child: Path, parent: Path) -> bool:
        try:
            child.resolve().relative_to(parent.resolve())
            return True
        except Exception:
            return False

    def _dir_is_empty(p: Path) -> bool:
        try:
            return not any(p.iterdir())
        except Exception:
            return False

    def _can_write_dir(p: Path) -> bool:
        try:
            p.mkdir(parents=True, exist_ok=True)
            test = p / ".__hmt_write_test__"
            test.write_text("ok", encoding="utf-8")
            test.unlink(missing_ok=True)
            return True
        except Exception:
            return False

    def _load_results_meta(out_dir: Path) -> dict:
        mp = out_dir / RESULTS_META_NAME
        if not mp.exists():
            return {}
        try:
            return _json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_results_meta(out_dir: Path, *, schema_version: int) -> None:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            mp = out_dir / RESULTS_META_NAME
            _json.dump(
                {"tool": "HMT", "schema_version": int(schema_version)},
                mp.open("w", encoding="utf-8"),
                indent=2,
                ensure_ascii=False,
            )
        except Exception:
            pass

    def _is_ours_output(out_dir: Path) -> bool:
        # Heuristic: if it has our meta file OR last_selection.json OR results.csv
        if (out_dir / RESULTS_META_NAME).exists():
            return True
        if (out_dir / "last_selection.json").exists():
            return True
        if (out_dir / "results.csv").exists():
            return True
        return False

    # Load previous folder choices (optional)
    prev_in = ""
    prev_out = ""
    prev_inside = False
    prev_subfolders = True
    try:
        if SETTINGS_PATH.exists():
            data = _json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            prev_in = str(data.get("input_dir") or "")
            prev_out = str(data.get("output_dir") or "")
            prev_inside = bool(data.get("create_output_inside") or False)
            prev_subfolders = bool(data.get("process_subfolders") if "process_subfolders" in data else True)
    except Exception:
        pass

    def load_processed(csv_path: Path) -> set[str]:
        processed: set[str] = set()
        if not csv_path.exists():
            return processed
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    if not row:
                        continue
                    p = row.get("image_path")
                    st = (row.get("status") or "").strip().lower()
                    if p and st == "ok":
                        processed.add(str(Path(p).expanduser().resolve()))
        except Exception:
            return processed
        return processed

    def iter_images(root_dir: Path, *, subfolders: bool) -> list[Path]:
        # os.walk is usually faster than Path.rglob for big folders
        out: list[Path] = []
        if not subfolders:
            for p in root_dir.iterdir():
                if p.is_file() and p.suffix.lower() in exts:
                    out.append(p)
        else:
            for dp, _dns, fns in os.walk(root_dir):
                for fn in fns:
                    p = Path(dp) / fn
                    if p.suffix.lower() in exts:
                        out.append(p)
        out.sort(key=lambda p: str(p).lower())
        return out

    def make_thumb(p: Path, size: int = 120) -> ImageTk.PhotoImage | None:
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((size, size))
            return ImageTk.PhotoImage(im)
        except Exception:
            return None

    # --- Single window UI ---
    root = tk.Tk()
    root.title(title)

    # Top area: folder selection
    folder_frame = ttk.Frame(root, padding=10)
    folder_frame.grid(row=0, column=0, sticky="ew")
    root.columnconfigure(0, weight=1)

    var_in = tk.StringVar(value=prev_in)
    var_out = tk.StringVar(value=prev_out)
    var_create_inside = tk.BooleanVar(value=prev_inside)
    var_process_subfolders = tk.BooleanVar(value=prev_subfolders)

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

    def set_enabled(widgets: list[tk.Widget], enabled: bool) -> None:
        st = "normal" if enabled else "disabled"
        for w in widgets:
            try:
                w.configure(state=st)
            except Exception:
                pass

    def update_ok_button_state() -> None:
        in_ok = bool(var_in.get().strip())
        out_ok = True if var_create_inside.get() else bool(var_out.get().strip())
        btn_ok.configure(state="normal" if (in_ok and out_ok) else "disabled")

    def apply_create_inside_state() -> None:
        if var_create_inside.get():
            entry_out.configure(state="disabled")
            btn_browse_out.configure(state="disabled")
            if var_in.get().strip():
                p = Path(var_in.get()).expanduser().resolve() / "output"
                var_out.set(str(p))
        else:
            entry_out.configure(state="readonly")
            btn_browse_out.configure(state="normal")
        update_ok_button_state()

    def browse_in() -> None:
        d = filedialog.askdirectory(title="Select input folder (images)", parent=root)
        if d:
            var_in.set(d)
            if var_create_inside.get():
                p = Path(d).expanduser().resolve() / "output"
                var_out.set(str(p))
        update_ok_button_state()

    def browse_out() -> None:
        d = filedialog.askdirectory(title="Select output folder (results)", parent=root)
        if d:
            var_out.set(d)
        update_ok_button_state()

    btn_browse_in.configure(command=browse_in)
    btn_browse_out.configure(command=browse_out)
    chk_create_inside.configure(command=apply_create_inside_state)

    def is_processed(p: Path) -> bool:
        return str(p.expanduser().resolve()) in processed

    def update_count() -> None:
        total = len(vars_sel)
        sel = sum(1 for v in vars_sel if v.get())
        lbl_count.configure(text=f"selected {sel}/{total}")

    def clear_list() -> None:
        for w in inner.winfo_children():
            w.destroy()
        thumbs.clear()
        vars_sel.clear()
        meta.clear()
        update_count()
        on_inner_config()

    def build_rows_incremental(paths_to_show: list[Path]) -> None:
        clear_list()
        total = len(paths_to_show)
        if total == 0:
            status_var.set("No images found in the selected folder.")
            btn_run.configure(state="disabled")
            return

        status_var.set(f"Loading thumbnails 0/{total}…")

        row_idx = 0

        def add_chunk(start: int) -> None:
            nonlocal row_idx
            end = min(start + 40, total)
            for p in paths_to_show[start:end]:
                proc = is_processed(p)
                v = tk.BooleanVar(value=(not proc))
                vars_sel.append(v)
                meta.append({"path": p, "processed": proc})

                ph = make_thumb(p)
                if ph is not None:
                    thumbs.append(ph)

                line = ttk.Frame(inner)
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
                    name = p.relative_to(Path(var_in.get()).expanduser().resolve())
                except Exception:
                    name = p.name

                ttk.Label(line, text=f"{name}").grid(row=0, column=2, sticky="w")
                ttk.Label(line, text=("processed" if proc else "new")).grid(row=0, column=3, sticky="e")

                row_idx += 1

            status_var.set(f"Loading thumbnails {end}/{total}…")
            update_count()
            root.update_idletasks()
            on_inner_config()

            if end < total:
                root.after(1, lambda: add_chunk(end))
            else:
                status_var.set("Ready.")
                btn_run.configure(state="normal")

        root.after(1, lambda: add_chunk(0))

    def rebuild_list() -> None:
        # Called when show_processed toggles
        if not img_paths:
            return
        show_proc = bool(var_show_processed.get())
        paths_to_show = [p for p in img_paths if (show_proc or not is_processed(p))]
        build_rows_incremental(paths_to_show)

    def set_all(val: bool) -> None:
        for v in vars_sel:
            v.set(bool(val))
        update_count()

    def invert() -> None:
        for v in vars_sel:
            v.set(not v.get())
        update_count()

    btn_all.configure(command=lambda: set_all(True))
    btn_none.configure(command=lambda: set_all(False))
    btn_invert.configure(command=invert)

    var_show_processed.trace_add("write", lambda *_: rebuild_list())

    def persist_folder_choices() -> None:
        try:
            in_dir_tmp = var_in.get().strip()
            out_dir_tmp = var_out.get().strip()
            _json.dump(
                {
                    "input_dir": in_dir_tmp,
                    "output_dir": out_dir_tmp,
                    "create_output_inside": bool(var_create_inside.get()),
                    "process_subfolders": bool(var_process_subfolders.get()),
                },
                SETTINGS_PATH.open("w", encoding="utf-8"),
                indent=2,
                ensure_ascii=False,
            )
        except Exception:
            pass

    def on_cancel_any() -> None:
        result["done"] = False
        result["selected"] = []
        root.destroy()

    btn_cancel.configure(command=on_cancel_any)
    btn_cancel2.configure(command=on_cancel_any)

    def on_ok() -> None:
        nonlocal img_paths, processed, csv_path

        persist_folder_choices()

        in_dir = var_in.get().strip()
        out_dir = var_out.get().strip()
        in_path = Path(in_dir).expanduser().resolve()
        out_path = Path(out_dir).expanduser().resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        # Conflicts / warnings
        if _is_subpath(out_path, in_path) and not _dir_is_empty(out_path) and not _is_ours_output(out_path):
            go = messagebox.askyesno(
                "Output folder not empty",
                "The output folder is inside the input folder and is not empty.\n"
                "It may already contain unrelated files.\n\nContinue anyway?",
                parent=root,
            )
            if not go:
                set_enabled([btn_browse_in, btn_browse_out, chk_create_inside, btn_ok], True)
                apply_create_inside_state()
                return

        if out_path.exists() and not _dir_is_empty(out_path) and not _is_ours_output(out_path):
            go = messagebox.askyesno(
                "Output folder not empty",
                "The output folder is not empty and does not look like an existing HMT project.\n"
                "Existing files may be overwritten for selected images.\n\nContinue anyway?",
                parent=root,
            )
            if not go:
                set_enabled([btn_browse_in, btn_browse_out, chk_create_inside, btn_ok], True)
                apply_create_inside_state()
                return

        # Results schema/version check
        csv_default = out_path / "results.csv"
        meta = _load_results_meta(out_path)
        if csv_default.exists():
            existing_ver = int(meta.get("schema_version") or 0)
            if existing_ver and existing_ver != RESULTS_SCHEMA_VERSION:
                ans = messagebox.askyesnocancel(
                    "Results version mismatch",
                    f"An existing results.csv was created with schema version {existing_ver}, "
                    f"but this tool expects version {RESULTS_SCHEMA_VERSION}.\n\n"
                    "Yes: use existing results.csv (best effort).\n"
                    "No: create a new results_vX.csv in the output folder.\n"
                    "Cancel: abort.",
                    parent=root,
                )
                if ans is None:
                    set_enabled([btn_browse_in, btn_browse_out, chk_create_inside, btn_ok], True)
                    apply_create_inside_state()
                    return
                if ans is False:
                    csv_default = out_path / f"results_v{RESULTS_SCHEMA_VERSION}.csv"

        csv_path = csv_default
        _write_results_meta(out_path, schema_version=RESULTS_SCHEMA_VERSION)
        processed = load_processed(csv_path)

        # Disable folder selection while loading
        set_enabled([btn_browse_in, btn_browse_out, chk_create_inside, btn_ok], False)
        set_enabled([chk_show_processed, btn_all, btn_none, btn_invert, btn_run], False)

        status_var.set("Please wait, scanning folder…")
        root.update_idletasks()

        img_paths = iter_images(in_path, subfolders=bool(var_process_subfolders.get()))
        if not img_paths:
            status_var.set("No images found in the selected folder.")
            set_enabled([btn_browse_in, btn_browse_out, chk_create_inside, btn_ok], True)
            apply_create_inside_state()
            return

        n_proc = sum(1 for p in img_paths if is_processed(p))
        n_new = len(img_paths) - n_proc
        status_var.set(f"Found {len(img_paths)} images ({n_new} new, {n_proc} processed). Loading thumbnails 0/{len(img_paths)}…")

        # Re-enable selection controls; rows will appear incrementally
        set_enabled([chk_show_processed, btn_all, btn_none, btn_invert], True)
        rebuild_list()

    btn_ok.configure(command=on_ok)

    def on_run() -> None:
        selected: list[str] = []
        for v, m in zip(vars_sel, meta):
            if v.get():
                selected.append(str(Path(m["path"]).expanduser().resolve()))
        result["done"] = True
        result["selected"] = selected

        # Save last selection for reproducibility (best-effort)
        try:
            if var_out.get().strip():
                out_path = Path(var_out.get()).expanduser().resolve()
                out_path.mkdir(parents=True, exist_ok=True)
                save_path = out_path / "last_selection.json"
                with save_path.open("w", encoding="utf-8") as f:
                    _json.dump(
                        {
                            "input_dir": str(Path(var_in.get()).expanduser().resolve()),
                            "output_dir": str(out_path),
                            "selected": selected,
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
        except Exception:
            pass

        root.destroy()

    btn_run.configure(command=on_run)

    def validate_paths(*_args) -> bool:
        in_dir = var_in.get().strip()
        out_dir = var_out.get().strip()

        if not in_dir:
            status_var.set("Select an input folder.")
            btn_ok.configure(state="disabled")
            return False

        in_path = Path(in_dir).expanduser()
        if not in_path.exists() or not in_path.is_dir():
            status_var.set("Input folder does not exist or is not a directory.")
            btn_ok.configure(state="disabled")
            return False

        # Output folder handling
        if var_create_inside.get():
            out_path = in_path.resolve() / "output"
            var_out.set(str(out_path))
        else:
            if not out_dir:
                status_var.set("Select an output folder.")
                btn_ok.configure(state="disabled")
                return False
            out_path = Path(out_dir).expanduser().resolve()

        in_res = in_path.resolve()
        out_res = out_path.resolve()

        if in_res == out_res:
            status_var.set("Invalid setup: output folder cannot be the same as input folder.")
            btn_ok.configure(state="disabled")
            return False

        if not _can_write_dir(out_res):
            status_var.set("Cannot write to output folder (permission denied or read-only location).")
            btn_ok.configure(state="disabled")
            return False

        # Soft warnings (do not disable OK)
        if _is_subpath(out_res, in_res):
            # This is allowed, but it should be deliberate
            status_var.set("Warning: output is inside input. Results will be stored alongside raw data.")
        else:
            status_var.set("Ready to scan. Press OK.")

        btn_ok.configure(state="normal")
        return True

    def update_ok_button_state() -> None:
        # replaced by validate_paths
        pass

    def apply_create_inside_state() -> None:
        if var_create_inside.get():
            entry_out.configure(state="disabled")
            btn_browse_out.configure(state="disabled")
            if var_in.get().strip():
                p = Path(var_in.get()).expanduser().resolve() / "output"
                var_out.set(str(p))
        else:
            entry_out.configure(state="readonly")
            btn_browse_out.configure(state="normal")
        validate_paths()

    def browse_in() -> None:
        d = filedialog.askdirectory(title="Select input folder (images)", parent=root)
        if d:
            var_in.set(d)
            if var_create_inside.get():
                p = Path(d).expanduser().resolve() / "output"
                var_out.set(str(p))
        validate_paths()

    def browse_out() -> None:
        d = filedialog.askdirectory(title="Select output folder (results)", parent=root)
        if d:
            var_out.set(d)
        validate_paths()

    btn_browse_in.configure(command=browse_in)
    btn_browse_out.configure(command=browse_out)
    chk_create_inside.configure(command=apply_create_inside_state)

    var_in.trace_add("write", validate_paths)
    var_out.trace_add("write", validate_paths)
    var_create_inside.trace_add("write", validate_paths)

    # Prevent "invalid command name ... after" errors by cancelling any scheduled work
    after_ids: list[str] = []

    def safe_after(ms: int, fn):
        aid = root.after(ms, fn)
        after_ids.append(aid)
        return aid

    def on_destroy(_ev=None):
        for aid in after_ids:
            try:
                root.after_cancel(aid)
            except Exception:
                pass

    root.bind("<Destroy>", on_destroy)

    # Patch build_rows_incremental to use safe_after
    def build_rows_incremental(paths_to_show: list[Path]) -> None:
        clear_list()
        total = len(paths_to_show)
        if total == 0:
            status_var.set("No images found in the selected folder.")
            btn_run.configure(state="disabled")
            return

        status_var.set(f"Loading thumbnails 0/{total}…")

        row_idx = 0

        def add_chunk(start: int) -> None:
            nonlocal row_idx
            end = min(start + 40, total)
            for p in paths_to_show[start:end]:
                proc = is_processed(p)
                v = tk.BooleanVar(value=(not proc))
                vars_sel.append(v)
                meta.append({"path": p, "processed": proc})

                ph = make_thumb(p)
                if ph is not None:
                    thumbs.append(ph)

                line = ttk.Frame(inner)
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
                    name = p.relative_to(Path(var_in.get()).expanduser().resolve())
                except Exception:
                    name = p.name

                ttk.Label(line, text=f"{name}").grid(row=0, column=2, sticky="w")
                ttk.Label(line, text=("processed" if proc else "new")).grid(row=0, column=3, sticky="e")

                row_idx += 1

            status_var.set(f"Loading thumbnails {end}/{total}…")
            update_count()
            root.update_idletasks()
            on_inner_config()

            if end < total:
                safe_after(1, lambda: add_chunk(end))
            else:
                status_var.set("Ready.")
                btn_run.configure(state="normal")

        safe_after(1, lambda: add_chunk(0))

    # Override the previous function reference
    globals()["build_rows_incremental"] = build_rows_incremental

    apply_create_inside_state()
    validate_paths()

    root.mainloop()

    if not result.get("done", False):
        return None

    in_final = var_in.get().strip()
    out_final = var_out.get().strip()
    return (in_final, out_final, list(result.get("selected", [])))
