from __future__ import annotations

from pathlib import Path
import json as _json

SETTINGS_PATH = Path.home() / ".hipoca_histo_last_selection.json"


def run_folder_and_selection_ui(
    *,
    title: str = "Select input folder and images",
) -> tuple[str, str, list[str]] | None:
    """Pick input folder + output folder, then select images via thumbnails.

    Already-processed images (status=="ok" in <out_dir>/results.csv) are unchecked by default.

    Returns: (input_dir, out_dir, selected_abs_paths) or None if cancelled.
    """

    import csv
    import tkinter as tk
    from tkinter import filedialog, ttk

    from PIL import Image, ImageTk

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    # Load previous folder choices (optional)
    prev_in = ""
    prev_out = ""
    prev_inside = False
    try:
        if SETTINGS_PATH.exists():
            data = _json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            prev_in = str(data.get("input_dir") or "")
            prev_out = str(data.get("output_dir") or "")
            prev_inside = bool(data.get("create_output_inside") or False)
    except Exception:
        pass

    def iter_images(root: Path) -> list[Path]:
        paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
        paths.sort(key=lambda p: str(p).lower())
        return paths

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

    # --- Step 1: choose folders (UI) ---
    root0 = tk.Tk()
    root0.title("Select folders")

    var_in = tk.StringVar(value=prev_in)
    var_out = tk.StringVar(value=prev_out)
    var_create_inside = tk.BooleanVar(value=prev_inside)

    def browse_in():
        d = filedialog.askdirectory(title="Select input folder (images)", parent=root0)
        if d:
            var_in.set(d)
            if var_create_inside.get():
                path = Path(d).expanduser().resolve() / "output"
                var_out.set(str(path))
            update_ok_button_state()

    def browse_out():
        d = filedialog.askdirectory(title="Select output folder (results)", parent=root0)
        if d:
            var_out.set(d)
            update_ok_button_state()

    def on_create_inside_toggle():
        if var_create_inside.get():
            entry_out.configure(state="disabled")
            btn_browse_out.configure(state="disabled")
            if var_in.get():
                path = Path(var_in.get()).expanduser().resolve() / "output"
                var_out.set(str(path))
        else:
            entry_out.configure(state="normal")
            btn_browse_out.configure(state="normal")
        update_ok_button_state()

    def update_ok_button_state():
        in_ok = bool(var_in.get())
        out_ok = True if var_create_inside.get() else bool(var_out.get())
        btn_ok.configure(state="normal" if (in_ok and out_ok) else "disabled")

    lbl_in = ttk.Label(root0, text="Input folder (images):")
    lbl_in.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 0))
    entry_in = ttk.Entry(root0, textvariable=var_in, state="readonly", width=50)
    entry_in.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=(10, 0))
    btn_browse_in = ttk.Button(root0, text="Browse…", command=browse_in)
    btn_browse_in.grid(row=0, column=2, sticky="e", padx=(0, 10), pady=(10, 0))

    lbl_out = ttk.Label(root0, text="Output folder (results):")
    lbl_out.grid(row=1, column=0, sticky="w", padx=10, pady=(8, 0))
    entry_out = ttk.Entry(root0, textvariable=var_out, state="readonly", width=50)
    entry_out.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=(8, 0))
    btn_browse_out = ttk.Button(root0, text="Browse…", command=browse_out)
    btn_browse_out.grid(row=1, column=2, sticky="e", padx=(0, 10), pady=(8, 0))

    chk_create_inside = ttk.Checkbutton(
        root0,
        text="Create 'output' inside the input folder",
        variable=var_create_inside,
        command=on_create_inside_toggle,
    )
    chk_create_inside.grid(row=2, column=0, columnspan=3, sticky="w", padx=10, pady=(10, 0))

    lbl_help = ttk.Label(
        root0,
        text="The output folder will contain (or update) results.csv and last_selection.json.",
        foreground="gray",
        wraplength=400,
        justify="left",
    )
    lbl_help.grid(row=3, column=0, columnspan=3, sticky="w", padx=10, pady=(2, 10))

    bar = ttk.Frame(root0)
    bar.grid(row=4, column=0, columnspan=3, sticky="ew", padx=10, pady=(0, 10))
    bar.columnconfigure(0, weight=1)

    result_folder = {"done": False}

    def on_cancel():
        result_folder["done"] = False
        root0.destroy()

    def on_ok():
        # Persist choices for next run
        try:
            in_dir_tmp = var_in.get().strip()
            out_dir_tmp = var_out.get().strip()
            _json.dump(
                {
                    "input_dir": in_dir_tmp,
                    "output_dir": out_dir_tmp,
                    "create_output_inside": bool(var_create_inside.get()),
                },
                SETTINGS_PATH.open("w", encoding="utf-8"),
                indent=2,
                ensure_ascii=False,
            )
        except Exception:
            pass

        result_folder["done"] = True
        root0.destroy()

    btn_cancel = ttk.Button(bar, text="Cancel", command=on_cancel)
    btn_cancel.grid(row=0, column=0, sticky="w")
    btn_ok = ttk.Button(bar, text="OK", command=on_ok, state="disabled")
    btn_ok.grid(row=0, column=1, sticky="e")

    root0.columnconfigure(1, weight=1)

    on_create_inside_toggle()
    update_ok_button_state()
    root0.mainloop()

    if not result_folder.get("done", False):
        return None

    in_dir = var_in.get()
    out_dir = var_out.get()

    in_path = Path(in_dir).expanduser().resolve()
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    csv_path = out_path / "results.csv"
    processed = load_processed(csv_path)

    img_paths = iter_images(in_path)
    if not img_paths:
        return (str(in_path), str(out_path), [])

    # --- Step 2: selection UI ---
    root = tk.Tk()
    root.title(title)

    frm = ttk.Frame(root, padding=10)
    frm.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    top = ttk.Frame(frm)
    top.grid(row=0, column=0, sticky="ew")
    top.columnconfigure(6, weight=1)

    ttk.Label(top, text=f"IN:  {in_path}").grid(row=0, column=0, columnspan=7, sticky="w")
    ttk.Label(top, text=f"OUT: {out_path}   |   CSV: {csv_path.name}").grid(
        row=1, column=0, columnspan=7, sticky="w", pady=(2, 0)
    )

    var_show_processed = tk.BooleanVar(value=True)
    ttk.Checkbutton(top, text="show processed", variable=var_show_processed).grid(
        row=2, column=0, sticky="w", pady=(8, 0)
    )

    btn_all = ttk.Button(top, text="Select all")
    btn_none = ttk.Button(top, text="Select none")
    btn_invert = ttk.Button(top, text="Invert")
    btn_all.grid(row=2, column=1, padx=(12, 0), pady=(8, 0))
    btn_none.grid(row=2, column=2, padx=(6, 0), pady=(8, 0))
    btn_invert.grid(row=2, column=3, padx=(6, 0), pady=(8, 0))

    lbl_count = ttk.Label(top, text="")
    lbl_count.grid(row=2, column=6, sticky="e", pady=(8, 0))

    canvas = tk.Canvas(frm, highlightthickness=0)
    canvas.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
    frm.rowconfigure(1, weight=1)

    vsb = ttk.Scrollbar(frm, orient="vertical", command=canvas.yview)
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

    thumbs: list[ImageTk.PhotoImage] = []
    vars_sel: list[tk.BooleanVar] = []
    meta: list[dict] = []

    def make_thumb(p: Path, size: int = 120) -> ImageTk.PhotoImage:
        im = Image.open(p).convert("RGB")
        im.thumbnail((size, size))
        return ImageTk.PhotoImage(im)

    def is_processed(p: Path) -> bool:
        return str(p.expanduser().resolve()) in processed

    def update_count() -> None:
        total = len(vars_sel)
        sel = sum(1 for v in vars_sel if v.get())
        lbl_count.configure(text=f"selected {sel}/{total}")

    def rebuild_list() -> None:
        for w in inner.winfo_children():
            w.destroy()
        thumbs.clear()
        vars_sel.clear()
        meta.clear()

        show_proc = bool(var_show_processed.get())

        row = 0
        for p in img_paths:
            proc = is_processed(p)
            if proc and not show_proc:
                continue

            v = tk.BooleanVar(value=(not proc))
            vars_sel.append(v)
            meta.append({"path": p, "processed": proc})

            try:
                ph = make_thumb(p)
            except Exception:
                ph = None

            if ph is not None:
                thumbs.append(ph)

            line = ttk.Frame(inner)
            line.grid(row=row, column=0, sticky="ew", pady=2)
            line.columnconfigure(3, weight=1)

            ttk.Checkbutton(line, variable=v).grid(row=0, column=0, padx=(0, 6), sticky="w")

            if ph is not None:
                lbl_img = ttk.Label(line, image=ph)
                lbl_img.image = ph
                lbl_img.grid(row=0, column=1, padx=(0, 8), sticky="w")
            else:
                ttk.Label(line, text="(no preview)").grid(row=0, column=1, padx=(0, 8), sticky="w")

            try:
                name = p.relative_to(in_path)
            except Exception:
                name = p.name

            ttk.Label(line, text=f"{name}").grid(row=0, column=2, sticky="w")
            ttk.Label(line, text=("processed" if proc else "new")).grid(row=0, column=3, sticky="e")

            row += 1

        update_count()
        root.update_idletasks()
        on_inner_config()

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

    bar2 = ttk.Frame(frm)
    bar2.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
    bar2.columnconfigure(0, weight=1)

    result = {"done": False, "selected": []}

    def on_cancel2() -> None:
        result["done"] = False
        root.after(10, root.destroy)

    def on_run() -> None:
        selected: list[str] = []
        for v, m in zip(vars_sel, meta):
            if v.get():
                selected.append(str(m["path"].expanduser().resolve()))
        result["done"] = True
        result["selected"] = selected
        root.after(10, root.destroy)

    ttk.Button(bar2, text="Cancel", command=on_cancel2).grid(row=0, column=0, sticky="w")
    ttk.Button(bar2, text="Run selected", command=on_run).grid(row=0, column=1, sticky="e")

    def periodic():
        update_count()
        root.after(250, periodic)

    rebuild_list()
    periodic()
    root.mainloop()

    if not result.get("done", False):
        return None

    selected_paths = list(result.get("selected", []))

    # Save last selection for reproducibility
    try:
        save_path = out_path / "last_selection.json"
        with save_path.open("w", encoding="utf-8") as f:
            _json.dump(
                {
                    "input_dir": str(in_path),
                    "output_dir": str(out_path),
                    "selected": selected_paths,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    except Exception as e:
        print("Warning: could not save last_selection.json:", e)

    return (str(in_path), str(out_path), selected_paths)
