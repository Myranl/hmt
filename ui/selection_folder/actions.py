from __future__ import annotations

from pathlib import Path
from tkinter import filedialog, messagebox

from config import RESULTS_SCHEMA_VERSION
from core.validation import _dir_is_empty, _is_subpath


def browse_in(root, var_in, var_out, var_create_inside, validate_cb) -> None:
    d = filedialog.askdirectory(title="Select input folder (images)", parent=root)
    if d:
        var_in.set(d)
        if var_create_inside.get():
            p = Path(d).expanduser().resolve() / "output"
            var_out.set(str(p))
    validate_cb()


def browse_out(root, var_out, validate_cb) -> None:
    d = filedialog.askdirectory(title="Select output folder (results)", parent=root)
    if d:
        var_out.set(d)
    validate_cb()


def apply_create_inside_state(var_create_inside, entry_out, btn_browse_out, var_in, var_out, validate_cb) -> None:
    if var_create_inside.get():
        entry_out.configure(state="disabled")
        btn_browse_out.configure(state="disabled")
        if var_in.get().strip():
            p = Path(var_in.get()).expanduser().resolve() / "output"
            var_out.set(str(p))
    else:
        entry_out.configure(state="readonly")
        btn_browse_out.configure(state="normal")
    validate_cb()


def set_enabled(widgets, enabled: bool) -> None:
    st = "normal" if enabled else "disabled"
    for w in widgets:
        try:
            w.configure(state=st)
        except Exception:
            pass


def make_on_cancel_any(*, root, result):
    def _cb() -> None:
        result["done"] = False
        result["selected"] = []
        root.destroy()

    return _cb


def make_on_run(*, root, result, vars_sel, meta, var_in, var_out, json_module):
    def _cb() -> None:
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
                    json_module.dump(
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

    return _cb


def make_on_ok(
    *,
    root,
    result,
    var_in,
    var_out,
    var_create_inside,
    var_process_subfolders,
    status_var,
    btn_browse_in,
    btn_browse_out,
    chk_create_inside,
    btn_ok,
    chk_show_processed,
    btn_all,
    btn_none,
    btn_invert,
    btn_run,
    persist_folder_choices_fn,
    iter_images_fn,
    is_processed_fn,
    load_processed_fn,
    is_ours_output_fn,
    load_results_meta_fn,
    write_results_meta_fn,
    set_data_fn,
    rebuild_list_fn,
):
    """Factory returning the OK callback. All dependencies are injected to avoid globals."""

    def _cb() -> None:
        try:
            persist_folder_choices_fn(
                input_dir=var_in.get().strip(),
                output_dir=var_out.get().strip(),
                create_output_inside=bool(var_create_inside.get()),
                process_subfolders=bool(var_process_subfolders.get()),
            )
        except TypeError:
            # Backward compatibility if caller still injects a no-arg function
            persist_folder_choices_fn()
        in_dir = var_in.get().strip()
        out_dir = var_out.get().strip()
        in_path = Path(in_dir).expanduser().resolve()
        out_path = Path(out_dir).expanduser().resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        # Conflicts / warnings
        if _is_subpath(out_path, in_path) and not _dir_is_empty(out_path) and not is_ours_output_fn(out_path):
            go = messagebox.askyesno(
                "Output folder not empty",
                "The output folder is inside the input folder and is not empty.\n"
                "It may already contain unrelated files.\n\nContinue anyway?",
                parent=root,
            )
            if not go:
                set_enabled([btn_browse_in, btn_browse_out, chk_create_inside, btn_ok], True)
                return

        if out_path.exists() and not _dir_is_empty(out_path) and not is_ours_output_fn(out_path):
            go = messagebox.askyesno(
                "Output folder not empty",
                "The output folder is not empty and does not look like an existing HMT project.\n"
                "Existing files may be overwritten for selected images.\n\nContinue anyway?",
                parent=root,
            )
            if not go:
                set_enabled([btn_browse_in, btn_browse_out, chk_create_inside, btn_ok], True)
                return

        # Results schema/version check
        csv_default = out_path / "results.csv"
        meta_obj = load_results_meta_fn(out_path)
        if csv_default.exists():
            existing_ver = int(meta_obj.get("schema_version") or 0)
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
                    return
                if ans is False:
                    csv_default = out_path / f"results_v{RESULTS_SCHEMA_VERSION}.csv"

        csv_path = csv_default
        write_results_meta_fn(out_path, schema_version=RESULTS_SCHEMA_VERSION)
        processed = load_processed_fn(csv_path)

        # Disable folder selection while loading
        set_enabled([btn_browse_in, btn_browse_out, chk_create_inside, btn_ok], False)
        set_enabled([chk_show_processed, btn_all, btn_none, btn_invert, btn_run], False)

        status_var.set("Please wait, scanning folder…")
        root.update_idletasks()

        img_paths = iter_images_fn(in_path, subfolders=bool(var_process_subfolders.get()))
        if not img_paths:
            status_var.set("No images found in the selected folder.")
            set_enabled([btn_browse_in, btn_browse_out, chk_create_inside, btn_ok], True)
            return

        n_proc = sum(1 for p in img_paths if is_processed_fn(p, processed))
        n_new = len(img_paths) - n_proc
        status_var.set(
            f"Found {len(img_paths)} images ({n_new} new, {n_proc} processed). Loading thumbnails 0/{len(img_paths)}…"
        )

        # Re-enable selection controls; rows will appear incrementally
        set_enabled([chk_show_processed, btn_all, btn_none, btn_invert], True)
        if set_data_fn is not None:
            set_data_fn(img_paths=img_paths, processed=processed)
        rebuild_list_fn()

        # Enable run once list exists
        set_enabled([btn_run], True)

        # Make csv_path available to caller
        result["csv_path"] = str(csv_path)

    return _cb