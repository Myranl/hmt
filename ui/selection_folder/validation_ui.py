from __future__ import annotations
from pathlib import Path
import json as _json
from PIL import Image, ImageTk
import csv
import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

def validate_paths_ui(
    *,
    var_in,
    var_out,
    var_create_inside,
    status_var,
    btn_ok,
    can_write_dir,
    is_subpath,
) -> bool:
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

    if not can_write_dir(out_res):
        status_var.set("Cannot write to output folder (permission denied or read-only location).")
        btn_ok.configure(state="disabled")
        return False

    if is_subpath(out_res, in_res):
        status_var.set("Warning: output is inside input. Results will be stored alongside raw data.")
    else:
        status_var.set("Ready to scan. Press OK.")

    btn_ok.configure(state="normal")
    return True