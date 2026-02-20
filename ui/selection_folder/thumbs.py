from __future__ import annotations
from pathlib import Path
import json as _json
from PIL import Image, ImageTk
import csv
import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

def make_thumb(p: Path, size: int = 120) -> ImageTk.PhotoImage | None:
    try:
        im = Image.open(p).convert("RGB")
        im.thumbnail((size, size))
        return ImageTk.PhotoImage(im)
    except Exception:
        return None