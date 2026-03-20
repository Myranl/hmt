"""Consistent screen size for fitting CTk windows (Windows HiDPI + Tk can report odd values)."""

from __future__ import annotations

import sys
import tkinter as tk


def layout_screen_wh(root: tk.Misc) -> tuple[int, int]:
    """Return (width, height) for layout heuristics (margins, max canvas, etc.)."""
    try:
        w = int(root.winfo_screenwidth())
        h = int(root.winfo_screenheight())
    except Exception:
        return 1400, 900
    if w < 400 or h < 300:
        return 1400, 900
    if sys.platform == "win32":
        # Per-monitor scaling: Tk/CTk sometimes reports very large or inconsistent sizes;
        # clamp so max_win_w / max_body_h stay in a sensible range.
        w = max(1024, min(w, 2560))
        h = max(768, min(h, 1600))
    return w, h
