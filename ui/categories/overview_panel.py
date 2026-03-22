"""«By value» tab: bucketed paths, rename, move between values."""

from __future__ import annotations

import sys
import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox, ttk

import customtkinter as ctk  # type: ignore[import-untyped]

from core.categories.mutations import (
    bucket_resolved_paths,
    clear_paths_column,
    rename_column_id,
    rename_column_title,
    rename_value_in_column,
    set_paths_to_value,
)
from ui.common.widgets import create_primary_button, create_secondary_button


def build_overview_tab(
    parent: ctk.CTkFrame,
    *,
    state: dict,
    top: ctk.CTk | ctk.CTkToplevel,
    base_font,
    small_font,
    all_rel_files: Callable[[], list[str]],
    refresh_tree: Callable[[], None],
    rebuild_column_menu: Callable[[], None],
    rebuild_value_chips: Callable[[], None],
) -> Callable[[], None]:
    """Build UI; returns ``full_refresh`` to sync column menus + lists."""

    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(2, weight=1)

    ctk.CTkLabel(
        parent,
        text="Paths grouped by resolved value (inheritance included). Multi-select in a list, then Move or Clear.",
        font=small_font,
        text_color="gray45",
        wraplength=720,
        anchor="w",
        justify="left",
    ).grid(row=0, column=0, sticky="ew", pady=(0, 8))

    bar = ctk.CTkFrame(parent, fg_color="transparent")
    bar.grid(row=1, column=0, sticky="ew", pady=(0, 6))

    ov_col_var = tk.StringVar(value="")
    ov_menu_holder: list[ctk.CTkOptionMenu | None] = [None]
    listboxes: dict[str, tk.Listbox] = {}

    scroll = ctk.CTkScrollableFrame(parent, fg_color=("gray96", "gray22"))
    scroll.grid(row=2, column=0, sticky="nsew")
    scroll.columnconfigure(0, weight=1)

    move_bar = ctk.CTkFrame(parent, fg_color="transparent")
    move_bar.grid(row=3, column=0, sticky="ew", pady=(10, 0))
    move_target_var = tk.StringVar(value="")
    move_menu_holder: list[ctk.CTkOptionMenu | None] = [None]

    def rebuild_ov_column_menu() -> None:
        if ov_menu_holder[0] is not None:
            try:
                ov_menu_holder[0].destroy()
            except Exception:
                pass
            ov_menu_holder[0] = None
        cols = state["store"].columns
        if not cols:
            ov_col_var.set("")
            om = ctk.CTkOptionMenu(bar, values=["(no columns)"], variable=ov_col_var, state="disabled")
            om.pack(side="left", padx=(0, 8))
            ov_menu_holder[0] = om
            return
        labels = [f"{c.title} ({c.id})" for c in cols]
        cur = next((f"{c.title} ({c.id})" for c in cols if c.id == state.get("active_column")), labels[0])
        ov_col_var.set(cur)
        om = ctk.CTkOptionMenu(bar, values=labels, variable=ov_col_var, command=_sync_ov_column)
        om.pack(side="left", padx=(0, 8))
        ov_menu_holder[0] = om

    def rebuild_move_menu() -> None:
        if move_menu_holder[0] is not None:
            try:
                move_menu_holder[0].destroy()
            except Exception:
                pass
            move_menu_holder[0] = None
        col_id = state.get("active_column") or ""
        col = next((c for c in state["store"].columns if c.id == col_id), None)
        if not col or not col.values:
            move_menu_holder[0] = ctk.CTkOptionMenu(
                move_bar, values=["(no values)"], variable=move_target_var, state="disabled"
            )
            move_menu_holder[0].pack(side="left", padx=(0, 8))
            return
        move_target_var.set(col.values[0])
        move_menu_holder[0] = ctk.CTkOptionMenu(move_bar, values=list(col.values), variable=move_target_var)
        move_menu_holder[0].pack(side="left", padx=(0, 8))

    def refresh_overview_inner() -> None:
        for w in scroll.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
        listboxes.clear()
        st = state["store"]
        col_id = state.get("active_column") or ""
        if not col_id:
            ctk.CTkLabel(scroll, text="Add and select a column.", font=small_font).pack(anchor="w", padx=8, pady=8)
            return
        rel_files = all_rel_files()
        buckets = bucket_resolved_paths(st, col_id, rel_files)
        order = sorted(buckets.keys(), key=lambda x: (x != "(unassigned)", str(x).lower()))
        for val in order:
            fr = ctk.CTkFrame(scroll, fg_color="transparent")
            fr.pack(fill="x", padx=4, pady=6)
            ctk.CTkLabel(fr, text=f"{val}  ({len(buckets[val])} paths)", font=base_font, anchor="w").pack(anchor="w")
            inner = tk.Frame(fr, bg="#eaeaea" if sys.platform == "darwin" else "#e8e8e8")
            inner.pack(fill="both", expand=True)
            h = min(10, max(3, len(buckets[val])))
            lb = tk.Listbox(inner, selectmode=tk.EXTENDED, height=h, font=("TkFixedFont", 10))
            sy = ttk.Scrollbar(inner, orient="vertical", command=lb.yview)
            lb.configure(yscrollcommand=sy.set)
            lb.grid(row=0, column=0, sticky="nsew")
            sy.grid(row=0, column=1, sticky="ns")
            inner.rowconfigure(0, weight=1)
            inner.columnconfigure(0, weight=1)
            for p in buckets[val]:
                lb.insert(tk.END, p)
            listboxes[val] = lb

    def _sync_ov_column(_choice: str | None = None) -> None:
        for c in state["store"].columns:
            if f"{c.title} ({c.id})" == ov_col_var.get():
                state["active_column"] = c.id
                break
        rebuild_move_menu()
        refresh_overview_inner()

    def _selected_paths() -> list[str]:
        out: list[str] = []
        for _val, lb in listboxes.items():
            for i in lb.curselection():
                out.append(str(lb.get(i)))
        return out

    def _apply_move() -> None:
        st = state["store"]
        col_id = state.get("active_column") or ""
        col = next((c for c in st.columns if c.id == col_id), None)
        tgt = move_target_var.get().strip()
        if not col or not tgt or tgt not in col.values:
            messagebox.showwarning("Move", "Pick a target value from the list.", parent=top)
            return
        paths = _selected_paths()
        if not paths:
            messagebox.showwarning("Move", "Select one or more paths in the lists above.", parent=top)
            return
        n = set_paths_to_value(st, paths, col_id, tgt, rel_files_set=set(all_rel_files()))
        messagebox.showinfo("Move", f"Updated {n} path(s).", parent=top)
        full_refresh()

    def _apply_clear_move() -> None:
        st = state["store"]
        col_id = state.get("active_column") or ""
        if not col_id:
            return
        paths = _selected_paths()
        if not paths:
            messagebox.showwarning("Clear", "Select paths first.", parent=top)
            return
        n = clear_paths_column(st, paths, col_id)
        messagebox.showinfo("Clear", f"Cleared explicit assignment on {n} path(s).", parent=top)
        full_refresh()

    def _dlg_rename_column() -> None:
        st = state["store"]
        col_id = state.get("active_column") or ""
        col = next((c for c in st.columns if c.id == col_id), None)
        if not col:
            messagebox.showwarning("Rename", "Select a column in the dropdown.", parent=top)
            return
        d = ctk.CTkToplevel(top)
        d.title("Rename column")
        d.geometry("420x200")
        d.transient(top)
        try:
            d.grab_set()
        except Exception:
            pass
        ctk.CTkLabel(d, text="Display title", font=base_font).pack(anchor="w", padx=14, pady=(14, 4))
        e_title = ctk.CTkEntry(d, width=360)
        e_title.insert(0, col.title)
        e_title.pack(padx=14, pady=4)
        ctk.CTkLabel(d, text="CSV column id (changes header in results.csv)", font=small_font).pack(
            anchor="w", padx=14, pady=(8, 4)
        )
        e_id = ctk.CTkEntry(d, width=360)
        e_id.insert(0, col.id)
        e_id.pack(padx=14, pady=4)

        def apply() -> None:
            nt = e_title.get().strip()
            nid = e_id.get().strip()
            if nt:
                rename_column_title(st, col_id, nt)
            if nid and nid != col_id:
                ok, msg = rename_column_id(st, col_id, nid)
                if not ok:
                    messagebox.showerror("Rename", msg or "Could not rename id.", parent=d)
                    return
                state["active_column"] = nid
            rebuild_column_menu()
            rebuild_value_chips()
            full_refresh()
            d.destroy()

        bf = ctk.CTkFrame(d, fg_color="transparent")
        bf.pack(pady=12)
        create_primary_button(bf, text="OK", command=apply, width=88).pack(side="left", padx=6)
        create_secondary_button(bf, text="Cancel", command=d.destroy, width=88).pack(side="left", padx=6)

    def _dlg_rename_value() -> None:
        st = state["store"]
        col_id = state.get("active_column") or ""
        col = next((c for c in st.columns if c.id == col_id), None)
        if not col or not col.values:
            messagebox.showwarning("Rename value", "Select a column with values.", parent=top)
            return
        d = ctk.CTkToplevel(top)
        d.title("Rename value")
        d.geometry("400x220")
        d.transient(top)
        try:
            d.grab_set()
        except Exception:
            pass
        ctk.CTkLabel(d, text="Old value", font=base_font).pack(anchor="w", padx=14, pady=(12, 4))
        old_var = tk.StringVar(value=col.values[0])
        ctk.CTkOptionMenu(d, values=list(col.values), variable=old_var, width=300).pack(padx=14, anchor="w")
        ctk.CTkLabel(d, text="New value", font=base_font).pack(anchor="w", padx=14, pady=(8, 4))
        e_new = ctk.CTkEntry(d, width=320)
        e_new.pack(padx=14, pady=4)

        def apply() -> None:
            ov = old_var.get().strip()
            nv = e_new.get().strip()
            ok, msg = rename_value_in_column(st, col_id, ov, nv)
            if not ok:
                messagebox.showerror("Rename value", msg, parent=d)
                return
            if state.get("active_value") == ov:
                state["active_value"] = nv
            rebuild_value_chips()
            full_refresh()
            refresh_tree()
            d.destroy()

        bf = ctk.CTkFrame(d, fg_color="transparent")
        bf.pack(pady=12)
        create_primary_button(bf, text="OK", command=apply, width=88).pack(side="left", padx=6)
        create_secondary_button(bf, text="Cancel", command=d.destroy, width=88).pack(side="left", padx=6)

    def full_refresh() -> None:
        rebuild_ov_column_menu()
        rebuild_move_menu()
        refresh_overview_inner()
        refresh_tree()

    rebuild_ov_column_menu()
    ctk.CTkLabel(move_bar, text="Move selection →", font=base_font).pack(side="left", padx=(0, 8))
    rebuild_move_menu()
    create_primary_button(move_bar, text="Move here", command=_apply_move, width=100).pack(side="left", padx=(0, 8))
    create_secondary_button(move_bar, text="Clear column for selection", command=_apply_clear_move, width=220).pack(
        side="left", padx=(0, 8)
    )

    create_secondary_button(bar, text="Refresh lists", command=full_refresh, width=110).pack(side="left", padx=(0, 8))
    create_secondary_button(bar, text="Rename column…", command=_dlg_rename_column, width=120).pack(
        side="left", padx=(0, 8)
    )
    create_secondary_button(bar, text="Rename value…", command=_dlg_rename_value, width=110).pack(side="left", padx=(0, 8))

    full_refresh()
    return full_refresh
