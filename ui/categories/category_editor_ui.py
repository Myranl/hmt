"""Category / group editor: independent CSV columns, file vs leaf-folder granularity."""

from __future__ import annotations

import sys
from pathlib import Path

import customtkinter as ctk  # type: ignore[import-untyped]
import tkinter as tk
from tkinter import messagebox, ttk

from core.categories.paths import posix_rel
from core.categories.resolve import resolve_labels_for_image
from core.categories.schema import CategoryColumn, CategoryStore
from core.categories.store_io import default_store_path, load_store, make_unique_column_id, save_store
from ui.categories.input_scan import iter_images_under
from ui.categories.overview_panel import build_overview_tab
from ui.common.theme import setup_theme, get_base_font, get_small_muted_font
from ui.common.widgets import create_card_frame, create_primary_button, create_secondary_button


def _parent_rel(dir_rel: str) -> str:
    if not dir_rel or "/" not in dir_rel:
        return ""
    return str(Path(dir_rel).parent.as_posix())


def _collect_dirs_from_files(rel_files: list[str]) -> list[str]:
    dirs: set[str] = set()
    for rf in rel_files:
        p = Path(rf)
        if len(p.parts) <= 1:
            continue
        for i in range(len(p.parts) - 1):
            dirs.add(str(Path(*p.parts[: i + 1]).as_posix()))
    return sorted(dirs, key=lambda x: (x.count("/"), x.lower()))


def _filter_files_unassigned(
    store: CategoryStore,
    rel_files: list[str],
    column_id: str,
) -> list[str]:
    if not column_id:
        return list(rel_files)
    out: list[str] = []
    root = Path(store.input_root)
    for rf in rel_files:
        abs_p = root / rf
        lab = resolve_labels_for_image(store, abs_p)
        if not str(lab.get(column_id, "")).strip():
            out.append(rf)
    return out


def run_category_editor_ui(parent: ctk.CTk | None, *, input_dir: str, output_dir: str) -> None:
    in_root = Path(input_dir).expanduser().resolve()
    out_root = Path(output_dir).expanduser().resolve()
    if not in_root.is_dir():
        messagebox.showerror("Categories", "Input folder is missing or not a directory.")
        return
    out_root.mkdir(parents=True, exist_ok=True)
    store_path = default_store_path(out_root)

    setup_theme()
    top: ctk.CTk | ctk.CTkToplevel
    if parent is not None and isinstance(parent, ctk.CTk):
        try:
            if bool(parent.winfo_exists()):
                top = ctk.CTkToplevel(parent)
                top.transient(parent)
            else:
                top = ctk.CTk()
        except Exception:
            top = ctk.CTk()
    else:
        top = ctk.CTk()

    top.title("Category assignments")
    top.configure(fg_color="white")
    top.minsize(960, 620)
    top.geometry("1100x680")

    loaded = load_store(store_path)
    if loaded is None:
        store = CategoryStore(
            input_root=str(in_root),
            granularity="file",
            columns=[],
            assignments={},
        )
    else:
        store = loaded
        store.input_root = str(in_root)

    state: dict = {
        "store": store,
        "recursive": True,
        "mode_auto": True,
        "active_column": "",
        "active_value": "",
        "only_unassigned": False,
        "overview_refresh": None,  # set after build_overview_tab
    }

    base_font = get_base_font()
    small = get_small_muted_font()

    outer = create_card_frame(top)
    outer.pack(fill="both", expand=True, padx=14, pady=14)
    outer.columnconfigure(0, weight=1)
    outer.rowconfigure(1, weight=1)

    hdr = ctk.CTkFrame(outer, fg_color="transparent")
    hdr.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    hdr.columnconfigure(1, weight=1)
    ctk.CTkLabel(hdr, text="Categories → CSV columns", font=ctk.CTkFont(size=17, weight="bold")).grid(
        row=0, column=0, sticky="w"
    )
    ctk.CTkLabel(
        hdr,
        text=f"Input: {in_root}\nOutput: {out_root}",
        font=small,
        text_color="gray40",
        justify="left",
    ).grid(row=0, column=1, sticky="e")

    tabview = ctk.CTkTabview(outer)
    tabview.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
    tab_tree = tabview.add("Tree & assign")
    tab_over = tabview.add("By value (overview)")
    tab_tree.columnconfigure(0, weight=2)
    tab_tree.columnconfigure(1, weight=3)
    tab_tree.rowconfigure(0, weight=1)

    body = tab_tree

    # --- Tree ---
    left = ctk.CTkFrame(body, fg_color=("gray96", "gray20"))
    left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
    left.rowconfigure(0, weight=1)
    left.columnconfigure(0, weight=1)

    tree_fr = tk.Frame(left, bg="#f3f3f3" if sys.platform == "darwin" else "#ececec")
    tree_fr.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
    tree_fr.rowconfigure(0, weight=1)
    tree_fr.columnconfigure(0, weight=1)

    tv = ttk.Treeview(tree_fr, selectmode="extended", show="tree")
    tv.grid(row=0, column=0, sticky="nsew")
    sy = ttk.Scrollbar(tree_fr, orient="vertical", command=tv.yview)
    sy.grid(row=0, column=1, sticky="ns")
    tv.configure(yscrollcommand=sy.set)
    # Files (active column): green = value resolved, gray = still missing for this column
    tv.tag_configure("assigned", foreground="#047857")
    tv.tag_configure("todo", foreground="#9ca3af")
    # Folders: explicit rows in category_assignments.json for this folder path
    tv.tag_configure("folder_explicit", foreground="#047857")
    tv.tag_configure("folder_plain", foreground="#9ca3af")

    # --- Right panel ---
    right = ctk.CTkFrame(body, fg_color="transparent")
    right.grid(row=0, column=1, sticky="nsew")
    right.columnconfigure(0, weight=1)

    r0 = ctk.CTkFrame(right, fg_color="transparent")
    r0.grid(row=0, column=0, sticky="ew", pady=(0, 6))
    ctk.CTkLabel(r0, text="Low level (assignment unit)", font=base_font).pack(anchor="w")
    _gran_labels = ("Image files", "Leaf folders (direct images)")
    _gran_to_val = {"Image files": "file", "Leaf folders (direct images)": "leaf_folder"}
    _val_to_gran = {v: k for k, v in _gran_to_val.items()}
    var_gran = tk.StringVar(value=_val_to_gran.get(store.granularity, "Image files"))

    def on_gran_change() -> None:
        g = _gran_to_val.get(var_gran.get(), "file")
        state["store"].granularity = g  # type: ignore[assignment]
        refresh_tree()

    seg = ctk.CTkSegmentedButton(
        r0,
        values=list(_gran_labels),
        variable=var_gran,
        command=lambda _v: on_gran_change(),
        font=base_font,
    )
    seg.pack(anchor="w", pady=(4, 0))
    ctk.CTkLabel(
        right,
        text="file = each image · leaf_folder = folder that directly contains images (deepest folder wins per column)",
        font=small,
        text_color="gray45",
        wraplength=420,
        justify="left",
    ).grid(row=1, column=0, sticky="w")

    r_mode = ctk.CTkFrame(right, fg_color="transparent")
    r_mode.grid(row=2, column=0, sticky="ew", pady=(12, 6))
    ctk.CTkLabel(r_mode, text="Mode", font=base_font).pack(anchor="w")
    _mode_labels = ("Auto (double-click folder)", "Manual (select + Assign)")
    _mode_to_auto = {"Auto (double-click folder)": True, "Manual (select + Assign)": False}
    var_mode = tk.StringVar(value=_mode_labels[0])

    def on_mode_change() -> None:
        state["mode_auto"] = bool(_mode_to_auto.get(var_mode.get(), True))

    mode_seg = ctk.CTkSegmentedButton(
        r_mode,
        values=list(_mode_labels),
        variable=var_mode,
        command=lambda _v: on_mode_change(),
        font=base_font,
    )
    mode_seg.pack(anchor="w", pady=(4, 0))
    ctk.CTkLabel(
        right,
        text="Auto: click a folder in the tree to assign the active value to everything under it (files) or to that folder (leaf mode).\n"
        "Manual: select items, then Assign.",
        font=small,
        text_color="gray45",
        wraplength=420,
        justify="left",
    ).grid(row=3, column=0, sticky="w", pady=(4, 0))

    r_col = ctk.CTkFrame(right, fg_color="transparent")
    r_col.grid(row=4, column=0, sticky="ew", pady=(12, 4))
    ctk.CTkLabel(r_col, text="Active column (separate CSV column each)", font=base_font).pack(anchor="w")
    col_menu_var = tk.StringVar(value="")

    col_frame = ctk.CTkFrame(r_col, fg_color="transparent")
    col_frame.pack(fill="x", pady=(4, 0))
    col_menu_widget: list[ctk.CTkOptionMenu | None] = [None]

    val_frame = ctk.CTkScrollableFrame(right, height=160, fg_color=("gray96", "gray22"))
    val_frame.grid(row=5, column=0, sticky="ew", pady=(8, 4))

    var_only_un = tk.BooleanVar(value=False)

    def _sync_un_var() -> None:
        state["only_unassigned"] = bool(var_only_un.get())
        refresh_tree()

    chk_un = ctk.CTkCheckBox(
        right,
        text="Tree: show only paths with no value in the active column",
        variable=var_only_un,
        font=base_font,
        command=_sync_un_var,
    )
    chk_un.grid(row=6, column=0, sticky="w", pady=(8, 4))

    def current_column() -> CategoryColumn | None:
        cid = state.get("active_column") or ""
        for c in state["store"].columns:
            if c.id == cid:
                return c
        return None

    def rebuild_value_chips() -> None:
        for w in val_frame.winfo_children():
            w.destroy()
        col = current_column()
        if not col:
            ctk.CTkLabel(val_frame, text="Add a column first.", font=small).pack(anchor="w", padx=4, pady=4)
            return
        if not col.values:
            ctk.CTkLabel(val_frame, text="Add allowed values for this column.", font=small).pack(anchor="w", padx=4, pady=4)
            return

        def pick(v: str) -> None:
            state["active_value"] = v

        for v in col.values:
            is_on = state.get("active_value") == v

            def _mk_cmd(x: str) -> None:
                pick(x)
                rebuild_value_chips()

            b = ctk.CTkButton(
                val_frame,
                text=v,
                width=min(200, 12 + 8 * len(v)),
                height=28,
                font=base_font,
                fg_color=("#2563eb", "#1d4ed8") if is_on else ("#e5e7eb", "#374151"),
                text_color=("white", "white") if is_on else ("gray20", "gray90"),
                command=lambda x=v: _mk_cmd(x),
            )
            b.pack(anchor="w", padx=4, pady=3)

    def rebuild_column_menu() -> None:
        if col_menu_widget[0] is not None:
            try:
                col_menu_widget[0].destroy()
            except Exception:
                pass
            col_menu_widget[0] = None
        cols = state["store"].columns
        if not cols:
            state["active_column"] = ""
            col_menu_var.set("")
            om = ctk.CTkOptionMenu(col_frame, values=["(no columns)"], variable=col_menu_var, state="disabled")
            om.pack(side="left", fill="x", expand=True)
            col_menu_widget[0] = om
            rebuild_value_chips()
            return

        def on_pick(choice: str) -> None:
            for c in cols:
                if f"{c.title} ({c.id})" == choice:
                    state["active_column"] = c.id
                    state["active_value"] = c.values[0] if c.values else ""
                    break
            rebuild_value_chips()
            refresh_tree()
            orf = state.get("overview_refresh")
            if callable(orf):
                orf()

        labels = [f"{c.title} ({c.id})" for c in cols]
        if not state["active_column"] and cols:
            state["active_column"] = cols[0].id
            state["active_value"] = cols[0].values[0] if cols[0].values else ""
        cur_lab = next((f"{c.title} ({c.id})" for c in cols if c.id == state["active_column"]), labels[0])
        col_menu_var.set(cur_lab)
        om = ctk.CTkOptionMenu(col_frame, values=labels, variable=col_menu_var, command=on_pick)
        om.pack(side="left", fill="x", expand=True)
        col_menu_widget[0] = om
        rebuild_value_chips()

    def all_rel_files() -> list[str]:
        files = iter_images_under(in_root, recursive=bool(state["recursive"]))
        out: list[str] = []
        for p in files:
            r = posix_rel(p, in_root)
            if r:
                out.append(r)
        return out

    def _folder_has_explicit_assignment(st: CategoryStore, folder_rel: str) -> bool:
        row = st.assignments.get(folder_rel) or {}
        return any(str(row.get(c.id, "")).strip() for c in st.columns)

    def _format_item_label(st: CategoryStore, rel_key: str, *, is_folder: bool) -> str:
        """Show folder/file name + [val]… from explicit JSON row, or · if no row for this key."""
        base = Path(rel_key).name if rel_key else ""
        row = st.assignments.get(rel_key) or {}
        parts: list[str] = []
        for c in st.columns:
            v = str(row.get(c.id, "")).strip()
            if v:
                parts.append(f"[{v}]")
        if parts:
            return f"{base}  {' '.join(parts)}"
        if is_folder:
            return f"{base}  ·"
        return base

    def refresh_tree() -> None:
        tv.delete(*tv.get_children())
        rel_files = all_rel_files()
        col_id = state.get("active_column") or ""
        if state.get("only_unassigned") and col_id:
            rel_files = _filter_files_unassigned(state["store"], rel_files, col_id)
        dirs = _collect_dirs_from_files(rel_files)
        st = state["store"]
        root = Path(st.input_root)

        def tag_for_file(rf: str) -> str:
            if not col_id:
                return "todo"
            abs_p = root / rf
            lab = resolve_labels_for_image(st, abs_p).get(col_id, "")
            return "assigned" if str(lab).strip() else "todo"

        for d in dirs:
            par = _parent_rel(d)
            ftag = "folder_explicit" if _folder_has_explicit_assignment(st, d) else "folder_plain"
            label = _format_item_label(st, d, is_folder=True)
            tv.insert(par if tv.exists(par) else "", "end", iid=d, text=label, open=True, tags=(ftag,))

        for rf in sorted(rel_files, key=str.lower):
            par = _parent_rel(rf) or ""
            if state["store"].granularity == "leaf_folder":
                if not tv.exists(par):
                    continue
                tag = tag_for_file(rf)
                if tv.exists(rf):
                    continue
                flabel = _format_item_label(st, rf, is_folder=False)
                tv.insert(par if tv.exists(par) else "", "end", iid=rf, text=flabel, tags=(tag,))
            else:
                p_par = par
                if not p_par:
                    p_par = ""
                if p_par and not tv.exists(p_par):
                    continue
                flabel = _format_item_label(st, rf, is_folder=False)
                tv.insert(p_par if tv.exists(p_par) else "", "end", iid=rf, text=flabel, tags=(tag_for_file(rf),))

    def folder_has_files_in_tree(folder_rel: str, rel_files: list[str]) -> bool:
        prefix = folder_rel + "/"
        for rf in rel_files:
            if rf == folder_rel or rf.startswith(prefix):
                return True
        return False

    def apply_to_folder_auto(folder_rel: str) -> None:
        col = current_column()
        if not col or not state.get("active_value"):
            messagebox.showwarning("Categories", "Pick a column and a value first.", parent=top)
            return
        val = str(state["active_value"]).strip()
        if val not in col.values:
            messagebox.showwarning("Categories", "Value must be one of the column's allowed values.", parent=top)
            return
        st = state["store"]
        root = Path(st.input_root)
        rel_files = all_rel_files()
        prefix = folder_rel + "/" if folder_rel else ""

        if st.granularity == "file":
            for rf in rel_files:
                if folder_rel == "" or rf == folder_rel or rf.startswith(prefix):
                    if rf in st.assignments:
                        st.assignments[rf][col.id] = val
                    else:
                        st.assignments[rf] = {col.id: val}
        else:
            key = folder_rel
            if key not in st.assignments:
                st.assignments[key] = {}
            st.assignments[key][col.id] = val
        refresh_tree()
        orf = state.get("overview_refresh")
        if callable(orf):
            orf()

    def apply_manual_selection() -> None:
        col = current_column()
        if not col or not state.get("active_value"):
            messagebox.showwarning("Categories", "Pick a column and a value first.", parent=top)
            return
        val = str(state["active_value"]).strip()
        if val not in col.values:
            messagebox.showwarning("Categories", "Value must be one of the column's allowed values.", parent=top)
            return
        st = state["store"]
        sel = tv.selection()
        if not sel:
            messagebox.showwarning("Categories", "Select one or more rows in the tree.", parent=top)
            return
        for iid in sel:
            rel = str(iid)
            if st.granularity == "file":
                if rel not in all_rel_files():
                    continue
                if rel not in st.assignments:
                    st.assignments[rel] = {}
                st.assignments[rel][col.id] = val
            else:
                # File path = per-file override; folder path = folder rule (see resolve.py)
                key = rel
                if key not in st.assignments:
                    st.assignments[key] = {}
                st.assignments[key][col.id] = val
        refresh_tree()
        orf = state.get("overview_refresh")
        if callable(orf):
            orf()

    def clear_selection_column() -> None:
        col = current_column()
        if not col:
            return
        st = state["store"]
        sel = tv.selection()
        targets = list(sel) if sel else []
        if not targets:
            messagebox.showwarning("Categories", "Select rows to clear, or use the tree selection.", parent=top)
            return
        rel_files = all_rel_files()
        for iid in targets:
            rel = str(iid)
            if st.granularity == "file":
                if rel in st.assignments and col.id in st.assignments[rel]:
                    del st.assignments[rel][col.id]
                    if not st.assignments[rel]:
                        del st.assignments[rel]
            else:
                key = rel
                if key in st.assignments and col.id in st.assignments[key]:
                    del st.assignments[key][col.id]
                    if not st.assignments[key]:
                        del st.assignments[key]
        refresh_tree()
        orf = state.get("overview_refresh")
        if callable(orf):
            orf()

    def on_tree_click(_ev) -> None:
        if not state.get("mode_auto"):
            return
        if not tv.selection():
            return
        iid = tv.selection()[0]
        rel = str(iid)
        if state["store"].granularity == "file":
            if rel in all_rel_files():
                p = str(Path(rel).parent.as_posix())
                apply_to_folder_auto(p)
            else:
                apply_to_folder_auto(rel)
        else:
            if rel in all_rel_files():
                rel = str(Path(rel).parent.as_posix())
            apply_to_folder_auto(rel)

    tv.bind("<<TreeviewSelect>>", lambda _e: None)
    tv.bind("<ButtonRelease-1>", on_tree_click)

    # fix: auto mode on single click fires for every click - use double-click for auto to avoid accidental assigns
    tv.unbind("<ButtonRelease-1>")
    tv.bind("<Double-1>", on_tree_click)

    # Columns management row
    man = ctk.CTkFrame(right, fg_color="transparent")
    man.grid(row=7, column=0, sticky="ew", pady=(12, 4))

    ent_col_title = ctk.CTkEntry(man, placeholder_text="New column title (e.g. Sex)", width=200)
    ent_col_title.pack(side="left", padx=(0, 6))

    def add_column() -> None:
        title = ent_col_title.get().strip()
        if not title:
            return
        ids = {c.id for c in state["store"].columns}
        cid = make_unique_column_id(title, set(ids))
        state["store"].columns.append(CategoryColumn(id=cid, title=title, values=[]))
        state["active_column"] = cid
        state["active_value"] = ""
        ent_col_title.delete(0, "end")
        rebuild_column_menu()
        refresh_tree()
        orf = state.get("overview_refresh")
        if callable(orf):
            orf()

    create_secondary_button(man, text="Add column", command=add_column, width=100).pack(side="left", padx=(0, 6))

    ent_val = ctk.CTkEntry(man, placeholder_text="New value for active column", width=200)
    ent_val.pack(side="left", padx=(0, 6))

    def add_value() -> None:
        col = current_column()
        if not col:
            messagebox.showwarning("Categories", "Select / create a column first.", parent=top)
            return
        v = ent_val.get().strip()
        if not v:
            return
        if v not in col.values:
            col.values.append(v)
        state["active_value"] = v
        ent_val.delete(0, "end")
        rebuild_value_chips()
        orf = state.get("overview_refresh")
        if callable(orf):
            orf()

    create_secondary_button(man, text="Add value", command=add_value, width=90).pack(side="left")

    btns = ctk.CTkFrame(right, fg_color="transparent")
    btns.grid(row=8, column=0, sticky="ew", pady=(14, 4))
    create_primary_button(btns, text="Assign to selection (manual)", command=apply_manual_selection, width=200).pack(
        side="left", padx=(0, 8)
    )
    create_secondary_button(btns, text="Clear active column for selection", command=clear_selection_column, width=220).pack(
        side="left", padx=(0, 8)
    )

    foot = ctk.CTkFrame(outer, fg_color="transparent")
    foot.grid(row=2, column=0, sticky="ew", pady=(10, 0))
    ctk.CTkLabel(
        foot,
        text="Tree: green + […] = saved rule on this folder; gray + · = no rule here yet. "
        "Files: green = active column filled; gray = missing. Save → JSON; Close / OK → refresh results.csv.",
        font=small,
        text_color="gray45",
        wraplength=900,
    ).pack(side="left", fill="x", expand=True)

    def on_save() -> None:
        state["store"].input_root = str(in_root)
        try:
            save_store(store_path, state["store"])
            messagebox.showinfo("Categories", f"Saved:\n{store_path}", parent=top)
        except OSError as e:
            messagebox.showerror("Categories", str(e), parent=top)

    create_primary_button(foot, text="Save", command=on_save, width=88).pack(side="right", padx=(8, 0))

    def on_close() -> None:
        try:
            from pipeline.batch import refresh_results_csv_categories

            ok, msg = refresh_results_csv_categories(str(out_root))
            if not ok:
                print(f"[Categories] results.csv refresh: {msg}")
        except Exception as exc:
            print(f"[Categories] results.csv refresh skipped: {exc}")
        top.destroy()

    create_secondary_button(foot, text="Close", command=on_close, width=88).pack(side="right")

    var_sub = tk.BooleanVar(value=True)

    def on_sub() -> None:
        state["recursive"] = bool(var_sub.get())
        refresh_tree()

    chk_sub = ctk.CTkCheckBox(
        left,
        text="Include subfolders when scanning",
        variable=var_sub,
        font=base_font,
        command=on_sub,
    )
    chk_sub.grid(row=1, column=0, sticky="w", padx=8, pady=(0, 8))

    state["overview_refresh"] = build_overview_tab(
        tab_over,
        state=state,
        top=top,
        base_font=base_font,
        small_font=small,
        all_rel_files=all_rel_files,
        refresh_tree=refresh_tree,
        rebuild_column_menu=rebuild_column_menu,
        rebuild_value_chips=rebuild_value_chips,
    )

    rebuild_column_menu()
    refresh_tree()

    top.update_idletasks()
    top.lift()
    try:
        top.focus_force()
    except Exception:
        pass
