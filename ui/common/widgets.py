from __future__ import annotations

import customtkinter as ctk  # type: ignore[import-untyped]

from ui.common.theme import get_base_font, get_small_muted_font


def create_card_frame(parent, **kwargs) -> ctk.CTkFrame:
    """Standard card-like frame used for grouped controls."""
    options: dict = {
        "fg_color": ("white", "gray20"),
        "corner_radius": 10,
    }
    options.update(kwargs)
    return ctk.CTkFrame(parent, **options)


def create_main_panel(parent, **kwargs) -> ctk.CTkFrame:
    """Main content panel frame."""
    options: dict = {
        "fg_color": ("white", "gray18"),
        "corner_radius": 10,
    }
    options.update(kwargs)
    return ctk.CTkFrame(parent, **options)


def create_primary_button(parent, text: str, **kwargs) -> ctk.CTkButton:
    """Primary action button (e.g. OK, Run)."""
    options: dict = {
        "width": 96,
        "height": 30,
        "corner_radius": 6,
        "font": kwargs.get("font", get_base_font()),
    }
    options.update(kwargs)
    return ctk.CTkButton(parent, text=text, **options)


def create_secondary_button(parent, text: str, **kwargs) -> ctk.CTkButton:
    """Secondary action button (e.g. Cancel)."""
    options: dict = {
        "width": 96,
        "height": 30,
        "corner_radius": 6,
        "font": kwargs.get("font", get_base_font()),
    }
    options.update(kwargs)
    return ctk.CTkButton(parent, text=text, **options)


def create_toolbar_button(parent, text: str, **kwargs) -> ctk.CTkButton:
    """Compact toolbar-style button (e.g. Select all / none)."""
    options: dict = {
        "width": 96,
        "height": 28,
        "corner_radius": 6,
        "font": kwargs.get("font", get_base_font()),
    }
    options.update(kwargs)
    return ctk.CTkButton(parent, text=text, **options)


def create_status_label(parent, *, textvariable=None, text: str = "", **kwargs) -> ctk.CTkLabel:
    """Muted status / helper text label."""
    options: dict = {
        "text_color": "gray40",
        "font": kwargs.get("font", get_small_muted_font()),
        "anchor": "w",
        "justify": "left",
    }
    options.update(kwargs)
    return ctk.CTkLabel(parent, textvariable=textvariable, text=text, **options)

