import customtkinter as ctk  # type: ignore[import-untyped]


def setup_theme() -> None:
    """Configure global CustomTkinter appearance for the app."""
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("green")


def get_base_font() -> ctk.CTkFont:
    """Return base font for controls.

    Call only after a Tk root / CTk window is created.
    """
    return ctk.CTkFont(size=13)


def get_small_muted_font() -> ctk.CTkFont:
    """Return smaller muted font for helper text."""
    return ctk.CTkFont(size=11)

