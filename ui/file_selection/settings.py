from __future__ import annotations
from pathlib import Path
import json as _json
from config import SETTINGS_PATH


def persist_folder_choices(
    *,
    input_dir: str,
    output_dir: str,
    create_output_inside: bool,
    process_subfolders: bool,
    settings_path: Path = SETTINGS_PATH,
) -> None:
    """Best-effort persistence of last-used folders and toggles."""
    try:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        prev: dict = {}
        if settings_path.exists():
            try:
                with settings_path.open("r", encoding="utf-8") as rf:
                    old_obj = _json.load(rf)
                if isinstance(old_obj, dict):
                    prev = old_obj
            except Exception:
                prev = {}
        prev.update(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "create_output_inside": bool(create_output_inside),
                "process_subfolders": bool(process_subfolders),
            }
        )
        _json.dump(
            prev,
            settings_path.open("w", encoding="utf-8"),
            indent=2,
            ensure_ascii=False,
        )
    except Exception:
        pass


def load_folder_choices(settings_path: Path = SETTINGS_PATH) -> dict:
    """Load last-used folders/toggles for UI prefilling."""
    try:
        if not settings_path.exists():
            return {}
        with settings_path.open("r", encoding="utf-8") as f:
            obj = _json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def persist_graph_units_settings(
    *,
    convert_enabled: bool,
    unit: str,
    pixels_per_unit: float,
    settings_path: Path = SETTINGS_PATH,
) -> None:
    """Persist graph unit conversion controls in the shared settings file."""
    try:
        obj = load_folder_choices(settings_path=settings_path)
        obj["graphs_convert_enabled"] = bool(convert_enabled)
        obj["graphs_unit"] = str(unit)
        try:
            obj["graphs_pixels_per_unit"] = float(pixels_per_unit)
        except Exception:
            obj["graphs_pixels_per_unit"] = 100.0
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        with settings_path.open("w", encoding="utf-8") as f:
            _json.dump(obj, f, indent=2, ensure_ascii=False)
    except Exception:
        pass