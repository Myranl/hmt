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
        _json.dump(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "create_output_inside": bool(create_output_inside),
                "process_subfolders": bool(process_subfolders),
            },
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