from pathlib import Path
import json as _json
from config import SETTINGS_PATH, RESULTS_SCHEMA_VERSION, RESULTS_META_NAME

def _is_subpath(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False

def _dir_is_empty(p: Path) -> bool:
    try:
        return not any(p.iterdir())
    except Exception:
        return False

def _is_ours_output(out_dir: Path) -> bool:
    # Heuristic: if it has our meta file OR last_selection.json OR results.csv
    if (out_dir / RESULTS_META_NAME).exists():
        return True
    if (out_dir / "last_selection.json").exists():
        return True
    if (out_dir / "results.csv").exists():
        return True
    return False

def _can_write_dir(p: Path) -> bool:
    try:
        p.mkdir(parents=True, exist_ok=True)
        test = p / ".__hmt_write_test__"
        test.write_text("ok", encoding="utf-8")
        test.unlink(missing_ok=True)
        return True
    except Exception:
        return False

def _load_results_meta(out_dir: Path) -> dict:
    mp = out_dir / RESULTS_META_NAME
    if not mp.exists():
        return {}
    try:
        return _json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _write_results_meta(out_dir: Path, *, schema_version: int) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        mp = out_dir / RESULTS_META_NAME
        _json.dump(
            {"tool": "HMT", "schema_version": int(schema_version)},
            mp.open("w", encoding="utf-8"),
            indent=2,
            ensure_ascii=False,
        )
    except Exception:
        pass