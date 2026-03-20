from __future__ import annotations

from pathlib import Path
import csv
import json


def _try_parse_json_string(value: object) -> object:
    """Best-effort parse for JSON-like strings (including nested JSON strings)."""
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s:
        return value
    if not ((s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))):
        return value
    try:
        parsed = json.loads(s)
    except Exception:
        return value

    # Handle cases like: "midline_pts": "[[1,2],[3,4]]"
    if isinstance(parsed, dict):
        out: dict[object, object] = {}
        for k, v in parsed.items():
            out[k] = _try_parse_json_string(v)
        return out
    if isinstance(parsed, list):
        return [_try_parse_json_string(v) for v in parsed]
    return parsed


def _format_value_for_debug(value: str) -> str:
    parsed = _try_parse_json_string(value)
    if isinstance(parsed, (dict, list)):
        try:
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            return str(parsed)
    return value


CANONICAL_HEADERS = [
    "image_path",
    "overlay_path",
    "img_name",
    "accepted",
    "contour_version",
    "brain_area_px",
    "brain_perim_px",
    "midline_area_left_px",
    "midline_area_right_px",
    "midline_perimeter_left_px",
    "midline_perimeter_right_px",
    "non_complete_contour",
    "hipp_area_left_px",
    "hipp_area_right_px",
    "hipp_perimeter_left_px",
    "hipp_perimeter_right_px",
]
def debug_print_results_head(out_dir: str, rows: int = 5) -> bool:
    """Print first N data rows from results.csv for debug."""
    out_path = Path(out_dir).expanduser().resolve()
    csv_path = out_path / "results.csv"

    print("\n[Reorganise result] Debug preview")
    print(f"Output dir: {out_path}")
    print(f"CSV path:   {csv_path}")

    if not csv_path.exists():
        print("results.csv not found.")
        return False

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rdr = csv.reader(f)
            headers = next(rdr, [])
            if not headers:
                print("No headers found.")
                return False
            print("Columns:", ", ".join(headers))

            shown = 0
            recovered = 0
            max_rows = max(1, int(rows))
            for raw_fields in rdr:
                if not raw_fields:
                    continue

                # Some legacy rows are stored as one giant quoted CSV line in column 0.
                # Recover by parsing that string again as a CSV record.
                fields = list(raw_fields)
                if len(fields) == 1 and "," in fields[0]:
                    try:
                        fields = next(csv.reader([fields[0]]))
                        recovered += 1
                    except Exception:
                        pass

                shown += 1
                print(f"\nRow {shown}:")
                for i, col in enumerate(headers):
                    val = fields[i] if i < len(fields) else ""
                    pretty = _format_value_for_debug(val)
                    if "\n" in pretty:
                        print(f"  {col}:")
                        for line in pretty.splitlines():
                            print(f"    {line}")
                    else:
                        print(f"  {col}: {pretty}")

                if shown >= max_rows:
                    break

            if shown == 0:
                print("results.csv has no data rows.")
            else:
                if recovered > 0:
                    print(f"\nRecovered {recovered} legacy row(s) that were packed into one field.")
                print(f"Printed {shown} row(s).")
            return True
    except Exception as exc:
        print(f"Failed to read results.csv: {exc}")
        return False


def reorganise_results_to_ok_csv(out_dir: str) -> tuple[bool, str, int]:
    """Reorganise results.csv and save as result_ok.csv in same folder.

    Returns (ok, output_path, rows_written).
    """
    out_path = Path(out_dir).expanduser().resolve()
    src = out_path / "results.csv"
    dst = out_path / "result_ok.csv"

    if not src.exists():
        print(f"[Reorganise result] Source file not found: {src}")
        return False, str(dst), 0

    try:
        with src.open("r", encoding="utf-8", newline="") as f:
            rd = csv.DictReader(f)
            if not rd.fieldnames:
                print("[Reorganise result] Empty headers, cannot reorganise.")
                return False, str(dst), 0

            rows_by_key: dict[str, dict[str, str]] = {}
            rows_no_key: list[dict[str, str]] = []

            for rec in rd:
                accepted = str(rec.get("accepted", "")).strip().lower()
                if accepted != "ok":
                    continue

                overlay_path = str(rec.get("overlay_path", "")).strip()
                img_name = str(rec.get("img_name", "")).strip()
                non_complete = str(rec.get("non_complete_contour", "")).strip().lower() in ("1", "true", "yes")
                contour_version = str(rec.get("contour_version", "")).strip().lower()
                if contour_version == "":
                    contour_version = "corrected" if non_complete else "single"

                canon = {h: str(rec.get(h, "") or "") for h in CANONICAL_HEADERS}
                canon["accepted"] = "ok"
                canon["contour_version"] = contour_version

                if non_complete:
                    # Allow two final versions per overlay_path: raw + corrected.
                    if contour_version not in ("raw", "corrected"):
                        contour_version = "corrected"
                        canon["contour_version"] = contour_version
                    key_base = overlay_path or img_name
                    key = f"{key_base}|{contour_version}" if key_base else ""
                else:
                    # Keep only one latest version per overlay_path (or img_name fallback).
                    key = overlay_path or img_name

                if key:
                    rows_by_key[key] = canon
                else:
                    rows_no_key.append(canon)

        ordered_rows = rows_no_key + list(rows_by_key.values())
        with dst.open("w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=CANONICAL_HEADERS)
            wr.writeheader()
            for rec in ordered_rows:
                wr.writerow({h: rec.get(h, "") for h in CANONICAL_HEADERS})

        print(f"[Reorganise result] Saved: {dst} (rows={len(ordered_rows)})")
        return True, str(dst), len(ordered_rows)
    except Exception as exc:
        print(f"[Reorganise result] Failed to save {dst}: {exc}")
        return False, str(dst), 0
