from pipeline.batch import process_folder
from pathlib import Path
from pipeline.batch import process_paths
from ui.folier_select import run_folder_and_selection_ui


if __name__ == "__main__":
#    process_folder(input_dir, out_dir, interactive=True)

    res = run_folder_and_selection_ui()
    if res is None:
        raise SystemExit(0)

    _input_dir, out_dir, selected = res

    if not selected:
        raise SystemExit(0)

    paths = [Path(p) for p in selected]
    process_paths(paths, out_dir, interactive=True, save_csv=True)
