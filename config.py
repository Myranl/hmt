from pathlib import Path

SETTINGS_PATH = Path.home() / ".hipoca_histo_last_selection.json"

RESULTS_SCHEMA_VERSION = 1
RESULTS_META_NAME = "results.meta.json"

# User-defined category columns + assignments (merged into results.csv rows).
CATEGORIES_STORE_NAME = "category_assignments.json"