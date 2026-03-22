# HMT Project Structure

HMT/
├── main.py                 # Entry point: file selection → run pipeline for the list
├── config.py               # Configuration paths, results schema version
├── requirements.txt
│
├── core/                   # Shared logic, validation
│   ├── validation.py       # Path validation, writing results.meta
│   └── categories/         # User-defined CSV columns (assignments JSON + resolve)
│       ├── schema.py       # CategoryColumn / CategoryStore
│       ├── paths.py        # Relative POSIX keys under input root
│       ├── store_io.py     # Load/save category_assignments.json
│       ├── resolve.py      # Merge labels into pipeline rows (+ file overrides in leaf mode)
│       └── mutations.py    # Rename column/value, move paths between values
│
├── pipeline/               # Processing orchestration
│   ├── batch.py            # Loop over images, write results.csv
│   └── single_image.py     # Single slice: brain → midline → hippocampus → metrics
│
├── preproc/                # Image preprocessing
│   ├── resize.py           # ds/orig scales, ROI
│   ├── retina.py           # Downsample, CLAHE, local mean subtraction
│   └── quantize.py         # 3-threshold map (sketch), midline barrier, small→gray
│
├── segmentation/           # Mask post-processing
│   └── postprocess.py      # smooth_fill_mask (close, open, holes, blur)
│
├── viz/                    # Visualization (formerly analysis)
│   └── overlay.py          # Overlay hippocampus masks on the image
│
└── ui/                     # Interfaces for pipeline stages
    ├── common/             # Shared UI utilities
    │   ├── tk_after.py     # Cancel after callbacks on destroy
    │   └── tk_utils.py     # Images, grid, panels
    │
    ├── file_selection/     # Folder and file list selection (formerly selection_folder)
    │   ├── main_selection.py
    │   ├── actions.py
    │   ├── settings.py
    │   ├── validation_ui.py
    │   ├── thumbs.py
    │   └── list_view.py
    │
    ├── brain/              # Everything related to brain contour (formerly brain_mask)
    │   ├── threshold_ui.py      # Brain threshold mask
    │   ├── brain_outline_UI.py  # Contour refinement, manual edits
    │   ├── hemisphere.py        # Midline
    │   ├── contour_editor_ui.py # Incomplete contour editor (Break/Missing)
    │   ├── mask_compute.py      # Threshold-based mask algorithm
    │   ├── mask_morphology.py   # Morphology, components, holes
    │   └── mask_utils.py        # Mask and display utilities
    │
    ├── roi/                # ROI and three-threshold map (sketch)
    │   ├── roi_picker_ui.py     # ROI rectangle selection
    │   ├── bins_sketch_ui.py    # Thresholds t1/t2, small_to_gray
    │   └── run_ui_and_get_params.py  # Combined step: ROI + sketch
    │
    ├── pick_components.py  # Hippocampus component selection (clicks, Cut/Add)
    ├── review_ui.py        # Review and re-edit selection
    │
    └── categories/         # Group / experiment metadata (separate CSV columns)
        ├── category_editor_ui.py  # Tabs: tree assign + by-value overview
        ├── overview_panel.py        # Buckets per value, rename, move
        └── input_scan.py            # Image scan for the editor tree