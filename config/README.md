# config/sequences.csv

One row per video sequence.

| Column | Meaning |
|---|---|
| `seq_id` | Animal number used in the Observer XT export (column `vache`) |
| `animal` | Animal name; most scripts accept it for `--sequence` / `--animal` |
| `sequence` | Base name shared by every file of the sequence |
| `video_file` | Video file expected in `data/videos/` |
| `resolution` | Video resolution (width × height) |
| `LT_x … RT_y` | Stall corners in pixels: left-top, left-bottom, right-bottom, right-top. Used by `06_normalize_to_stall.py` (and the registration QC). These are the corners that reproduce `data/06_scaled` exactly. |
| `roi_x1 … roi_y2` | Region of interest of the studied animal (pixels), used by `04_assign_boxes_roi.py`. Recovered from the thesis trajectories (the original hand-drawn ROIs were not saved). |

To add a sequence: add a row, put the detector CSV in `data/raw/detections/`
(`<sequence>-WH-csv.csv`), the video in `data/videos/`, and pick the corners / ROI with
`06_normalize_to_stall.py --interactive` and `04_assign_boxes_roi.py --interactive`.
