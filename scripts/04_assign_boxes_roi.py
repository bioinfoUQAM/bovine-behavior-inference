"""
04_assign_boxes_roi.py
======================
Thesis section 3.3.1 (selection of the region of interest).

Each video shows several stalls, but only one animal is studied. A rectangular
region of interest (ROI) delimiting the stall of that animal is drawn by hand
on the first frame of the video, and only the entities produced by
03_rearrange_boxes.py that fall inside it are kept:

  * the body-box centre must lie inside the ROI;
  * the head / snout centres (when present) must lie inside the ROI *or*
    inside the body box (tolerance for postures where the head leaves the
    stall, e.g. eating at the feeder).

Each entity is assigned to the first ROI that accepts it. The row order of
the arranged file is preserved, so the first row of every frame is still the
best-scoring entity.

Input
-----
    data/03_arranged/<sequence>_arranged_boxes.csv
    data/videos/<sequence>.mp4               (only for interactive ROI drawing)

Output
------
    data/04_roi/<sequence>_roi_<id>.csv      (one file per ROI)
    data/04_roi/<sequence>_rois.json         (ROI coordinates, for reproducibility)

The original hand-drawn ROIs were not saved. The ROIs stored in
config/sequences.csv (roi_x1 .. roi_y2) were recovered from the shipped
trajectories: with them, 05_build_trajectories.py reproduces > 99 % of the
frames of data/05_trajectories with identical boxes (see README).

Usage
-----
    # default: every sequence, ROI from config/sequences.csv
    python scripts/04_assign_boxes_roi.py

    # one sequence (animal name or full sequence name)
    python scripts/04_assign_boxes_roi.py --sequence Versace

    # interactive: click-and-drag one rectangle per ROI, ESC to confirm
    python scripts/04_assign_boxes_roi.py --sequence Versace --interactive

    # explicit ROI coordinates (pixels, x1 y1 x2 y2) or a saved JSON
    python scripts/04_assign_boxes_roi.py --sequence Versace --roi 220 331 1000 1236
    python scripts/04_assign_boxes_roi.py --sequence Versace --roi_json data/04_roi/<sequence>_rois.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

BODY = ("cow_L", "cow_T", "cow_W", "cow_H")
HEAD = ("head_L", "head_T", "head_W", "head_H")
SNOUT = ("snout_L", "snout_T", "snout_W", "snout_H")


# ---------------------------------------------------------------------------
# Interactive ROI drawing
# ---------------------------------------------------------------------------

def select_rois_from_first_frame(video_path):
    """Show the first frame and let the user draw rectangles (ESC to finish)."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Cannot read the first frame of {video_path}")

    rois, start = [], []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            start[:] = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP and start:
            (x1, y1), (x2, y2) = start[0], (x, y)
            rois.append({"id": len(rois) + 1,
                         "x1": min(x1, x2), "y1": min(y1, y2),
                         "x2": max(x1, x2), "y2": max(y1, y2)})
            start.clear()

    win = "Select ROI (drag = rectangle, ESC = done)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        img = frame.copy()
        for r in rois:
            cv2.rectangle(img, (r["x1"], r["y1"]), (r["x2"], r["y2"]), (0, 255, 0), 2)
            cv2.putText(img, f"ID {r['id']}", (r["x1"], r["y1"] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(win, img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    print(f"{len(rois)} ROI(s) defined: {rois}")
    return rois


# ---------------------------------------------------------------------------
# Spatial filtering
# ---------------------------------------------------------------------------

def _inside(px, py, x1, y1, x2, y2):
    return (x1 <= px) & (px <= x2) & (y1 <= py) & (py <= y2)


def filter_by_roi(df, roi):
    """Boolean mask of the entities accepted by one ROI (vectorised)."""
    bL, bT, bW, bH = (df[c].to_numpy(float) for c in BODY)
    r = (roi["x1"], roi["y1"], roi["x2"], roi["y2"])
    ok = ~np.isnan(bL + bT + bW + bH) & _inside(bL + bW / 2, bT + bH / 2, *r)

    for part in (HEAD, SNOUT):
        L, T, W, H = (df[c].to_numpy(float) for c in part)
        cx, cy = L + W / 2, T + H / 2
        present = ~np.isnan(L)
        part_ok = _inside(cx, cy, *r) | _inside(cx, cy, bL, bT, bL + bW, bT + bH)
        ok &= ~present | part_ok
    return ok


def assign(df, rois):
    """Return {roi_id: DataFrame}; an entity goes to the first ROI accepting it."""
    taken = np.zeros(len(df), dtype=bool)
    out = {}
    for roi in rois:
        m = filter_by_roi(df, roi) & ~taken
        taken |= m
        out[roi["id"]] = df.loc[m]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", help="Animal or sequence name (default: all sequences).")
    ap.add_argument("--config", default=ROOT / "config/sequences.csv")
    ap.add_argument("--arranged_dir", default=ROOT / "data/03_arranged")
    ap.add_argument("--video_dir", default=ROOT / "data/videos")
    ap.add_argument("--output_dir", default=ROOT / "data/04_roi")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--interactive", action="store_true", help="Draw the ROI(s) on the first frame.")
    g.add_argument("--roi", type=int, nargs=4, action="append", metavar=("X1", "Y1", "X2", "Y2"),
                   help="ROI in pixels (repeat the option for several ROIs).")
    g.add_argument("--roi_json", help="JSON file written by a previous run.")
    args = ap.parse_args()

    seqs = pd.read_csv(args.config)
    if args.sequence:
        seqs = seqs[(seqs["animal"] == args.sequence) | (seqs["sequence"] == args.sequence)]
        if seqs.empty:
            raise SystemExit(f"{args.sequence} not found in {args.config}")
    elif args.interactive or args.roi or args.roi_json:
        raise SystemExit("--interactive / --roi / --roi_json require --sequence.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for _, row in seqs.iterrows():
        name = row["sequence"]
        if args.roi:
            rois = [{"id": i + 1, "x1": a, "y1": b, "x2": c, "y2": d}
                    for i, (a, b, c, d) in enumerate(args.roi)]
        elif args.roi_json:
            rois = json.loads(Path(args.roi_json).read_text())
        elif args.interactive:
            rois = select_rois_from_first_frame(Path(args.video_dir) / row["video_file"])
        else:
            rois = [{"id": 1, **{k: int(row[f"roi_{k}"]) for k in ("x1", "y1", "x2", "y2")}}]
        if not rois:
            raise SystemExit("No ROI defined.")
        (out_dir / f"{name}_rois.json").write_text(json.dumps(rois, indent=2))

        df = pd.read_csv(Path(args.arranged_dir) / f"{name}_arranged_boxes.csv")
        for roi_id, sub in assign(df, rois).items():
            out = out_dir / f"{name}_roi_{roi_id}.csv"
            sub.to_csv(out, index=False)
            print(f"{row['animal']:10s} ROI {roi_id} {rois[roi_id - 1] if roi_id <= len(rois) else ''}: "
                  f"{len(sub)} entities on {sub['frame'].nunique()} frames -> {out.name}")


if __name__ == "__main__":
    main()
