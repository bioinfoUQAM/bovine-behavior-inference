"""
05_build_trajectories.py
========================
Thesis sections 3.5 (temporal continuity) and 3.2.2 / 4.2 (fusion with the
behavioural annotations).

Builds the per-frame trajectory of the studied animal and attaches its
multi-label annotation vector:

  1. Temporal association - for every frame, the first entity of the ROI file
     (the best-scoring one, see 03_rearrange_boxes.py) is the studied animal.
  2. Raw pixel derivatives - "ddt <part> box <X>" = backward difference
     between consecutive video frames of this track, computed before step 3
     (the first frame re-uses the second value). They are kept for
     inspection only; the model recomputes its own derivatives on the
     registered features (08_gmmhmm_loocv.py).
  3. Completeness - frames where this entity is not a full triplet
     (body + head + snout) are dropped ("clipped"), because the GMM-HMM
     emission density needs a complete observation vector (thesis 3.6).
  4. Annotation fusion - the 17 behaviour columns of
     01_observer_events_to_binary.py are joined on the frame number.

The column names follow the convention used by all downstream scripts:
"body box L", "head box T", "snout box W", ...

Input
-----
    data/04_roi/<sequence>_roi_<id>.csv
    data/01_annotations_binary/<sequence>_behaviors.csv

Output
------
    data/05_trajectories/<sequence>_interp_clipped_frames.csv
        Frame, Times, body/head/snout box L T W H, ddt ..., 17 behaviours

The trajectories shipped in data/05_trajectories are the ones used in the
thesis. Re-running steps 03 -> 05 with the recovered ROIs gives > 99 % of
identical frames but not a bit-exact copy (the original hand-drawn ROIs were
not saved), so existing files are only replaced with --overwrite.

Usage
-----
    python scripts/05_build_trajectories.py                       # all sequences
    python scripts/05_build_trajectories.py --sequence Versace [--roi_id 1]
    python scripts/05_build_trajectories.py --output_dir data/05_trajectories_rebuilt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PARTS = {"body box": "cow", "head box": "head", "snout box": "snout"}
COORDS = ["L", "T", "W", "H"]


def build_trajectory(roi_df, labels, fps=30.0, frame_col="frame"):
    """Return the per-frame trajectory table (see module docstring)."""
    # 1. best entity of every frame (row order of the ROI file is preserved)
    first = roi_df.groupby(frame_col, sort=True).head(1).sort_values(frame_col)

    out = pd.DataFrame({"Frame": first[frame_col].astype(int).to_numpy()})
    for new, old in PARTS.items():
        for c in COORDS:
            out[f"{new} {c}"] = first[f"{old}_{c}"].to_numpy(float)

    # 2. raw backward differences, computed on the full track BEFORE clipping
    #    (so that, after a dropped frame, the derivative still refers to the
    #    previous video frame); the first value is back-filled.
    box_cols = [f"{p} {c}" for p in PARTS for c in COORDS]
    for c in box_cols:
        out[f"ddt {c}"] = out[c].diff().bfill(limit=1)

    # 3. keep complete triplets only ("clipped" frames)
    ddt_cols = [f"ddt {c}" for c in box_cols]
    out = out.dropna(subset=box_cols).reset_index(drop=True)
    out[box_cols] = out[box_cols].round().astype(int)
    # a part missing on the previous frame gives an undefined derivative -> 0
    out[ddt_cols] = out[ddt_cols].fillna(0).round().astype(int)
    out.insert(1, "Times", (out["Frame"] - 1) / fps)

    # 4. annotations
    out = out.merge(labels, on="Frame", how="left")
    beh = [c for c in labels.columns if c != "Frame"]
    out[beh] = out[beh].fillna(0).astype(int)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", help="Animal or sequence name (default: all sequences).")
    ap.add_argument("--config", default=ROOT / "config/sequences.csv")
    ap.add_argument("--roi_id", type=int, default=1, help="ROI of the studied animal.")
    ap.add_argument("--roi_dir", default=ROOT / "data/04_roi")
    ap.add_argument("--labels_dir", default=ROOT / "data/01_annotations_binary")
    ap.add_argument("--output_dir", default=ROOT / "data/05_trajectories")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--overwrite", action="store_true", help="Replace existing trajectory files.")
    args = ap.parse_args()

    seqs = pd.read_csv(args.config)
    if args.sequence:
        seqs = seqs[(seqs["animal"] == args.sequence) | (seqs["sequence"] == args.sequence)]
        if seqs.empty:
            raise SystemExit(f"{args.sequence} not found in {args.config}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in seqs["sequence"]:
        out = out_dir / f"{name}_interp_clipped_frames.csv"
        if out.exists() and not args.overwrite:
            print(f"{name}: {out} exists, skipped (use --overwrite or another --output_dir)")
            continue
        roi = pd.read_csv(Path(args.roi_dir) / f"{name}_roi_{args.roi_id}.csv")
        labels = pd.read_csv(Path(args.labels_dir) / f"{name}_behaviors.csv")
        traj = build_trajectory(roi, labels, fps=args.fps)
        traj.to_csv(out, index=False)
        print(f"{name}: {len(traj)} complete frames kept / {roi['frame'].nunique()} "
              f"frames with an entity -> {out}")


if __name__ == "__main__":
    main()
