"""
01_observer_events_to_binary.py
===============================
Thesis section 3.2.2 (annotation pre-processing).

Converts the interval-based behavioural annotations exported from
The Observer XT (one row per "State start" / "State stop" / "State point"
event) into one frame-level, multi-label binary table per sequence.

For every event k of animal i:

    t_start = Time_Relative_sf
    t_end   = t_start + Duration_sf
    f_start = round(t_start * FPS) + 1        (frames are 1-based)
    f_end   = round(t_end   * FPS)
    y[f, k] = 1   for every frame f in [f_start, f_end]

Frames covered by no event receive 0. Several behaviours can be active on the
same frame (multi-label). "State point" events activate a single frame.

Input
-----
    data/raw/annotations/UQAM_Validation_Fangio_Annotation_APC - Events_fixed.xlsx
    config/sequences.csv  (maps the Observer column `vache` to a sequence name)

Output
------
    data/01_annotations_binary/<sequence>_behaviors.csv
        columns: Frame, <17 behaviours in alphabetical order>

Usage
-----
    python scripts/01_observer_events_to_binary.py
    python scripts/01_observer_events_to_binary.py --fps 30 --session Original
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# The 17 categories present in the Observer XT ethogram (thesis table 3.3).
ALL_BEHAVIOURS = [
    "Defecation", "Drinking", "Eating", "Exploration", "Grooming",
    "Head Shaking (as if to detach)", "Kneeling", "Lying", "Lying Down",
    "Not Visible", "Other (including inactive)", "Scratching",
    "Social Interaction", "Standing", "Standing Up", "Urination",
    "Vigilance toward the door",
]


def count_video_frames(video_path):
    """Return the number of frames of a video, or None if it cannot be read."""
    try:
        import cv2
    except ImportError:
        return None
    if not Path(video_path).exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n if n > 0 else None


def count_detection_frames(detection_csv):
    """Fallback: number of frames = last frame index in the raw detection CSV."""
    if not Path(detection_csv).exists():
        return None
    return int(pd.read_csv(detection_csv, usecols=["frame"])["frame"].max())


def events_to_binary(events, n_frames, fps, behaviours):
    """
    Project Observer XT events on the frame grid.

    Parameters
    ----------
    events : pd.DataFrame
        Rows of one animal, with columns Time_Relative_sf, Duration_sf,
        Behavior and Event_Type.
    n_frames : int
        Number of frames of the sequence (frames are numbered 1..n_frames).
    fps : float
        Video frame rate.
    behaviours : list of str
        Output columns.

    Returns
    -------
    pd.DataFrame with a Frame column and one 0/1 column per behaviour.
    """
    Y = np.zeros((n_frames, len(behaviours)), dtype=np.int8)
    col = {b: j for j, b in enumerate(behaviours)}

    for _, ev in events.iterrows():
        if ev["Event_Type"] == "State stop":
            continue                    # the duration is carried by "State start"
        beh = ev["Behavior"]
        if beh not in col:
            print(f"  [WARN] unknown behaviour '{beh}' ignored")
            continue
        t_start = float(ev["Time_Relative_sf"])
        f_start = int(round(t_start * fps)) + 1
        if ev["Event_Type"] == "State point":
            f_end = f_start
        else:
            t_end = t_start + float(ev["Duration_sf"])
            f_end = int(round(t_end * fps))
        f_start, f_end = max(f_start, 1), min(f_end, n_frames)
        if f_end >= f_start:
            Y[f_start - 1:f_end, col[beh]] = 1

    out = pd.DataFrame(Y, columns=behaviours)
    out.insert(0, "Frame", np.arange(1, n_frames + 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", default=ROOT / "data/raw/annotations/"
                    "UQAM_Validation_Fangio_Annotation_APC - Events_fixed.xlsx")
    ap.add_argument("--sequences", default=ROOT / "config/sequences.csv")
    ap.add_argument("--video_dir", default=ROOT / "data/videos",
                    help="Used only to read the number of frames.")
    ap.add_argument("--detection_dir", default=ROOT / "data/raw/detections",
                    help="Fallback to get the number of frames when videos are absent.")
    ap.add_argument("--output_dir", default=ROOT / "data/01_annotations_binary")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--session", default="Original",
                    help="Value of the Observer column `jour` to keep "
                         "(the thesis uses 'Original').")
    args = ap.parse_args()

    events = pd.read_excel(args.events)
    seqs = pd.read_csv(args.sequences)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, s in seqs.iterrows():
        ev = events[(events["vache"] == s["seq_id"]) & (events["jour"] == args.session)]
        n_frames = (count_video_frames(Path(args.video_dir) / s["video_file"])
                    or count_detection_frames(Path(args.detection_dir) / f"{s['sequence']}-WH-csv.csv"))
        if n_frames is None:
            raise SystemExit(f"Cannot determine the number of frames for {s['sequence']}")

        table = events_to_binary(ev, n_frames, args.fps, ALL_BEHAVIOURS)
        out = out_dir / f"{s['sequence']}_behaviors.csv"
        table.to_csv(out, index=False)
        print(f"{s['animal']:<10s} {len(ev):4d} events -> {n_frames} frames -> {out}")


if __name__ == "__main__":
    main()
