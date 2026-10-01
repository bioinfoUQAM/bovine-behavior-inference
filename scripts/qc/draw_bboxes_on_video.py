"""
qc/draw_bboxes_on_video.py
==========================
Visual check of the detections / trajectories: draws the body (cow), head
and snout boxes on the video and writes an annotated copy.

Accepted box tables (the column layout is detected automatically):
  * raw detections / arranged boxes  : frame, cow_L .. cow_H, head_L .., snout_L ..
    (data/raw/detections, data/03_arranged, data/04_roi) - several entities
    per frame are all drawn;
  * trajectories and later stages    : Frame, "body box L" .., "head box L" ..,
    "snout box L" .. (data/05_trajectories, data/06_scaled) - pixel columns.
Frame numbers are 1-based (frame 1 = first image of the video).

Usage
-----
    python scripts/qc/draw_bboxes_on_video.py \\
        --csv   data/05_trajectories/<sequence>_interp_clipped_frames.csv \\
        --video data/videos/<sequence>.mp4 \\
        --output results/qc_videos/<sequence>_annotated.mp4 --start 1 --end 5000
"""
import argparse
from pathlib import Path

import cv2
import pandas as pd

STYLES = {  # BGR
    "body":  dict(color=(255, 0, 0), thickness=3, label="Cow"),
    "head":  dict(color=(0, 200, 255), thickness=2, label="Head"),
    "snout": dict(color=(0, 255, 0), thickness=2, label="Snout"),
}
FONT, FONT_SCALE, FONT_THICK, PAD = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2, 4


def column_map(columns):
    """part -> (L, T, W, H) column names, and the frame column."""
    cols = set(columns)
    if "cow_L" in cols:
        parts = {"body": "cow", "head": "head", "snout": "snout"}
        cmap = {p: [f"{q}_{c}" for c in "LTWH"] for p, q in parts.items()}
    elif "body box L" in cols:
        cmap = {p: [f"{p} box {c}" for c in "LTWH"] for p in STYLES}
    else:
        raise KeyError("Unknown box column layout (expected cow_L... or 'body box L'...).")
    frame_col = "frame" if "frame" in cols else "Frame"
    return cmap, frame_col


def draw_bbox(img, L, T, W, H, color, thickness, label):
    if any(pd.isna(v) for v in (L, T, W, H)):
        return
    x1, y1, x2, y2 = int(L), int(T), int(L + W), int(T + H)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)
    ly1, ly2 = y1 - th - 2 * PAD, y1
    if ly1 < 0:                                   # label below the box at the top edge
        ly1, ly2 = y2, y2 + th + 2 * PAD
    cv2.rectangle(img, (x1, ly1), (x1 + tw + 2 * PAD, ly2), color, cv2.FILLED)
    cv2.putText(img, label, (x1 + PAD, ly2 - PAD), FONT, FONT_SCALE, (0, 0, 0), FONT_THICK, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=5000)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip()
    cmap, fcol = column_map(df.columns)
    for c in sum(cmap.values(), []):
        df[c] = pd.to_numeric(df[c], errors="coerce")       # "none" -> NaN
    df = df[(df[fcol] >= args.start) & (df[fcol] <= args.end)]
    by_frame = {int(f): g for f, g in df.groupby(fcol)}

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video : {w}x{h} @ {fps:.1f} fps | {total} frames")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start - 1)
    frame = args.start - 1
    while frame < args.end:
        ok, img = cap.read()
        if not ok:
            break
        frame += 1                                          # 1-based, as in the CSVs
        for _, row in by_frame.get(frame, pd.DataFrame()).iterrows():
            for part, st in STYLES.items():
                draw_bbox(img, *(row[c] for c in cmap[part]), st["color"], st["thickness"], st["label"])
        out.write(img)
        if frame % 500 == 0:
            print(f"  frame {frame} / {min(args.end, total)}")
    cap.release()
    out.release()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
