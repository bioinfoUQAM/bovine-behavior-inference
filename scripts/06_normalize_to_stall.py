"""
06_normalize_to_stall.py
========================
Thesis sections 3.3.2 and 4.3.1 (normalisation relative to the stall).

The cameras differ in position, zoom and orientation (portrait 1080x1920 or
landscape 1920x1080), so pixel coordinates are not comparable between
sequences. The four corners of the stall (LT, LB, RB, RT) are clicked on the
first frame of the video and every box is expressed relative to the stall:

    cx, cy = centre of the LT-RB diagonal
    ws     = mean(|LT-RT|, |LB-RB|)       (stall width,  px)
    hs     = mean(|LT-LB|, |RT-RB|)       (stall height, px)

    Ln = (L - cx) / (ws/2)      Tn = (T - cy) / (hs/2)
    Wn = W / ws                 Hn = H / hs
    AR = W / H                  AR_stall = Wn / Hn
    logAR = log(AR)             logAR_stall = log(AR_stall)

The stall itself then spans roughly [-1, 1] x [-1, 1]. Its normalised corners
("floating" stall) are the input of the affine registration
(07_register_bboxes.py).

The corners used in the thesis are stored in config/sequences.csv, so the
script runs without any click by default; use --interactive to click new
corners (they are printed and saved to JSON).

Input
-----
    data/05_trajectories/<sequence>_interp_clipped_frames.csv
    config/sequences.csv

Output
------
    data/06_scaled/<sequence>_interp_clipped_frames_scaled.csv
        input columns + "<part> Ln/Tn/Wn/Hn/AR/AR_stall/logAR/logAR_stall"

Usage
-----
    python scripts/06_normalize_to_stall.py                       # all sequences, config corners
    python scripts/06_normalize_to_stall.py --sequence <sequence>
    python scripts/06_normalize_to_stall.py --sequence <sequence> --interactive
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = ["body box", "head box", "snout box"]
CORNERS = ["LT", "LB", "RB", "RT"]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def stall_center_and_size_from_4pts(stall_pts_px):
    """Centre (cx, cy), width ws and height hs of the stall from LT, LB, RB, RT."""
    pts = np.asarray(stall_pts_px, dtype=np.float64)
    if pts.shape != (4, 2):
        raise ValueError("Exactly 4 stall points required in order LT, LB, RB, RT.")
    LT, LB, RB, RT = pts
    cx = 0.5 * (LT[0] + RB[0])
    cy = 0.5 * (LT[1] + RB[1])
    ws = 0.5 * (np.linalg.norm(LT - RT) + np.linalg.norm(LB - RB))
    hs = 0.5 * (np.linalg.norm(LT - LB) + np.linalg.norm(RT - RB))
    return cx, cy, ws, hs


def floating_stall(stall_pts_px):
    """Normalised stall corners (input of the affine registration)."""
    cx, cy, ws, hs = stall_center_and_size_from_4pts(stall_pts_px)
    return (np.asarray(stall_pts_px, float) - [cx, cy]) / [ws / 2.0, hs / 2.0]


def normalize_tlwh(df, prefix, cx, cy, ws, hs):
    """Add the normalised columns of one anatomical part."""
    L, T, W, H = (df[f"{prefix} {c}"].astype(float).to_numpy() for c in "LTWH")
    df[f"{prefix} Ln"] = (L - cx) / (ws / 2.0)
    df[f"{prefix} Tn"] = (T - cy) / (hs / 2.0)
    df[f"{prefix} Wn"] = W / ws
    df[f"{prefix} Hn"] = H / hs
    AR = W / (H + 1e-9)
    AR_stall = (W / ws) / ((H / hs) + 1e-12)
    df[f"{prefix} AR"] = AR
    df[f"{prefix} AR_stall"] = AR_stall
    df[f"{prefix} logAR"] = np.log(AR)
    df[f"{prefix} logAR_stall"] = np.log(AR_stall + 1e-12)
    return df


# ---------------------------------------------------------------------------
# Interactive corner selection
# ---------------------------------------------------------------------------

def click_stall_corners(video_path, display_w=1200, display_h=800):
    """
    Show the first frame (resized to fit the screen) and collect clicks.
    Left click = add, right click = remove last, ESC = done.
    Returns the points in ORIGINAL frame pixels.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Cannot read the first frame of {video_path}")
    h, w = frame.shape[:2]
    scale = min(display_w / w, display_h / h)
    disp = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))))
    pts_disp, pts_orig = [], []
    win = "Click stall corners LT, LB, RB, RT  |  right click: undo  |  ESC: done"

    def redraw():
        img = disp.copy()
        for i, (x, y) in enumerate(pts_disp):
            cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
            cv2.circle(img, (x, y), 10, (0, 0, 0), 2)
            cv2.putText(img, CORNERS[i] if i < 4 else str(i + 1), (x + 12, y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow(win, img)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts_disp.append((x, y))
            pts_orig.append((int(x / scale), int(y / scale)))
            print(f"  point {len(pts_orig)}: display({x},{y}) -> original{pts_orig[-1]}")
        elif event == cv2.EVENT_RBUTTONDOWN and pts_disp:
            pts_disp.pop(), pts_orig.pop()
        redraw()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(win, on_mouse)
    redraw()
    while (cv2.waitKey(30) & 0xFF) != 27:
        pass
    cv2.destroyAllWindows()
    return pts_orig


def corners_from_config(row):
    return [(int(row[f"{c}_x"]), int(row[f"{c}_y"])) for c in CORNERS]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", help="Process one sequence only (default: all).")
    ap.add_argument("--sequences", default=ROOT / "config/sequences.csv")
    ap.add_argument("--input_dir", default=ROOT / "data/05_trajectories")
    ap.add_argument("--output_dir", default=ROOT / "data/06_scaled")
    ap.add_argument("--video_dir", default=ROOT / "data/videos")
    ap.add_argument("--interactive", action="store_true",
                    help="Click the stall corners instead of reading config/sequences.csv.")
    args = ap.parse_args()

    seqs = pd.read_csv(args.sequences)
    if args.sequence:
        seqs = seqs[seqs["sequence"] == args.sequence]
        if seqs.empty:
            raise SystemExit(f"Unknown sequence {args.sequence}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, s in seqs.iterrows():
        name = s["sequence"]
        if args.interactive:
            pts = click_stall_corners(Path(args.video_dir) / s["video_file"])
            if len(pts) != 4:
                raise SystemExit(f"Need exactly 4 stall points, got {len(pts)}.")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            (out_dir / f"{name}_stall_corners_{stamp}.json").write_text(
                json.dumps(dict(zip(CORNERS, pts)), indent=2))
            print("  copy these corners into config/sequences.csv to make them the default")
        else:
            pts = corners_from_config(s)

        cx, cy, ws, hs = stall_center_and_size_from_4pts(pts)
        df = pd.read_csv(Path(args.input_dir) / f"{name}_interp_clipped_frames.csv")
        for p in PREFIXES:
            df = normalize_tlwh(df, p, cx, cy, ws, hs)
        out = out_dir / f"{name}_interp_clipped_frames_scaled.csv"
        df.to_csv(out, index=False)

        fl = np.round(floating_stall(pts), 6).tolist()
        print(f"{s['animal']:<10s} corners(px)={pts}  centre=({cx:.1f},{cy:.1f})  "
              f"ws={ws:.1f}  hs={hs:.1f}\n           floating stall={fl}\n           -> {out}")


if __name__ == "__main__":
    main()
