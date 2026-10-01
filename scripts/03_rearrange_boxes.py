"""
03_rearrange_boxes.py
=====================
Thesis section 3.4 (spatial association of anatomical parts).

The detector outputs body, head and snout boxes without any identity: on a
given row, the three boxes do not necessarily belong to the same animal. This
script regroups them, frame by frame, with a greedy overlap-based strategy:

  1. Triplets  (body + head + snout): score = I(b,h) + I(b,s) + I(h,s)
  2. Duets     (body + head or body + snout): score = I(b,h) or I(b,s)
  3. Singletons (body alone)

where I(.,.) is the intersection area of two boxes. At each level the
candidate with the highest score is kept, all candidates sharing one of its
boxes are discarded, and so on until no candidate remains. Within each frame
the output rows are therefore ordered triplets > duets > singletons, each by
decreasing score: the first row of a frame is the most anatomically coherent
entity.

Input
-----
    data/raw/detections/<sequence>-WH-csv.csv
        columns: frame, cow_L, cow_T, cow_W, cow_H, cow_confidence,
                 head_L ... head_confidence, snout_L ... snout_confidence
        (missing values are written "none")

Output
------
    data/03_arranged/<sequence>_arranged_boxes.csv
        columns: frame, cow_L..cow_H, head_L..head_H, snout_L..snout_H
        (one row per reconstructed entity; missing parts are NaN)

Usage
-----
    python scripts/03_rearrange_boxes.py                       # all raw files
    python scripts/03_rearrange_boxes.py --input data/raw/detections/<file>.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

BODY = ("cow_L", "cow_T", "cow_W", "cow_H")
HEAD = ("head_L", "head_T", "head_W", "head_H")
SNOUT = ("snout_L", "snout_T", "snout_W", "snout_H")


def calculate_overlap(box1, box2):
    """Intersection area of two boxes given as (L, T, W, H)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _greedy(candidates):
    """
    Greedy selection: repeatedly keep the best-scoring candidate and drop every
    candidate that re-uses one of its boxes. A candidate is
    [body_id, head_id|None, snout_id|None, score]. Python's sort is stable, so
    ties keep their enumeration order.
    """
    candidates = sorted(candidates, key=lambda c: c[-1], reverse=True)
    kept, used_b, used_h, used_s = [], set(), set(), set()
    for c in candidates:
        b, h, s, _ = c
        if b in used_b or (h is not None and h in used_h) or (s is not None and s in used_s):
            continue
        kept.append(c)
        used_b.add(b)
        if h is not None:
            used_h.add(h)
        if s is not None:
            used_s.add(s)
    return kept, used_b, used_h, used_s


def associate_frame(body, head, snout):
    """
    Associate the boxes of one frame.

    body, head, snout : dict {row_id: (L, T, W, H)} of the non-missing boxes.
    Returns the list of entities (body_id, head_id|None, snout_id|None),
    ordered triplets > duets > singletons.
    """
    # 1. triplets
    cands = [[b, h, s, calculate_overlap(bb, hb) + calculate_overlap(bb, sb)
              + calculate_overlap(hb, sb)]
             for b, bb in body.items() for h, hb in head.items() for s, sb in snout.items()]
    triplets, ub, uh, us = _greedy(cands)
    body = {k: v for k, v in body.items() if k not in ub}
    head = {k: v for k, v in head.items() if k not in uh}
    snout = {k: v for k, v in snout.items() if k not in us}

    # 2. duets (body+head and body+snout compete together)
    cands = []
    for b, bb in body.items():
        cands += [[b, h, None, calculate_overlap(bb, hb)] for h, hb in head.items()]
        cands += [[b, None, s, calculate_overlap(bb, sb)] for s, sb in snout.items()]
    duets, ub, _, _ = _greedy(cands)

    # 3. singletons
    singles = [[b, None, None, 0.0] for b in body if b not in ub]

    return [tuple(c[:3]) for c in triplets + duets + singles]


def rearrange(df, frame_col="frame"):
    """Apply associate_frame() to every frame of a raw detection table."""
    df = df.reset_index(drop=True)
    B = df[list(BODY)].to_numpy(float)
    H = df[list(HEAD)].to_numpy(float)
    S = df[list(SNOUT)].to_numpy(float)
    nan4 = [np.nan] * 4
    rows = []

    for frame, idx in tqdm(df.groupby(frame_col, sort=False).indices.items(),
                           desc="Associating boxes"):
        body = {i: tuple(B[i]) for i in idx if not np.isnan(B[i]).any()}
        head = {i: tuple(H[i]) for i in idx if not np.isnan(H[i]).any()}
        snout = {i: tuple(S[i]) for i in idx if not np.isnan(S[i]).any()}
        for b, h, s in associate_frame(body, head, snout):
            rows.append([frame, *B[b],
                         *(H[h] if h is not None else nan4),
                         *(S[s] if s is not None else nan4)])

    return pd.DataFrame(rows, columns=[frame_col, *BODY, *HEAD, *SNOUT])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=ROOT / "data/raw/detections",
                    help="A raw detection CSV or a folder of them.")
    ap.add_argument("--output_dir", default=ROOT / "data/03_arranged")
    args = ap.parse_args()

    inp = Path(args.input)
    files = [inp] if inp.is_file() else sorted(inp.glob("*-WH-csv.csv"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        print(f"\n{f.name}")
        raw = pd.read_csv(f, na_values=["none"])
        arranged = rearrange(raw)
        seq = f.name.replace("-WH-csv.csv", "")
        out = out_dir / f"{seq}_arranged_boxes.csv"
        arranged.to_csv(out, index=False)
        print(f"  -> {out}  ({len(arranged)} entities, {arranged['frame'].nunique()} frames)")


if __name__ == "__main__":
    main()
