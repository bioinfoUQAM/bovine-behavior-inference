"""
07_register_bboxes.py
=====================
Thesis sections 3.6.1, 4.3.2 and 4.3.3 (affine registration of the boxes).

The stall normalisation (06) corrects scale and position but not the residual
perspective / rotation differences between cameras. For each sequence:

  1. the floating stall (normalised stall corners) is computed from the
     corners stored in config/sequences.csv;
  2. stall_affine_search.py finds the 7-parameter affine transform that best
     maps it onto the canonical stall (x in [-0.8, 0.8], y in [-1, 1]);
  3. the transform is applied to the four corners of every normalised box,
     and the registered box is the axis-aligned bounding box of the
     transformed corners:

        Ln_reg, Tn_reg = min(x), min(y)
        Wn_reg, Hn_reg = (max(x)-min(x))/2, (max(y)-min(y))/2
        AR_reg = Wn_reg / Hn_reg,  logAR_reg = log(AR_reg)

     (Wn and Hn are in stall units while Ln/Tn are in half-stall units,
     hence the factor 2 when building the corners and the 1/2 afterwards.)

Input
-----
    data/06_scaled/<sequence>_interp_clipped_frames_scaled.csv
    config/sequences.csv

Output
------
    data/07_registered/<sequence>_interp_clipped_frames_scaled_reg.csv
        input columns + "<part> Ln_reg/Tn_reg/Wn_reg/Hn_reg/AR_reg/logAR_reg"
    data/07_registered/registration_params.csv   (transform found per sequence)

Usage
-----
    python scripts/07_register_bboxes.py                 # all sequences
    python scripts/07_register_bboxes.py --sequence <sequence> --nr 7
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import stall_affine_search as sas  # noqa: E402

PREFIXES = ["body box", "head box", "snout box"]
CORNERS = ["LT", "LB", "RB", "RT"]


def _load_normalisation_module():
    """Re-use the geometry of 06_normalize_to_stall.py (file name starts with a digit)."""
    spec = importlib.util.spec_from_file_location("normalize", HERE / "06_normalize_to_stall.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def register_bboxes(df, coeffs, prefixes=PREFIXES):
    """Apply the affine transform to the normalised boxes (vectorised)."""
    a00, a01, a02, a10, a11, a12 = coeffs
    for p in prefixes:
        Ln, Tn, Wn, Hn = (df[f"{p} {c}"].to_numpy(np.float64) for c in ("Ln", "Tn", "Wn", "Hn"))
        # corners LT, LB, RB, RT of every box, shape (N, 4)
        xs = np.stack([Ln, Ln, Ln + 2 * Wn, Ln + 2 * Wn], axis=1)
        ys = np.stack([Tn, Tn + 2 * Hn, Tn + 2 * Hn, Tn], axis=1)
        xt = xs * a00 + ys * a01 + a02
        yt = xs * a10 + ys * a11 + a12
        x_min, x_max = xt.min(axis=1), xt.max(axis=1)
        y_min, y_max = yt.min(axis=1), yt.max(axis=1)
        df[f"{p} Ln_reg"] = x_min
        df[f"{p} Tn_reg"] = y_min
        df[f"{p} Wn_reg"] = 0.5 * (x_max - x_min)
        df[f"{p} Hn_reg"] = 0.5 * (y_max - y_min)
        df[f"{p} AR_reg"] = df[f"{p} Wn_reg"] / (df[f"{p} Hn_reg"] + 1e-9)
        df[f"{p} logAR_reg"] = np.log(df[f"{p} AR_reg"] + 1e-12)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", help="Process one sequence only (default: all).")
    ap.add_argument("--sequences", default=ROOT / "config/sequences.csv")
    ap.add_argument("--input_dir", default=ROOT / "data/06_scaled")
    ap.add_argument("--output_dir", default=ROOT / "data/07_registered")
    ap.add_argument("--nr", type=int, default=7, help="Grid points per parameter (Nr^7 cells).")
    args = ap.parse_args()

    norm = _load_normalisation_module()
    seqs = pd.read_csv(args.sequences)
    if args.sequence:
        seqs = seqs[seqs["sequence"] == args.sequence]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = []
    for _, s in seqs.iterrows():
        name = s["sequence"]
        corners_px = [(s[f"{c}_x"], s[f"{c}_y"]) for c in CORNERS]
        floating = norm.floating_stall(corners_px)
        print(f"\n{s['animal']}: floating stall = {np.round(floating, 6).tolist()}")

        params, coeffs, cost = sas.best_affine(floating, sas.STANDARD_STALL, Nr=args.nr)
        print("  best transform: " + "  ".join(f"{k}={v:.4f}" for k, v in params.items())
              + f"  cost={cost:.6f}")

        df = pd.read_csv(Path(args.input_dir) / f"{name}_interp_clipped_frames_scaled.csv")
        df = register_bboxes(df, coeffs)
        out = out_dir / f"{name}_interp_clipped_frames_scaled_reg.csv"
        df.to_csv(out, index=False)
        print(f"  -> {out}")
        log.append(dict(sequence=name, animal=s["animal"], nr=args.nr, cost=cost, **params))

    params_csv = out_dir / "registration_params.csv"
    new = pd.DataFrame(log)
    if params_csv.exists() and args.sequence:
        old = pd.read_csv(params_csv)
        new = pd.concat([old[~old["sequence"].isin(new["sequence"])], new], ignore_index=True)
    new.to_csv(params_csv, index=False)
    print(f"\nTransform parameters -> {params_csv}")


if __name__ == "__main__":
    main()
