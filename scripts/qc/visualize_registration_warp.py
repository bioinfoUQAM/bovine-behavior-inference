"""
qc/visualize_registration_warp.py
=================================
Quality control of the stall registration (thesis 3.5, Figure 3.x).

Applies the affine transform found by stall_affine_search (in normalised
stall space) to the video image itself with cv2.warpAffine, to obtain a
real before / after view:
    top    : original frame / warped frame, with the reference, floating and
             transformed stall corners overlaid;
    bottom : the same comparison in normalised stall space.

Input : config/sequences.csv (stall corners), data/videos/<sequence>.mp4
Output: results/registration_qc/<animal>_registration_warp.pdf

Usage
-----
    python scripts/qc/visualize_registration_warp.py --animal Maisie
    python scripts/qc/visualize_registration_warp.py --animal Maisie --frame 1000
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _qc_common import (ROOT, CORNER_LABELS, sequence_row, stall_registration,  # noqa: E402
                        norm_to_pixel, norm_affine_to_pixel_affine, video_frame, close_loop)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animal", required=True, help="Animal name or sequence id from config/sequences.csv")
    ap.add_argument("--video_dir", default=ROOT / "data/videos")
    ap.add_argument("--frame", type=int, help="Frame index (default: middle of the video)")
    ap.add_argument("--nr", type=int, default=7)
    ap.add_argument("--output_dir", default=ROOT / "results/registration_qc")
    args = ap.parse_args()

    row = sequence_row(args.animal)
    r = stall_registration(row, args.nr)
    img, idx, n = video_frame(Path(args.video_dir) / row["video_file"], args.frame)
    h, w = img.shape[:2]
    geo = (r["center"], r["half_w"], r["half_h"])
    fl_px, std_px, tr_px = (norm_to_pixel(p, *geo) for p in (r["floating"], r["standard"], r["transformed"]))
    warped = cv2.warpAffine(img, norm_affine_to_pixel_affine(r["coeffs"], *geo), (w, h),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(30, 30, 30))

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1.0])
    panels = [(img, fl_px, "o-", "tab:orange", "Floating (raw)", f"BEFORE - original frame ({idx}/{n})"),
              (warped, tr_px, "s--", "tab:red", "Transformed (warp)", f"AFTER - warped frame (cost={r['cost']:.4f})")]
    for col, (im, pts, style, color, lab, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(im)
        ax.plot(*close_loop(std_px).T, "o-", color="tab:green", label="Reference")
        ax.plot(*close_loop(pts).T, style, color=color, label=lab)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper right")
        ax.axis("off")
    for col, (pts, style, color, lab, title) in enumerate([
            (r["floating"], "o-", "tab:orange", "Floating (raw)", "Normalised space - before"),
            (r["transformed"], "s--", "tab:red", "Transformed", "Normalised space - after")]):
        ax = fig.add_subplot(gs[1, col])
        ax.plot(*close_loop(r["standard"]).T, "o-", color="tab:green", label="Reference")
        ax.plot(*close_loop(pts).T, style, color=color, label=lab)
        for i, c in enumerate(CORNER_LABELS):
            ax.annotate(c, pts[i], fontsize=7)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{row['animal']}_registration_warp.pdf"
    plt.savefig(path, dpi=200)
    plt.close()
    print("Transform:", {k: round(v, 4) for k, v in r["params"].items()})
    print("Saved:", path)


if __name__ == "__main__":
    main()
