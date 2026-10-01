"""
qc/registration_error_report.py
===============================
Quality control of the stall registration (thesis 3.5), complementary to
visualize_registration_warp.py:
    row 1 : video frame with the floating (before) and transformed (after)
            stall overlaid on the reference stall;
    row 2 : residual Euclidean error of each corner (LT, LB, RB, RT) before
            and after the transform (normalised space);
    row 3 : before / after in normalised stall space.
Without --video_dir/video file the first row is skipped, so the report can
be produced from the repository data alone.

Input : config/sequences.csv, (optional) data/videos/<sequence>.mp4
Output: results/registration_qc/<animal>_registration_error_report.pdf

Usage
-----
    python scripts/qc/registration_error_report.py --animal Maisie
    python scripts/qc/registration_error_report.py --all
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _qc_common import (ROOT, CORNER_LABELS, sequence_row, stall_registration,  # noqa: E402
                        norm_to_pixel, video_frame, close_loop)


def report(row, video_dir, nr, out):
    r = stall_registration(row, nr)
    err_before = np.linalg.norm(r["floating"] - r["standard"], axis=1)
    err_after = np.linalg.norm(r["transformed"] - r["standard"], axis=1)
    video = Path(video_dir) / row["video_file"]
    has_video = video.exists()

    fig = plt.figure(figsize=(14, 14 if has_video else 9))
    gs = fig.add_gridspec(3 if has_video else 2, 2,
                          height_ratios=[1.6, 0.9, 1.0] if has_video else [0.9, 1.0])
    k = 0
    if has_video:
        img, idx, n = video_frame(video)
        geo = (r["center"], r["half_w"], r["half_h"])
        std_px = norm_to_pixel(r["standard"], *geo)
        for col, (pts, title, color, style) in enumerate([
                (r["floating"], "BEFORE registration (raw floating)", "tab:orange", "o-"),
                (r["transformed"], "AFTER registration (transformed)", "tab:red", "s-")]):
            ax = fig.add_subplot(gs[0, col])
            ax.imshow(img)
            ax.plot(*close_loop(std_px).T, "o-", color="tab:green", label="Reference")
            ax.plot(*close_loop(norm_to_pixel(pts, *geo)).T, style, color=color, label=title.split(" (")[0])
            ax.set_title(f"Frame {idx}/{n} - {title}")
            ax.legend(fontsize=8, loc="upper right")
            ax.axis("off")
        k = 1
    else:
        print(f"  [{row['animal']}] video not found ({video}); image row skipped.")

    ax = fig.add_subplot(gs[k, :])
    x = np.arange(4)
    ax.bar(x - 0.175, err_before, 0.35, color="tab:orange", label="Before registration")
    ax.bar(x + 0.175, err_after, 0.35, color="tab:red", label="After registration")
    ax.set_xticks(x, CORNER_LABELS)
    ax.set_ylabel("Euclidean error (normalised space)")
    ax.set_title(f"{row['animal']} - residual error per corner (sum of squares = {r['cost']:.4f})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    for col, (pts, title, color, style) in enumerate([
            (r["floating"], "Raw floating vs reference", "tab:orange", "o-"),
            (r["transformed"], "Transformed vs reference", "tab:red", "s--")]):
        ax = fig.add_subplot(gs[k + 1, col])
        ax.plot(*close_loop(r["standard"]).T, "o-", color="tab:green", label="Reference")
        ax.plot(*close_loop(pts).T, style, color=color, label=title.split(" vs")[0])
        for i, c in enumerate(CORNER_LABELS):
            ax.annotate(c, pts[i], fontsize=7)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    path = out / f"{row['animal']}_registration_error_report.pdf"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  {row['animal']}: total corner error before {err_before.sum():.4f} | "
          f"after {err_after.sum():.4f} -> {path}")
    return dict(animal=row["animal"], **{f"err_before_{c}": e for c, e in zip(CORNER_LABELS, err_before)},
                **{f"err_after_{c}": e for c, e in zip(CORNER_LABELS, err_after)}, cost=r["cost"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--animal")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--video_dir", default=ROOT / "data/videos")
    ap.add_argument("--nr", type=int, default=7)
    ap.add_argument("--output_dir", default=ROOT / "results/registration_qc")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = (pd.read_csv(ROOT / "config/sequences.csv").to_dict("records") if args.all
            else [sequence_row(args.animal)])
    res = [report(r, args.video_dir, args.nr, out) for r in rows]
    if args.all:
        pd.DataFrame(res).to_csv(out / "corner_errors.csv", index=False)


if __name__ == "__main__":
    main()
