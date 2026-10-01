"""
11_behaviour_matrices.py
========================
Thesis section 4.2.2 / Figure 4.1 (structure of the annotations).

Computes, on the frame-level labels of all sequences:
  * co-occurrence matrix   - (i, j) = frames where behaviours i and j are both
                             active (diagonal = frequency of each behaviour),
                             in counts, % of all frames, and conditional
                             % P(j active | i active);
  * transition matrix      - t -> t+1, every pair of labels active at t and
                             t+1 is counted, never across two sequences;
                             row-normalised to P(j at t+1 | i at t);
  * posture x action matrix - the co-occurrence restricted to postures
                             (rows) and actions (columns), in % of all frames
                             and conditional on the posture.

Input
-----
    data/06_scaled/*_interp_clipped_frames_scaled.csv  (default, as in the
    thesis; data/05_trajectories gives identical labels)

Output (results/behaviour_matrices/)
------
    behaviour_matrices.xlsx                      every matrix, one sheet each
    transition_matrix.pdf                        Figure 4.1
    posture_action_percent.pdf                   posture x action (% of frames)
    posture_action_conditional.pdf               posture x action (conditional %)

Usage
-----
    python scripts/11_behaviour_matrices.py
    python scripts/11_behaviour_matrices.py --without_technical   # 15 behaviours
"""
import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BEHAVIOURS = [
    "Defecation", "Drinking", "Eating", "Exploration", "Grooming",
    "Head Shaking (as if to detach)", "Kneeling", "Lying", "Lying Down",
    "Not Visible", "Other (including inactive)", "Scratching",
    "Social Interaction", "Standing", "Standing Up", "Urination",
    "Vigilance toward the door",
]
TECHNICAL = ["Not Visible", "Other (including inactive)"]
POSTURES = ["Standing", "Lying", "Kneeling", "Lying Down", "Standing Up"]
ACTIONS = ["Defecation", "Drinking", "Eating", "Exploration", "Grooming",
           "Head Shaking (as if to detach)", "Other (including inactive)", "Scratching",
           "Social Interaction", "Urination", "Vigilance toward the door"]


def save_heatmap(matrix, title, path, fmt, figsize, xlabel, ylabel):
    plt.figure(figsize=figsize)
    sns.heatmap(matrix, cmap="YlOrRd", annot=True, fmt=fmt, linewidths=0.4,
                linecolor="white", cbar_kws={"label": "Percentage"})
    plt.title(title, fontsize=14, weight="bold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(ROOT / "data/06_scaled/*_scaled.csv"))
    ap.add_argument("--output_dir", default=ROOT / "results/behaviour_matrices")
    ap.add_argument("--without_technical", action="store_true",
                    help="Drop 'Not Visible' and 'Other (including inactive)'.")
    args = ap.parse_args()

    cols = [c for c in BEHAVIOURS if not (args.without_technical and c in TECHNICAL)]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white")

    seqs = []
    for f in sorted(glob.glob(args.input)):
        df = pd.read_csv(f)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{f}: missing columns {missing}")
        Y = (df[cols].apply(pd.to_numeric, errors="coerce").fillna(0) > 0).astype(np.int64)
        n_act = Y.sum(axis=1)
        print(f"{Path(f).stem}: {len(Y):,} frames | active labels/frame mean={n_act.mean():.3f} "
              f"min={n_act.min()} max={n_act.max()} | exactly 2 labels: {(n_act == 2).mean() * 100:.2f}%")
        seqs.append(Y.to_numpy())
    if not seqs:
        raise FileNotFoundError(f"No file matches {args.input}")

    n_frames = sum(len(Y) for Y in seqs)
    co = pd.DataFrame(sum(Y.T @ Y for Y in seqs), index=cols, columns=cols)
    co_pct = (co / n_frames * 100).round(3)
    co_cond = (co.div(np.diag(co).astype(float), axis=0)
               .replace([np.inf, -np.inf], np.nan) * 100).round(2)

    tr = pd.DataFrame(sum(Y[:-1].T @ Y[1:] for Y in seqs if len(Y) > 1), index=cols, columns=cols)
    tr_pct = (tr.div(tr.sum(axis=1).replace(0, np.nan), axis=0) * 100).round(2)

    post = [c for c in POSTURES if c in cols]
    act = [c for c in ACTIONS if c in cols]
    pa = co.loc[post, act]
    pa_pct = (pa / n_frames * 100).round(3)
    pa_cond = (pa.div(pa.sum(axis=1).replace(0, np.nan), axis=0) * 100).round(2)

    with pd.ExcelWriter(out / "behaviour_matrices.xlsx", engine="openpyxl") as w:
        for name, m in [("Cooccurrence_counts", co), ("Cooccurrence_pct", co_pct),
                        ("Cooccurrence_conditional", co_cond), ("Transitions_counts", tr),
                        ("Transitions_pct", tr_pct), ("Posture_action_counts", pa),
                        ("Posture_action_pct", pa_pct), ("Posture_action_conditional", pa_cond)]:
            m.to_excel(w, sheet_name=name)
    print(f"Excel: {out / 'behaviour_matrices.xlsx'}")

    save_heatmap(pa_pct, "Posture-action co-occurrence in the whole corpus (%)",
                 out / "posture_action_percent.pdf", ".2f", (15, 5), "Action", "Posture")
    save_heatmap(pa_cond, "Distribution of actions conditional on posture (%)",
                 out / "posture_action_conditional.pdf", ".1f", (15, 5), "Action", "Posture")
    save_heatmap(tr_pct, "Behaviour transitions between consecutive frames (%)",
                 out / "transition_matrix.pdf", ".1f", (16, 13),
                 "Behaviour at $t+1$", "Behaviour at $t$")

    print("\nPosture x action (% of all frames)\n" + pa_pct.to_string())
    print("\nPosture x action (% conditional on posture)\n" + pa_cond.to_string())


if __name__ == "__main__":
    main()
