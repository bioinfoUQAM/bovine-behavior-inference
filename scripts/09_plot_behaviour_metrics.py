"""
09_plot_behaviour_metrics.py
============================
Thesis chapter 5 (per-behaviour AUC / NLL figures).

Bar plots of the per-behaviour AUC and mean NLL produced by
08_gmmhmm_loocv.py, averaged over the test sequences and for each sequence.

Input
-----
    results/<experiment>/eval_per_behaviour.csv
    (older files with a "prev" column instead of "prevalence" are accepted)

Output (default: results/<experiment>/behaviour_plots/)
------
    GLOBAL_AUC_per_behaviour.png, GLOBAL_NLL_per_behaviour.png
    <sequence>_AUC_per_behaviour.png, <sequence>_NLL_per_behaviour.png

Usage
-----
    python scripts/09_plot_behaviour_metrics.py --eval_csv results/exp1/eval_per_behaviour.csv
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BEH_COLS = [
    "Drinking", "Eating", "Exploration", "Grooming",
    "Head Shaking (as if to detach)", "Kneeling", "Lying", "Lying Down",
    "Scratching", "Social Interaction", "Standing", "Standing Up",
]


def bar(values, labels, ylabel, title, color, path, ylim=None):
    x = np.arange(len(labels))
    plt.figure(figsize=(12, 5))
    plt.bar(x, values, color=color)
    plt.xticks(x, labels, rotation=45, ha="right")
    if ylim:
        plt.ylim(*ylim)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print("Saved:", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_csv", required=True)
    ap.add_argument("--output_dir")
    args = ap.parse_args()

    df = pd.read_csv(args.eval_csv).rename(columns={"prev": "prevalence"})
    missing = {"seq_name", "behaviour", "mean_nll", "auc"} - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {args.eval_csv}: {missing}")
    out = Path(args.output_dir or Path(args.eval_csv).parent / "behaviour_plots")
    out.mkdir(parents=True, exist_ok=True)

    order = [b for b in BEH_COLS if b in set(df["behaviour"])]
    auc = df.groupby("behaviour")["auc"].mean().reindex(order)
    nll = df.groupby("behaviour")["mean_nll"].mean().reindex(order)
    bar(auc.values, order, "AUC", "Per-behaviour AUC (mean over test sequences)",
        "tab:blue", out / "GLOBAL_AUC_per_behaviour.png", (0, 1))
    bar(nll.values, order, "Mean NLL", "Per-behaviour NLL (mean over test sequences, lower = better)",
        "tab:red", out / "GLOBAL_NLL_per_behaviour.png")

    for name, sub in df.groupby("seq_name"):
        sub = sub.set_index("behaviour").reindex(order)
        stem = Path(name).stem
        bar(sub["auc"].values, order, "AUC", f"AUC per behaviour - {stem}",
            "tab:blue", out / f"{stem}_AUC_per_behaviour.png", (0, 1))
        bar(sub["mean_nll"].values, order, "Mean NLL", f"NLL per behaviour - {stem} (lower = better)",
            "tab:red", out / f"{stem}_NLL_per_behaviour.png")


if __name__ == "__main__":
    main()
