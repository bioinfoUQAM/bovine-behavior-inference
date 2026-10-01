"""
10_annotation_statistics.py
===========================
Thesis section 3.2.2 / Table 3.3 (descriptive statistics of the annotations).

For every behaviour, over the whole corpus:
  * cumulative duration (h)  - each row weighs the time step to the next row
                               (Times column; the last row uses the median step);
  * % of the total corpus time;
  * number of occurrences    - continuous episodes (0 -> 1 transitions, an
                               episode active on the first row counts as one).
Several behaviours can be active at the same time, so the percentages can
sum to more than 100 %.

Only the 17 Observer behaviour columns are analysed. (The original version
took every column from "Defecation" to the end of the file, which on the
scaled CSVs also included the box features; the thesis table only reports
the behaviours, whose total is 134 occurrences.)

Input
-----
    data/05_trajectories/*_interp_clipped_frames.csv   (default; any CSV with
    a "Times" column and the behaviour columns works, e.g. data/06_scaled)

Output (results/annotation_statistics/)
------
    annotation_statistics.csv     Table 3.3
    sequence_summary.csv          duration and number of rows per sequence
    annotation_statistics.tex     LaTeX rows of Table 3.3

Usage
-----
    python scripts/10_annotation_statistics.py
    python scripts/10_annotation_statistics.py --input "data/06_scaled/*.csv"
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BEHAVIOURS = [
    "Defecation", "Drinking", "Eating", "Exploration", "Grooming",
    "Head Shaking (as if to detach)", "Kneeling", "Lying", "Lying Down",
    "Not Visible", "Other (including inactive)", "Scratching",
    "Social Interaction", "Standing", "Standing Up", "Urination",
    "Vigilance toward the door",
]


def row_durations(times):
    t = pd.to_numeric(times, errors="coerce").to_numpy(float)
    if np.isnan(t).any() or len(t) < 2:
        raise ValueError("The Times column must be numeric with at least two rows.")
    dt = np.diff(t)
    if (dt <= 0).any():
        raise ValueError("Times must be strictly increasing.")
    return np.r_[dt, float(np.median(dt))]


def count_episodes(active):
    active = active.astype(bool)
    return int(active[0]) + int(np.sum(~active[:-1] & active[1:]))


def analyse_sequence(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.replace("\ufeff", "").str.strip()
    df = df.sort_values("Times", kind="stable").reset_index(drop=True)
    w = row_durations(df["Times"])
    rows = []
    for b in [c for c in BEHAVIOURS if c in df.columns]:
        active = pd.to_numeric(df[b], errors="coerce").fillna(0).to_numpy() > 0
        rows.append(dict(Behaviour=b, Duration_s=float(w[active].sum()),
                         Occurrences=count_episodes(active)))
    return pd.DataFrame(rows), float(w.sum()), len(df)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(ROOT / "data/05_trajectories/*_interp_clipped_frames.csv"))
    ap.add_argument("--output_dir", default=ROOT / "results/annotation_statistics")
    args = ap.parse_args()

    files = sorted(glob.glob(args.input))
    if not files:
        raise FileNotFoundError(f"No file matches {args.input}")

    stats, summary = [], []
    for f in files:
        s, dur, n = analyse_sequence(f)
        stats.append(s)
        summary.append(dict(Sequence=Path(f).name, Duration_s=dur, Duration_h=dur / 3600, Rows=n))
    total_s = sum(s["Duration_s"] for s in summary)

    table = (pd.concat(stats).groupby("Behaviour", as_index=False, sort=False)
             .agg(Duration_s=("Duration_s", "sum"), Occurrences=("Occurrences", "sum")))
    table["Cumulative_duration_h"] = table["Duration_s"] / 3600
    table["Percent_total_time"] = 100 * table["Duration_s"] / total_s
    table = table.sort_values("Cumulative_duration_h", ascending=False)
    table = table[["Behaviour", "Cumulative_duration_h", "Percent_total_time", "Occurrences"]]
    total = pd.DataFrame([dict(Behaviour="Total (corpus duration)", Cumulative_duration_h=total_s / 3600,
                               Percent_total_time=100.0, Occurrences=int(table["Occurrences"].sum()))])
    table = pd.concat([table, total], ignore_index=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "annotation_statistics.csv", index=False, float_format="%.4f")
    pd.DataFrame(summary).to_csv(out / "sequence_summary.csv", index=False, float_format="%.4f")
    tex = [f"{r.Behaviour.replace('&', chr(92) + '&')} & {r.Cumulative_duration_h:.2f} & "
           f"{r.Percent_total_time:.2f} & {int(r.Occurrences)} \\\\" for r in table.itertuples()]
    (out / "annotation_statistics.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
