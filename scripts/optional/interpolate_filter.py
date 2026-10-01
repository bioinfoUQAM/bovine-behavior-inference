"""
OPTIONAL - NOT USED FOR THE THESIS RESULTS.

Linear interpolation of missing frames + zero-phase Butterworth low-pass
filter + derivatives (earlier variant of the trajectory step). The thesis
models use the unfiltered trajectories of 05_build_trajectories.py; the
filtered files kept in data/05_trajectories/filtered/ were produced by an
earlier version of this script and are provided for reference only.
"""
"""
inter_filt.py
-------------
Interpolates missing frames, applies a zero-phase Butterworth low-pass filter
to smooth bounding box trajectories, and computes frame-to-frame derivatives.
The output is the final feature-ready CSV for HMM-based behaviour analysis.

Usage:
    python inter_filt.py --input_dir <folder> --output_dir <folder> [options]

    Or process a single file:
    python inter_filt.py --input_dir path/to/roi_1_new.csv --output_dir path/to/output/

Arguments:
    --input_dir     Folder of per-ROI CSVs (output of assign_boxes.py), or single CSV path.
    --output_dir    Folder where feature CSVs will be saved.
    --frame_col     Name of the frame column (default: frame).
    --feature_cols  Space-separated list of columns to filter and differentiate.
                    Default: cow_L cow_T cow_W cow_H head_L head_T head_W head_H
                             snout_L snout_T snout_W snout_H
    --cutoff        Butterworth filter cutoff frequency in normalised units [0, 1]
                    (default: 0.1).
    --fs            Sampling frequency in Hz, i.e. video frame rate (default: 30).
    --order         Butterworth filter order (default: 1).
    --suffix        Output filename suffix (default: _interpolated_filtered_derivates).
    --plot          If set, show KDE plots of each feature column after processing.
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

try:
    import seaborn as sns
    _SEABORN = True
except ImportError:
    _SEABORN = False


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def fill_missing_frames(df, frame_col='frame', feature_cols=None):
    """
    Insert NaN rows for any frame number absent from the sequence.

    Parameters
    ----------
    df : pd.DataFrame
    frame_col : str
    feature_cols : list of str or None
        Columns to fill with NaN. If None, all non-frame columns are used.

    Returns
    -------
    pd.DataFrame sorted by frame number.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != frame_col]

    all_frames     = set(range(int(df[frame_col].min()), int(df[frame_col].max()) + 1))
    existing_frames = set(df[frame_col].unique())
    missing_frames  = sorted(all_frames - existing_frames)

    if not missing_frames:
        return df.sort_values(by=frame_col).reset_index(drop=True)

    missing_rows = [{frame_col: f, **{col: np.nan for col in feature_cols}}
                    for f in missing_frames]
    full_df = pd.concat([df, pd.DataFrame(missing_rows)], ignore_index=True)
    return full_df.sort_values(by=frame_col).reset_index(drop=True)


def interpolate_data(df, method='linear'):
    """
    Fill NaN values by linear interpolation along each column.

    Parameters
    ----------
    df : pd.DataFrame
    method : str
        Interpolation method passed to pd.DataFrame.interpolate (default: 'linear').

    Returns
    -------
    pd.DataFrame
    """
    return df.interpolate(method=method)


def butter_lowpass_filter(data, cutoff=0.1, fs=30, order=1):
    """
    Apply a zero-phase Butterworth low-pass filter to a 1-D signal.

    Parameters
    ----------
    data : array-like
    cutoff : float
        Cutoff frequency as a fraction of the Nyquist frequency (0 < cutoff < 1).
    fs : float
        Sampling frequency in Hz (video frame rate).
    order : int
        Filter order.

    Returns
    -------
    np.ndarray
    """
    nyquist = fs / 2.0
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def calculate_derivatives(df, frame_col='frame'):
    """
    Compute numerical first-order derivatives (via np.gradient) for all
    non-frame columns. Derivative columns are prefixed with 'd_'.

    Parameters
    ----------
    df : pd.DataFrame
    frame_col : str

    Returns
    -------
    pd.DataFrame of derivative columns (same length as df).
    """
    derivatives = {
        f'd_{col}': np.gradient(df[col].values)
        for col in df.columns if col != frame_col
    }
    return pd.DataFrame(derivatives)


def plot_kde(df, columns, title_prefix=''):
    """Plot KDE distributions for the given columns."""
    for col in columns:
        plt.figure(figsize=(8, 4))
        if _SEABORN:
            sns.kdeplot(df[col].dropna(), fill=True)
        else:
            df[col].dropna().plot(kind='kde')
        plt.title(f"{title_prefix}{col}")
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def process_file(csv_path, output_dir,
                 frame_col='frame',
                 feature_cols=None,
                 cutoff=0.1, fs=30, order=1,
                 suffix='_interpolated_filtered_derivates',
                 show_plots=False):
    """
    Full interpolation → filtering → derivative pipeline for one CSV file.

    Parameters
    ----------
    csv_path : str
    output_dir : str
    frame_col : str
    feature_cols : list of str or None
        Columns to process. If None, all non-frame columns are used.
    cutoff : float
    fs : float
    order : int
    suffix : str
    show_plots : bool
    """
    print(f"\nProcessing: {csv_path}")
    data = pd.read_csv(csv_path)
    data = data.replace('none', np.nan)

    if feature_cols is None:
        feature_cols = [c for c in data.columns if c != frame_col]

    # Keep only the required columns
    cols_to_keep = [frame_col] + [c for c in feature_cols if c in data.columns]
    data = data[cols_to_keep].copy()
    original_df = data.copy()

    # Step 1 — fill missing frames
    full_df = fill_missing_frames(data, frame_col=frame_col, feature_cols=feature_cols)

    # Step 2 — interpolate
    interpolated_df = interpolate_data(full_df)

    # Step 3 — Butterworth low-pass filter
    filtered_df = interpolated_df[feature_cols].copy()
    for col in feature_cols:
        filtered_df[col] = butter_lowpass_filter(filtered_df[col].values,
                                                  cutoff=cutoff, fs=fs, order=order)
    filtered_df.insert(0, frame_col, interpolated_df[frame_col].values)

    # Step 4 — derivatives
    derivatives_df = calculate_derivatives(filtered_df, frame_col=frame_col)

    # Concatenate and drop the last row (gradient boundary artefact)
    output_df = pd.concat([filtered_df, derivatives_df], axis=1).iloc[:-1]

    os.makedirs(output_dir, exist_ok=True)
    stem     = os.path.splitext(os.path.basename(csv_path))[0]
    out_path = os.path.join(output_dir, f"{stem}{suffix}.csv")
    output_df.to_csv(out_path, index=False)
    print(f"Saved to: {out_path}  |  shape: {output_df.shape}")

    if show_plots:
        plot_kde(output_df, feature_cols, title_prefix=f"{stem} — ")

    return output_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

DEFAULT_FEATURE_COLS = [
    'cow_L', 'cow_T', 'cow_W', 'cow_H',
    'head_L', 'head_T', 'head_W', 'head_H',
    'snout_L', 'snout_T', 'snout_W', 'snout_H'
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interpolate, filter, and differentiate bounding box trajectories.")
    parser.add_argument("--input_dir",    required=True,
                        help="Folder of per-ROI CSVs or path to a single CSV.")
    parser.add_argument("--output_dir",   required=True,
                        help="Output folder for feature CSVs.")
    parser.add_argument("--frame_col",    default="frame",
                        help="Name of the frame column.")
    parser.add_argument("--feature_cols", nargs='+', default=DEFAULT_FEATURE_COLS,
                        help="Columns to filter and differentiate.")
    parser.add_argument("--cutoff",  type=float, default=0.1,
                        help="Butterworth cutoff frequency (normalised, default: 0.1).")
    parser.add_argument("--fs",      type=float, default=30.0,
                        help="Video frame rate in Hz (default: 30).")
    parser.add_argument("--order",   type=int,   default=1,
                        help="Butterworth filter order (default: 1).")
    parser.add_argument("--suffix",  default="_interpolated_filtered_derivates",
                        help="Output filename suffix.")
    parser.add_argument("--plot",    action="store_true",
                        help="Show KDE plots after processing.")
    args = parser.parse_args()

    if os.path.isfile(args.input_dir):
        csv_files = [args.input_dir]
    else:
        csv_files = sorted([
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir) if f.endswith(".csv")
        ])

    for csv_path in csv_files:
        process_file(csv_path,
                     output_dir=args.output_dir,
                     frame_col=args.frame_col,
                     feature_cols=args.feature_cols,
                     cutoff=args.cutoff,
                     fs=args.fs,
                     order=args.order,
                     suffix=args.suffix,
                     show_plots=args.plot)

    print("\nAll files processed.")
