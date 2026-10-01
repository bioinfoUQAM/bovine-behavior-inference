"""Shared helpers of the QC scripts (stall geometry, video frame access)."""
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1]
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
import stall_affine_search as sas  # noqa: E402

_spec = importlib.util.spec_from_file_location("normalize_to_stall", SCRIPTS / "06_normalize_to_stall.py")
norm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(norm)

CORNER_LABELS = ["LT", "LB", "RB", "RT"]


def sequence_row(animal_or_sequence, config=ROOT / "config/sequences.csv"):
    seqs = pd.read_csv(config)
    m = seqs[(seqs["animal"] == animal_or_sequence) | (seqs["sequence"] == animal_or_sequence)]
    if m.empty:
        raise KeyError(f"{animal_or_sequence} not in {config} (animals: {list(seqs['animal'])})")
    return m.iloc[0]


def stall_registration(row, nr=7):
    """Pixel geometry of the stall, floating stall and best affine transform."""
    corners = norm.corners_from_config(row)
    cx, cy, ws, hs = norm.stall_center_and_size_from_4pts(corners)
    floating = norm.floating_stall(corners)
    params, coeffs, cost = sas.best_affine(floating, sas.STANDARD_STALL, Nr=nr)
    return dict(center=(cx, cy), half_w=ws / 2, half_h=hs / 2, floating=floating,
                standard=sas.STANDARD_STALL, params=params, coeffs=coeffs, cost=cost,
                transformed=sas.apply_affine(floating, *coeffs))


def norm_to_pixel(pts, center, half_w, half_h):
    return np.stack([center[0] + pts[:, 0] * half_w, center[1] + pts[:, 1] * half_h], axis=1)


def norm_affine_to_pixel_affine(coeffs, center, half_w, half_h):
    """2x3 pixel matrix of pixel -> normalised -> affine -> pixel."""
    a00, a01, a02, a10, a11, a12 = coeffs
    A = np.array([[a00, a01], [a10, a11]])
    D = np.diag([half_w, half_h])
    M = D @ A @ np.linalg.inv(D)
    t = np.asarray(center, float) + D @ np.array([a02, a12]) - M @ np.asarray(center, float)
    return np.hstack([M, t.reshape(2, 1)])


def video_frame(video_path, index=None):
    """RGB frame `index` (default: middle frame), index, number of frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path} (see data/videos/README.md)")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    index = n // 2 if index is None else index
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"Cannot read frame {index} of {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), index, n


def close_loop(pts):
    return np.vstack([pts, pts[0]])
