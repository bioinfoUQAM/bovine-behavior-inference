"""
08_gmmhmm_loocv.py
==================
Thesis sections 3.6 - 3.8 and 4.4 - 4.5 (features, GMM-HMM, behaviour
classifier, leave-one-sequence-out evaluation).

Pipeline
--------
1. Features (thesis 3.6.2 - 3.6.3), computed per sequence from the registered
   boxes of 07_register_bboxes.py:
     * 15 spatial descriptors: Ln_reg, Tn_reg, Wn_reg, Hn_reg, logAR_reg for
       the body, head and snout boxes;
     * 6 relative descriptors: head-body offset (L, T), snout-head offset
       (L, T), snout-body vertical offset, head/body width ratio;
     * first and second temporal differences of these 21 descriptors
       -> 63-dimensional observation vector x_t.
   Rows with a non-finite component (the first two rows of each sequence)
   are dropped.
2. Standardisation of x_t (global or per sequence, see --scaling).
3. GMM-HMM (hmmlearn.GMMHMM, diagonal covariances, N_mix components per
   state) fitted on the training sequences; its posterior state
   probabilities gamma_t (forward-backward) are the input of
4. one calibrated logistic regression per behaviour (multi-label,
   12 behaviours), CalibratedClassifierCV(sigmoid, cv=3).
5. Leave-one-sequence-out cross-validation (LOOCV):
     a. the number of hidden states k is chosen among --loocv_k by the
        macro AUC of the out-of-fold predictions;
     b. the LOOCV is (re)run with the selected k; per-behaviour decision
        thresholds maximise F1 on the training folds only;
     c. a final model is fitted on all sequences, k chosen by BIC among
        --final_k (deployment model, also reported in the thesis).
6. Metrics: mean binary NLL, macro AUC, micro / macro F1, globally (on the
   median-smoothed out-of-fold probabilities, threshold 0.5) and per
   sequence (raw probabilities, train-fold thresholds).

Thesis experiments
------------------
    --preset exp1   all 5 sequences                    (thesis Exp1)
    --preset exp2   without Mack                       (thesis Exp2)
    --preset exp3   without Mack and Maisie            (thesis Exp3)
A preset only fills default values; any option given explicitly wins.

Input
-----
    data/07_registered/<sequence>_interp_clipped_frames_scaled_reg.csv

Output (in --output_dir, default results/<preset or "run">/)
------
    report.txt                     summary (same layout as the thesis tables)
    eval_by_sequence.csv           per-sequence metrics
    eval_per_behaviour.csv         per-sequence x behaviour metrics
    k_selection.json               macro AUC of every k tried + chosen k
    final_model_bic.json           BIC of every k of the final model
    loocv_predictions.npz          out-of-fold Y / P / sequence index / thresholds
    *_NLL_heatmap.png, BIC_state_selection.png
    frames_with_probabilities.csv  (only with --save_frame_csv; large file)
    final_model.pkl                (only with --save_model)

Usage
-----
    python scripts/08_gmmhmm_loocv.py --preset exp1
    python scripts/08_gmmhmm_loocv.py --preset exp2
    python scripts/08_gmmhmm_loocv.py --exclude Mack --loocv_k 14 --output_dir results/my_run
"""

import argparse
import json
import multiprocessing
import pickle
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from hmmlearn import hmm  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.ndimage import median_filter  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]

# The 12 behaviours retained for classification (thesis 4.2.2). Defecation,
# Urination, Vigilance toward the door, Not Visible and Other are excluded.
BEH_COLS = [
    "Drinking", "Eating", "Exploration", "Grooming",
    "Head Shaking (as if to detach)", "Kneeling", "Lying", "Lying Down",
    "Scratching", "Social Interaction", "Standing", "Standing Up",
]
PARTS = ["body box", "head box", "snout box"]
BASE_FEATURES = [f"{p} {c}_reg" for p in PARTS for c in ("Ln", "Tn", "Wn", "Hn", "logAR")]
REL_FEATURES = ["feat_head_rel_L", "feat_head_rel_T", "feat_snout_rel_L",
                "feat_snout_rel_T", "feat_snout_body_T_diff", "feat_head_body_W_ratio"]
EPS = 1e-6

# Settings of the three experiments reported in the thesis (chapter 5).
# Exp1 was run with global standardisation and without class re-weighting;
# Exp2 and Exp3 with per-sequence standardisation, class re-weighting of the
# rare behaviours and k fixed to 14 (see README, "Reproducibility notes").
PRESETS = {
    "exp1": dict(exclude=[], scaling="global", class_weight="none", constant_label_fix=False,
                 loocv_k=[8, 10, 12, 14, 16], final_k=[10, 12, 14, 16, 18]),
    "exp2": dict(exclude=["Mack"], scaling="per-sequence", class_weight="balanced-rare",
                 constant_label_fix=False, loocv_k=[14], final_k=[14]),
    "exp3": dict(exclude=["Mack", "Maisie"], scaling="per-sequence", class_weight="balanced-rare",
                 constant_label_fix=False, loocv_k=[14], final_k=[14]),
}
DEFAULTS = dict(exclude=[], scaling="per-sequence", class_weight="balanced-rare",
                constant_label_fix=True,
                loocv_k=[10, 12, 14, 16, 18], final_k=[10, 12, 14, 16, 18])


# ─────────────────────────── metrics & plots ───────────────────────────

def bce_nll(Y, P, eps=1e-6):
    P = np.clip(P, eps, 1 - eps)
    return -(Y * np.log(P) + (1 - Y) * np.log(1 - P))


def safe_auc(y, s):
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def eval_metrics(Y, P, thresholds=None):
    nll = bce_nll(Y, P)
    aucs = [safe_auc(Y[:, j], P[:, j]) for j in range(Y.shape[1])]
    th = np.array([0.5] * len(BEH_COLS) if thresholds is None
                  else [thresholds[b] for b in BEH_COLS])
    Yb = (P >= th).astype(int)
    return dict(
        mean_nll=float(nll.mean()), mean_nll_per_beh=nll.mean(axis=0), aucs=aucs,
        macro_auc=float(np.nanmean(aucs)) if np.isfinite(aucs).any() else np.nan,
        f1_micro=float(f1_score(Y.astype(int), Yb, average="micro", zero_division=0)),
        f1_macro=float(f1_score(Y.astype(int), Yb, average="macro", zero_division=0)),
    )


def save_nll_heatmap(Y, P, title, out_png, max_frames=20_000):
    nll = bce_nll(Y, P)
    n = min(len(nll), max_frames)
    idx = np.argsort(-Y.mean(axis=0))[:len(BEH_COLS)]
    plt.figure(figsize=(16, 6))
    plt.imshow(nll[:n, idx].T, aspect="auto", interpolation="nearest")
    plt.colorbar(label="NLL (lower = better)")
    plt.yticks(np.arange(len(idx)), [BEH_COLS[j] for j in idx])
    plt.xlabel("Frame index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ─────────────────────────── features ───────────────────────────

def load_sequences(files):
    dfs = []
    for i, f in enumerate(files):
        df = pd.read_csv(f).sort_values("Frame").reset_index(drop=True)
        missing = [c for c in BEH_COLS + BASE_FEATURES if c not in df.columns]
        if missing:
            raise KeyError(f"{f.name}: missing columns {missing}")
        df["seq_id"] = i
        df["seq_name"] = f.name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def build_features(big):
    big["feat_head_rel_L"] = big["head box Ln_reg"] - big["body box Ln_reg"]
    big["feat_head_rel_T"] = big["head box Tn_reg"] - big["body box Tn_reg"]
    big["feat_snout_rel_L"] = big["snout box Ln_reg"] - big["head box Ln_reg"]
    big["feat_snout_rel_T"] = big["snout box Tn_reg"] - big["head box Tn_reg"]
    big["feat_snout_body_T_diff"] = big["snout box Tn_reg"] - big["body box Tn_reg"]
    big["feat_head_body_W_ratio"] = big["head box Wn_reg"] / (big["body box Wn_reg"] + EPS)

    feats = BASE_FEATURES + REL_FEATURES
    d1 = []
    for c in feats:                       # first difference, within each sequence
        big[f"ddt {c}"] = big.groupby("seq_id")[c].diff()
        d1.append(f"ddt {c}")
    d2 = []
    for c in d1:                          # second difference
        big[f"ddt2 {c}"] = big.groupby("seq_id")[c].diff()
        d2.append(f"ddt2 {c}")
    all_feats = feats + d1 + d2

    ok = np.isfinite(big[all_feats].to_numpy(float)).all(axis=1)
    big = big.loc[ok].reset_index(drop=True)
    return big, all_feats


def standardise(X, lengths, mode):
    if mode == "global":
        return StandardScaler().fit_transform(X)
    ends = np.cumsum(lengths)
    starts = ends - np.asarray(lengths)
    return np.vstack([StandardScaler().fit_transform(X[s:e]) for s, e in zip(starts, ends)])


# ─────────────────────────── models ───────────────────────────

def bic_gmmhmm(model, X, lengths):
    k, d, m = model.n_components, X.shape[1], model.n_mix
    n_params = k * (k - 1) + 2 * k * m * d + k * (m - 1)
    return -2 * model.score(X, lengths) + n_params * np.log(len(X))


def fit_gmmhmm(k, X, lengths, cfg, n_iter, n_restarts):
    """Best of `n_restarts` fits (seeds 0..n-1) by log-likelihood."""
    best, best_ll = None, -np.inf
    for seed in range(n_restarts):
        try:
            m = hmm.GMMHMM(n_components=k, n_mix=cfg.n_mix, covariance_type="diag",
                           n_iter=n_iter, tol=cfg.tol, min_covar=cfg.min_covar,
                           random_state=seed, implementation="log")
            m.fit(X, lengths)
            ll = m.score(X, lengths)
        except Exception as e:                       # numerical failure of one restart
            print(f"    [WARN] k={k} seed={seed} failed: {e}")
            continue
        if ll > best_ll:
            best, best_ll = m, ll
    return best, best_ll


def fit_behaviour_classifiers(gamma, Y, class_weight):
    clfs = []
    for j in range(Y.shape[1]):
        y = Y[:, j].astype(int)
        if len(np.unique(y)) < 2:
            clfs.append(DummyClassifier(strategy="most_frequent").fit(gamma, y))
            continue
        cw = "balanced" if (class_weight == "balanced-rare" and y.mean() < 0.15) else None
        base = LogisticRegression(max_iter=1000, C=0.1, class_weight=cw, solver="lbfgs")
        clfs.append(CalibratedClassifierCV(base, method="sigmoid", cv=3).fit(gamma, y))
    return clfs


def predict_proba(clfs, gamma, constant_label_fix=True):
    """
    P(behaviour active | gamma) for every behaviour. When a behaviour is
    constant in the training folds a DummyClassifier is used; the original
    thesis code then returned 0 even if the constant was 1 (e.g. Standing,
    always active in the training folds of Exp2/Exp3). constant_label_fix=True
    returns the constant instead; False reproduces the thesis behaviour.
    """
    cols = []
    for c in clfs:
        p = c.predict_proba(gamma)
        if p.shape[1] == 2:
            cols.append(p[:, 1])
        else:
            const = float(c.classes_[0]) if constant_label_fix else 0.0
            cols.append(np.full(len(gamma), const))
    return np.column_stack(cols)


def optimal_thresholds(Y, P):
    th = {}
    for j, b in enumerate(BEH_COLS):
        if Y[:, j].sum() == 0:
            th[b] = 0.5
            continue
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.05, 0.95, 0.05):
            f1 = f1_score(Y[:, j].astype(int), (P[:, j] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_t, best_f1 = float(t), f1
        th[b] = best_t
    return th


def run_loocv(Xz, Y, lengths, k, cfg):
    ends = np.cumsum(lengths)
    starts = ends - np.asarray(lengths)
    folds = []
    for i in range(len(lengths)):
        tr = [j for j in range(len(lengths)) if j != i]
        Xtr = np.vstack([Xz[starts[j]:ends[j]] for j in tr])
        Ytr = np.vstack([Y[starts[j]:ends[j]] for j in tr])
        Ltr = [lengths[j] for j in tr]
        Xte, Yte = Xz[starts[i]:ends[i]], Y[starts[i]:ends[i]]

        t0 = time.perf_counter()
        model, _ = fit_gmmhmm(k, Xtr, Ltr, cfg, cfg.loocv_n_iter, cfg.loocv_restarts)
        g_tr, g_te = model.predict_proba(Xtr, Ltr), model.predict_proba(Xte, [lengths[i]])
        clfs = fit_behaviour_classifiers(g_tr, Ytr, cfg.class_weight)
        P_tr = predict_proba(clfs, g_tr, cfg.constant_label_fix)
        P_te = predict_proba(clfs, g_te, cfg.constant_label_fix)
        folds.append(dict(seq_id=i, Y_test=Yte, P_test=P_te,
                          thresholds=optimal_thresholds(Ytr, P_tr)))
        print(f"    k={k} fold {i + 1}/{len(lengths)} done ({time.perf_counter() - t0:.0f}s)")
    return folds


# ─────────────────────────── main ───────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), help="Thesis experiment settings.")
    ap.add_argument("--input_dir", default=ROOT / "data/07_registered")
    ap.add_argument("--sequences", default=ROOT / "config/sequences.csv")
    ap.add_argument("--exclude", nargs="*", help="Animals to leave out (e.g. Mack Maisie).")
    ap.add_argument("--output_dir")
    ap.add_argument("--scaling", choices=["global", "per-sequence"])
    ap.add_argument("--class_weight", choices=["none", "balanced-rare"],
                    help="'balanced-rare': class_weight='balanced' for behaviours with prevalence < 15%%.")
    ap.add_argument("--constant_label_fix", type=lambda v: v.lower() in ("1", "true", "yes"),
                    metavar="{true,false}",
                    help="Predict the constant value (instead of 0) for behaviours that are "
                         "constant in the training folds. Presets use false (thesis behaviour).")
    ap.add_argument("--loocv_k", type=int, nargs="+", help="Candidate k for the LOOCV selection.")
    ap.add_argument("--final_k", type=int, nargs="+", help="Candidate k for the final (BIC) model.")
    ap.add_argument("--n_mix", type=int, default=2)
    ap.add_argument("--min_covar", type=float, default=0.1)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--loocv_n_iter", type=int, default=50)
    ap.add_argument("--loocv_restarts", type=int, default=1)
    ap.add_argument("--final_n_iter", type=int, default=200)
    ap.add_argument("--final_restarts", type=int, default=5)
    ap.add_argument("--smooth_window", type=int, default=7, help="Median filter (frames).")
    ap.add_argument("--skip_final", action="store_true", help="Skip the final BIC model.")
    ap.add_argument("--save_frame_csv", action="store_true")
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()

    base = dict(DEFAULTS, **(PRESETS[args.preset] if args.preset else {}))
    for key, val in base.items():
        if getattr(args, key) is None:
            setattr(args, key, val)
    if args.output_dir is None:
        args.output_dir = ROOT / "results" / (args.preset or "run")
    return args


def main():
    cfg = parse_args()
    t_all = time.perf_counter()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    seqs = pd.read_csv(cfg.sequences)
    seqs = seqs[~seqs["animal"].isin(cfg.exclude)]
    files = [Path(cfg.input_dir) / f"{s}_interp_clipped_frames_scaled_reg.csv" for s in seqs["sequence"]]
    print(f"Sequences: {', '.join(seqs['animal'])}")
    print(f"Scaling={cfg.scaling}  class_weight={cfg.class_weight}  "
          f"constant_label_fix={cfg.constant_label_fix}  "
          f"loocv_k={cfg.loocv_k}  final_k={cfg.final_k}")

    big, feats = build_features(load_sequences(files))
    lengths = big.groupby("seq_id").size().tolist()
    X = big[feats].to_numpy(float)
    Y = big[BEH_COLS].to_numpy(float)
    Xz = standardise(X, lengths, cfg.scaling)
    print(f"{len(big)} frames, {len(feats)} features, lengths={lengths}")

    # ── 1. choice of k by LOOCV macro AUC ──
    k_auc, k_folds = {}, {}
    for k in cfg.loocv_k:
        print(f"\nLOOCV with k={k}")
        folds = run_loocv(Xz, Y, lengths, k, cfg)
        Yall = np.vstack([f["Y_test"] for f in folds])
        Pall = np.vstack([f["P_test"] for f in folds])
        k_auc[k] = float(np.nanmean([safe_auc(Yall[:, j], Pall[:, j]) for j in range(Y.shape[1])]))
        k_folds[k] = folds
        print(f"  k={k}: LOOCV macro AUC = {k_auc[k]:.4f}")
    best_k = max(k_auc, key=k_auc.get)
    folds = k_folds[best_k]          # deterministic: identical to re-running with best_k
    (out / "k_selection.json").write_text(json.dumps(
        {"auc_by_k": k_auc, "best_k": best_k}, indent=2))

    # ── 2. out-of-fold predictions ──
    ends = np.cumsum(lengths)
    starts = ends - np.asarray(lengths)
    P = np.zeros_like(Y)
    for f in folds:
        P[starts[f["seq_id"]]:ends[f["seq_id"]]] = f["P_test"]
    P_smooth = np.column_stack([median_filter(P[:, j], size=cfg.smooth_window)
                                for j in range(P.shape[1])])
    np.savez_compressed(out / "loocv_predictions.npz", Y=Y, P=P, P_smooth=P_smooth,
                        seq_id=big["seq_id"].to_numpy(), frame=big["Frame"].to_numpy(),
                        behaviours=np.array(BEH_COLS),
                        thresholds=np.array([[f["thresholds"][b] for b in BEH_COLS] for f in folds]))
    if cfg.save_frame_csv:
        frame_df = big.copy()
        for j, b in enumerate(BEH_COLS):
            frame_df[f"proba_{b}"] = P_smooth[:, j]
        frame_df.to_csv(out / "frames_with_probabilities.csv", index=False)

    # ── 3. final model on all sequences, k by BIC ──
    final_k = None
    if not cfg.skip_final:
        print("\nFinal model on all sequences (BIC)")
        n_jobs = multiprocessing.cpu_count() if cfg.n_jobs == -1 else cfg.n_jobs
        fits = Parallel(n_jobs=min(n_jobs, len(cfg.final_k)))(
            delayed(fit_gmmhmm)(k, Xz, lengths, cfg, cfg.final_n_iter, cfg.final_restarts)
            for k in cfg.final_k)
        bic = {k: float(bic_gmmhmm(m, Xz, lengths)) for k, (m, _) in zip(cfg.final_k, fits) if m}
        final_k = min(bic, key=bic.get)
        (out / "final_model_bic.json").write_text(json.dumps({"bic_by_k": bic, "best_k": final_k}, indent=2))
        plt.figure(figsize=(8, 4))
        plt.plot(list(bic), list(bic.values()), "o-", color="steelblue", linewidth=2)
        plt.axvline(final_k, color="tomato", linestyle="--", label=f"Best k={final_k}")
        plt.xlabel("Number of HMM states (k)")
        plt.ylabel("BIC (lower = better)")
        plt.title("GMM-HMM state selection via BIC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "BIC_state_selection.png", dpi=200)
        plt.close()
        if cfg.save_model:
            final = dict(zip(cfg.final_k, fits))[final_k][0]
            gamma = final.predict_proba(Xz, lengths)
            clfs = fit_behaviour_classifiers(gamma, Y, cfg.class_weight)
            with open(out / "final_model.pkl", "wb") as fh:
                pickle.dump(dict(hmm=final, classifiers=clfs, features=feats,
                                 behaviours=BEH_COLS, scaling=cfg.scaling), fh)

    # ── 4. evaluation ──
    glo = eval_metrics(Y, P_smooth)
    seq_rows, beh_rows = [], []
    for f in folds:
        i = f["seq_id"]
        name = big.loc[starts[i], "seq_name"]
        m = eval_metrics(f["Y_test"], f["P_test"], f["thresholds"])
        seq_rows.append(dict(seq_id=i, seq_name=name, frames=len(f["Y_test"]),
                             mean_nll=m["mean_nll"], macro_auc=m["macro_auc"],
                             f1_micro=m["f1_micro"], f1_macro=m["f1_macro"]))
        for j, b in enumerate(BEH_COLS):
            beh_rows.append(dict(seq_id=i, seq_name=name, behaviour=b,
                                 prevalence=float(f["Y_test"][:, j].mean()),
                                 mean_pred=float(f["P_test"][:, j].mean()),
                                 mean_nll=float(m["mean_nll_per_beh"][j]), auc=m["aucs"][j],
                                 threshold=f["thresholds"][b]))
        save_nll_heatmap(f["Y_test"], f["P_test"], f"NLL (LOOCV) - {name}",
                         out / f"{Path(name).stem}_NLL_heatmap.png")
    save_nll_heatmap(Y, P_smooth, "NLL (LOOCV out-of-fold) - all videos", out / "ALL_NLL_heatmap_LOOCV.png")
    seq_df, beh_df = pd.DataFrame(seq_rows), pd.DataFrame(beh_rows)
    seq_df.to_csv(out / "eval_by_sequence.csv", index=False)
    beh_df.to_csv(out / "eval_per_behaviour.csv", index=False)

    lines = [
        "GMM-HMM multi-label evaluation - leave-one-sequence-out CV",
        f"  Sequences    : {', '.join(seqs['animal'])}",
        f"  Total frames : {len(big)}",
        f"  Features     : {len(feats)} (21 spatial, 21 first and 21 second differences)",
        f"  Scaling      : {cfg.scaling} | class weight: {cfg.class_weight} | N_mix: {cfg.n_mix}"
        f" | constant-label fix: {cfg.constant_label_fix}",
        f"  k (LOOCV AUC): {best_k}   candidates {cfg.loocv_k}",
        f"  k (BIC final): {final_k}   candidates {cfg.final_k}",
        "",
        "Global metrics (smoothed out-of-fold probabilities, threshold 0.5)",
        f"  mean NLL  : {glo['mean_nll']:.4f}",
        f"  macro AUC : {glo['macro_auc']:.4f}",
        f"  F1 micro  : {glo['f1_micro']:.4f}",
        f"  F1 macro  : {glo['f1_macro']:.4f}",
        "",
        "Per sequence (raw out-of-fold probabilities, train-fold thresholds)",
        seq_df.to_string(index=False),
        "",
        "Per sequence: 5 hardest behaviours (by mean NLL)",
    ]
    for i, sub in beh_df.groupby("seq_id"):
        lines += [f"\n=== {sub['seq_name'].iloc[0]} ===",
                  sub.sort_values("mean_nll", ascending=False).head(5)
                  [["behaviour", "prevalence", "mean_pred", "mean_nll", "auc"]].to_string(index=False)]
    lines.append(f"\nTotal runtime (s): {time.perf_counter() - t_all:.0f}")
    (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:16]))
    print(f"\nResults in {out}")


if __name__ == "__main__":
    main()
