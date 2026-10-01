# bovine-behavior-inference

Code and data of the master's thesis

> **Inférence multi-label des comportements bovins à partir de détections visuelles brutes :
> normalisation géométrique et modélisation probabiliste temporelle dans le cadre du projet WELL-E**
> Lina Aggoune — Maîtrise en informatique, Université du Québec à Montréal (UQAM).

The pipeline starts from the raw output of a box detector (body, head and snout boxes of the
cows, one CSV per video) and from the Observer XT behaviour annotations, and ends with a
GMM-HMM + calibrated logistic regression model that predicts 12 behaviours frame by frame,
evaluated by leave-one-sequence-out cross-validation.

Every step is a **stand-alone script** in `scripts/`, numbered in execution order. The output
folder of a step is the input folder of the next one, and the data folders carry the same
number as the script that writes them.

---

## Pipeline

```
 data/raw/annotations/*.xlsx              data/raw/detections/*-WH-csv.csv
 (Observer XT events)                     (detector output, one row per box)
          │                                          │
  01_observer_events_to_binary.py           03_rearrange_boxes.py
          │                                          │
 data/01_annotations_binary/              data/03_arranged/
          │                                          │
  02_annotation_correction_tool.py (GUI,    04_assign_boxes_roi.py
   optional, edits 01 in place)                      │
          │                               data/04_roi/
          └──────────────┬───────────────────────────┘
                 05_build_trajectories.py
                         │
                data/05_trajectories/       (pixel boxes + 17 behaviour labels)
                         │
                06_normalize_to_stall.py    (stall-relative coordinates)
                         │
                data/06_scaled/
                         │
                07_register_bboxes.py       (affine registration to a canonical stall)
                         │
                data/07_registered/
                         │
                08_gmmhmm_loocv.py          (features, GMM-HMM, classifier, LOOCV)
                         │
                results/<experiment>/
                         │
                09_plot_behaviour_metrics.py

 Descriptive statistics:   10_annotation_statistics.py  -> results/annotation_statistics/
                           11_behaviour_matrices.py     -> results/behaviour_matrices/
 Quality control:          scripts/qc/                  -> results/registration_qc/, videos
```

| Step | Script | Thesis | Input | Output |
|---|---|---|---|---|
| 01 | `01_observer_events_to_binary.py` | 3.2.2 | `data/raw/annotations/*.xlsx` | `data/01_annotations_binary/<seq>_behaviors.csv` |
| 02 | `02_annotation_correction_tool.py` (GUI) | 3.2.3, 4.2.3 | video + step-01 CSV | step-01 CSV, corrected |
| 03 | `03_rearrange_boxes.py` | 3.4 | `data/raw/detections/<seq>-WH-csv.csv` | `data/03_arranged/<seq>_arranged_boxes.csv` |
| 04 | `04_assign_boxes_roi.py` | 3.3.1 | `data/03_arranged/` + ROI (`config/sequences.csv`) | `data/04_roi/<seq>_roi_1.csv` |
| 05 | `05_build_trajectories.py` | 3.5 | `data/04_roi/` + `data/01_annotations_binary/` | `data/05_trajectories/<seq>_interp_clipped_frames.csv` |
| 06 | `06_normalize_to_stall.py` | 3.3.2, 4.3.1 | `data/05_trajectories/` + stall corners | `data/06_scaled/<seq>_..._scaled.csv` |
| 07 | `07_register_bboxes.py` (+ `stall_affine_search.py`) | 3.6.1, 4.3.2–4.3.3 | `data/06_scaled/` | `data/07_registered/<seq>_..._scaled_reg.csv` |
| 08 | `08_gmmhmm_loocv.py` | 3.6.2–3.8, 4.4–4.5, ch. 5 | `data/07_registered/` | `results/<experiment>/` |
| 09 | `09_plot_behaviour_metrics.py` | ch. 5 | `results/<experiment>/eval_per_behaviour.csv` | `results/<experiment>/behaviour_plots/` |
| 10 | `10_annotation_statistics.py` | Table 3.3 | `data/05_trajectories/` | `results/annotation_statistics/` |
| 11 | `11_behaviour_matrices.py` | 4.2.2, Figure 4.1 | `data/06_scaled/` | `results/behaviour_matrices/` |

Each script has a detailed docstring (`python scripts/<script>.py --help`) describing the
method, its inputs/outputs and the corresponding thesis section.

### What each step does

1. **01 – Observer events → frame labels.** Converts the Observer XT state events
   (`State start` / `State stop`, session `Original`) of each animal into one 0/1 column per
   behaviour (17 behaviours), at 30 fps. An event active in `[t_start, t_end)` covers frames
   `round(30·t_start)+1 … round(30·t_end)` (1-based).
2. **02 – Manual correction (optional, interactive).** Tkinter tool to inspect the labels
   next to the video and fix them frame by frame or on a frame range. The corrections made
   for the thesis are already contained in `..._Events_fixed.xlsx`, so this step is not
   needed to reproduce the results.
3. **03 – Spatial association of the parts.** The detector outputs independent body, head and
   snout boxes. For each frame, a greedy algorithm groups them into anatomical entities
   (body + head + snout) by maximum overlap, entities sorted by score.
4. **04 – Region of interest.** Keeps only the entities inside the stall of the studied animal
   (body centre in the ROI; head/snout centres in the ROI or inside the body box).
5. **05 – Trajectory.** One entity per frame (the first/best one in the ROI); frames without a
   complete body + head + snout triplet are dropped ("clipped"); the frame-level labels of
   step 01 are joined on the frame number.
6. **06 – Stall normalisation.** Box coordinates are expressed relative to the stall
   (centre and size computed from its 4 corners, LT/LB/RB/RT), giving `Ln, Tn, Wn, Hn`,
   `AR`, `logAR`. The normalised stall corners ("floating stall") are the input of step 07.
7. **07 – Affine registration.** A 7-parameter affine transform (translation, rotation, scale,
   shear) mapping the floating stall onto the canonical stall `[±0.8, ±1]` is found by an
   exhaustive grid search with local-minimum detection (`stall_affine_search.py`, numba),
   then applied to every box.
8. **08 – Model and evaluation.** 63 features (21 spatial/relative descriptors + first and
   second differences), standardisation, GMM-HMM (`hmmlearn`, diagonal covariances,
   N_mix = 2), posterior state probabilities as input of one calibrated logistic regression
   per behaviour (12 behaviours), leave-one-sequence-out CV, choice of k by LOOCV macro AUC,
   final model by BIC, metrics NLL / macro AUC / F1 micro / F1 macro.

---

## Repository layout

```
config/
  sequences.csv            the 5 sequences: animal, video file, stall corners, ROI
data/
  raw/detections/          detector output (5 CSV)
  raw/annotations/         Observer XT export (events, corrected)
  01_annotations_binary/   frame-level labels                     (step 01)
  03_arranged/             grouped body/head/snout entities         (step 03)
  05_trajectories/         trajectories used in the thesis          (step 05)
    filtered/              legacy Butterworth-filtered variant (not used, see below)
  06_scaled/               stall-normalised boxes                   (step 06)
  07_registered/           registered boxes = model input           (step 07)
  videos/                  NOT in the repository (see data/videos/README.md)
scripts/
  01_ … 11_*.py            pipeline steps
  stall_affine_search.py   grid-search module used by step 07 and the QC scripts
  qc/                      quality-control visualisations
  optional/                code that is not part of the thesis results
results/
  thesis_runs/             model outputs reported in the thesis (see below)
  annotation_statistics/   Table 3.3
  behaviour_matrices/      Figure 4.1 and co-occurrence matrices
  registration_qc/         registration reports (+ thesis_figures/)
```

`data/04_roi/` is not shipped (it is an intermediate file identical to the ROI subset of
`data/03_arranged/`); it is recreated by step 04.

---

## Installation

Python ≥ 3.10.

```bash
git clone https://github.com/bioinfoUQAM/bovine-behavior-inference.git
cd bovine-behavior-inference
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The results were produced with `hmmlearn 0.3.3`; other versions may give slightly different
numbers.

### Videos

The five videos (≈ 1.1 GB each) exceed GitHub's file-size limit and are not included. They
are only needed for the interactive steps (02, `--interactive` options of 04 and 06) and the
QC scripts that draw on the image. Place them in `data/videos/` with the names given in
`config/sequences.csv` (`video_file` column) — see `data/videos/README.md`.

---

## Usage

Every script runs on all sequences with its default paths, so the full pipeline is:

```bash
python scripts/01_observer_events_to_binary.py
# python scripts/02_annotation_correction_tool.py          # optional, interactive
python scripts/03_rearrange_boxes.py
python scripts/04_assign_boxes_roi.py
python scripts/05_build_trajectories.py --output_dir data/05_trajectories_rebuilt   # see note
python scripts/06_normalize_to_stall.py
python scripts/07_register_bboxes.py
python scripts/08_gmmhmm_loocv.py --preset exp1
python scripts/08_gmmhmm_loocv.py --preset exp2
python scripts/08_gmmhmm_loocv.py --preset exp3
python scripts/09_plot_behaviour_metrics.py --eval_csv results/exp1/eval_per_behaviour.csv
python scripts/10_annotation_statistics.py
python scripts/11_behaviour_matrices.py
```

Because all intermediate data are versioned, any step can be run alone (e.g. only step 08 on
`data/07_registered/`). Most scripts accept `--sequence <animal or sequence name>` and
`--input_dir / --output_dir` options.

Running time (2 CPU cores): steps 01–07 take a few minutes; step 08 takes about 5 min per
LOOCV fold and value of k (Exp1 with 5 values of k ≈ 2 h, Exp2/Exp3 ≈ 20–30 min, plus the
final BIC model).

### Experiments (step 08)

| Preset | Sequences | Thesis | Shipped output |
|---|---|---|---|
| `exp1` | Haagendaz, Mack, Babelle, Maisie, Versace | Experiment 1 (5.4.1) | `results/thesis_runs/exp1_all_sequences/` |
| `exp2` | without Mack | Experiment 2 (5.4.2) | `results/thesis_runs/exp2_without_mack/` |
| `exp3` | without Mack and Maisie | Experiment 3 (5.4.3) | `results/thesis_runs/exp3_without_mack_maisie/` |

`results/thesis_runs/` also contains `supplementary_raw_aspect_ratio/` (the experiment with
the raw aspect ratio added to the features, section 4.6) and `exp1_with_exp2_settings/`
(the five sequences run with the Exp2/Exp3 settings, see the notes below).
Each run folder holds `report.txt`, `eval_by_sequence.csv`, `eval_per_behaviour.csv`,
`k_selection.json`, the NLL heat maps, the BIC curve and the out-of-fold predictions
(`loocv_folds.npz`: `Y_test_<i>`, `P_test_<i>`, `thresholds`).

### Quality control (`scripts/qc/`)

```bash
python scripts/qc/registration_error_report.py --all             # works without videos
python scripts/qc/visualize_registration_warp.py --animal Maisie  # needs the video
python scripts/qc/draw_bboxes_on_video.py --csv data/05_trajectories/<seq>_interp_clipped_frames.csv \
       --video data/videos/<seq>.mp4 --output results/qc_videos/<seq>_annotated.mp4
```

---

## Reproducibility notes

What was verified when the repository was assembled:

* **Step 01** rebuilds the labels contained in `data/05_trajectories` exactly
  (0 differing values on the 5 sequences).
* **Steps 06 and 07** rebuild `data/06_scaled` and `data/07_registered` exactly (max.
  difference 2·10⁻¹⁶) from `data/05_trajectories` and the stall corners of
  `config/sequences.csv`.
* **Step 03** gives the same entities as the original script (faster implementation).
* **Steps 04–05.** The hand-drawn ROIs of the thesis were not saved; the ROIs in
  `config/sequences.csv` were recovered from the shipped trajectories. With them, 99.2–99.97 %
  of the frames are identical (same frames kept, same boxes). The trajectories of the thesis
  are therefore kept in `data/05_trajectories/`, and step 05 does not overwrite existing files
  unless `--overwrite` is given. The `ddt …` columns of the trajectories are raw pixel
  derivatives kept for inspection only; the model recomputes its own derivatives.
* **Step 08.** With `--preset exp1/exp2/exp3`, the script reproduces the out-of-fold
  predictions of the thesis runs (full Exp1 run at k = 14, full Exp3 run, fold 0 of Exp2: max.
  difference < 3·10⁻⁷; NLL and F1 identical to 6 decimals; a few per-behaviour AUCs move by
  up to 0.007 because many predicted probabilities are tied and such tiny numerical
  differences, e.g. across library versions, change the tie order).

Points worth knowing when reading the thesis results:

1. **Exp1 and Exp2/Exp3 were run with different settings.** Exp1 used a *global*
   standardisation of the features and no class re-weighting; Exp2 and Exp3 used a
   *per-sequence* standardisation (thesis 4.4.2), `class_weight="balanced"` for behaviours
   with a prevalence < 15 %, and k fixed to 14. The presets encode these settings. Run on the
   five sequences, the Exp2/Exp3 settings give a macro AUC of 0.351
   (`exp1_with_exp2_settings`) instead of 0.469, so part of the Exp1 → Exp2 difference comes
   from the settings, not only from excluding Mack.
2. **Behaviours constant in the training folds.** When a behaviour has a single value in the
   training sequences, the original code used a constant classifier that always predicted 0,
   even when the constant value was 1. In Exp2 (test fold Maisie) and Exp3, *Standing* is
   always 1 in the training folds, so it is predicted 0 everywhere (thesis 5.4.2: "le modèle
   prédit Standing = 0.000"), which dominates the NLL of these runs. The presets keep this
   behaviour to reproduce the thesis; without a preset, the default
   (`--constant_label_fix true`) predicts the constant value instead.
3. **Hyper-parameters.** LOOCV models use 1 initialisation and 50 EM iterations; the final
   (BIC) model uses 5 initialisations and 200 iterations. Exp1 searched k in {8, …, 16} by
   LOOCV and {10, …, 18} by BIC.
4. **Global vs per-sequence metrics.** Global metrics are computed on the median-smoothed
   (window 7) out-of-fold probabilities with a 0.5 threshold; per-sequence metrics use the raw
   probabilities and the F1-optimal thresholds of the training folds. Global macro AUC values
   below 0.5 come from pooling sequences whose probabilities are on different scales; the
   per-sequence AUCs are above 0.5.
5. **Registration.** With the grid resolution used (Nr = 7) the best transform is the same
   for the five sequences — a horizontal scaling sx = 0.833, all other parameters at their
   identity value (`data/07_registered/registration_params.csv`).
6. **Annotation statistics.** The original statistics script treated every column after
   `Defecation` as a behaviour (including box features on the scaled files); step 10 only
   counts the 17 behaviours and reproduces Table 3.3 (134 occurrences, 1.25 h).
7. **Frame indexing.** Frames are 1-based everywhere (frame 1 = first image of the video).
8. **Filtered trajectories.** `data/05_trajectories/filtered/` and
   `scripts/optional/interpolate_filter.py` (interpolation + Butterworth low-pass filter)
   belong to an earlier version of the pipeline and are not used for the thesis results.

---

## Citation

If you use this code or data, please cite the thesis:

```
Aggoune, L. Inférence multi-label des comportements bovins à partir de détections visuelles
brutes : normalisation géométrique et modélisation probabiliste temporelle dans le cadre du
projet WELL-E. Mémoire de maîtrise en informatique, Université du Québec à Montréal.
```
