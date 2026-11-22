# Reproduction guide

Run from the repository root with the dataset under `./data/` (see the README)
and the environment from `requirements.txt` installed.

## 0. Verify the pipeline

```bash
python tests/test_pipeline.py
python tests/test_pruning.py
python tests/test_reporting.py
```
`test_pipeline.py` confirms the invariants that make the results meaningful: the
backbone normalisation is applied once, the standard CORN loss agrees with the
reference implementation, one prediction rule drives every metric, the split is
identical across training seeds, and no cross-validation fold touches the test
set. It exits non-zero if any check fails.

## 1. Per-configuration search (development set only)

```bash
python src/phases.py search --n-trials 12 --folds 3
```
Runs an independent Optuna study for every configuration in `core.CONFIGS`,
maximising mean validation QWK over the first three development folds. Writes
`outputs/hpo/best_<config>.json`.

## 2. Refit and collect out-of-fold predictions

```bash
python src/phases.py refit
```
Retrains each configuration at its selected hyperparameters across all five
development folds and stores the out-of-fold validation predictions in
`outputs/oof/<config>.npz`, with `outputs/selection.json`. These predictions back
every development-set table and figure.

## 3. Final model

```bash
python src/phases.py train --config focalcorn_gap_ours
```
Trains a single model on the entire development set for the median best epoch
observed during cross-validation, once per training seed. Omitting `--config`
selects the highest mean validation QWK; passing it records a documented
tie-break in `outputs/final_config.json`.

## 4. Evaluation

```bash
python src/phases.py evaluate
```
Evaluates each final model once on the held-out test set, writing per-seed
predictions to `outputs/test/` and metrics to `outputs/test_results.json`.
Predictions are not pooled across seeds.

The reported results use ten training seeds. Steps 3 and 4 cover the first five
(42, 123, 999, 7, 2024); the other five were fixed in advance and are trained and
evaluated with the same code path:

```bash
for c in focalcorn_gap_ours ce_gap ce_gapgmp focalcorn_gapgmp_std; do
  python src/alt_branch_run.py --config $c --seeds 0 1 2 3 4
done
python src/baseline_hoang.py --seeds 0 1 2 3 4
```
Their results are written to separate `*_seeds0-1-2-3-4.json` files, so the
first five are never overwritten.

## 5. Comparisons and robustness

```bash
python src/baseline_hoang.py                 # prior-best pipeline, same split
python src/baseline_hoang.py --grouped       # ... and on the session-disjoint split
python src/session_split_run.py              # proposed model, session-disjoint split
python src/session_split_run.py --config ce_gap          # CE + GAP on the same split
python src/session_split_run.py --oof                    # session-grouped OOF, for temperature
python src/session_split_run.py --config ce_gap --oof
python src/occlusion_test.py                 # eye / surroundings masking, inference only
python src/session_selection.py --n-splits 10   # selection under session-grouped validation
python src/volume_control.py --n 2988        # training-set-size control
python src/alt_branch_run.py --config focalcorn_gapgmp_std
python src/check_leakage.py                  # near-duplicate analysis
```

## 6. Statistics

```bash
python src/session_bootstrap.py --n-boot 10000   # every comparison, all splits
python src/calibration_analysis.py               # decoding, temperature, triage, binning
```
`session_bootstrap.py` resamples capture sessions (species, grade, capture date)
rather than images, together with each system's training runs, and applies Holm
correction within each family of comparisons. The reported difference is the one
observed on the full test set; the interval is the bootstrap percentile interval.
Temperature scaling follows each model's own decision rule: softmax logits are
divided by T, and the two CORN logits are divided by T and re-decoded.

A session is one species-and-grade folder on one capture date, read from the file
name the camera assigned (`IMG_YYYYMMDD_hhmmss`); see `sessions_of` in
`src/session_split_run.py`. The same script reports three sensitivity analyses
(two-stage, stratified by species and grade, and a paired session permutation
test), intervals for an absolute 0.95 confidence gate, and, on the
session-disjoint split, temperatures fitted on session-grouped out-of-fold
predictions.

## 7. Tables and figures

```bash
python src/report.py all --seeds 42 123 999 7 2024 0 1 2 3 4
python src/make_dataset_grid.py
python src/make_eigencam.py --seed 7              # representative (median-kappa) seed
cd figures && python make_prior_works.py && python make_focal_weight.py
```
Outputs land in `results/` and `figures/`.

## Notes

- Seeds are fixed and cuDNN runs deterministically.
- A fold counts as complete only once a `.done` marker is written beside its
  checkpoint, so an interrupted run never reuses a partially trained model.
- A single mid-range GPU (16 GB) is sufficient. The search stage dominates the
  cost; each final training stage is a few GPU-hours.
- `split/` already contains the exact partitions used, so the reported numbers can
  be reproduced without re-deriving the split. `src/export_split.py` regenerates
  them.
