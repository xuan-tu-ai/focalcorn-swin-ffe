# Ordinal Fish-Eye Freshness Assessment

Code for freshness grading of fish eyes on the public **Freshness of the Fish Eyes
(FFE)** benchmark, framed as an **ordinal** classification problem. A Swin-Tiny
backbone is paired with global average pooling and a **conditional ordinal
regression (CORN)** head trained under a focal-weighted ordinal loss.

## Method

- **Backbone:** Swin Transformer (Tiny), ImageNet-1k pretrained (via `timm`).
- **Head:** CORN ordinal head — `K-1 = 2` conditional logits with rank-consistent
  cumulative-product decoding.
- **Loss:** cross-entropy, focal cross-entropy, CORN, or focal CORN. Both the
  standard CORN formulation and a threshold-averaged variant are implemented and
  compared; on this benchmark they are equivalent.
- **Pooling:** global average pooling; concatenating a max-pooled vector is
  evaluated as an alternative.

## Experimental protocol

Selection is separated from evaluation:

1. **One fixed split.** An 80/20 species-stratified split (24 species × class
   strata) drawn once from a constant `SPLIT_SEED`, independent of the training
   seed, so the held-out test set is identical for every run.
2. **Development folds.** Species-stratified five-fold cross-validation on the
   80% development set, used for both architecture selection and the
   hyperparameter search.
3. **Selection.** Each candidate configuration receives its own Optuna study
   (TPE sampler, Hyperband pruner) maximising mean validation QWK, so the
   comparison is not biased by settings tuned for one variant.
4. **Freeze.** The winning configuration, its hyperparameters, and its epoch
   count are fixed.
5. **Final model.** A single model is trained on the entire development set,
   repeated across ten training seeds (42, 123, 999, 7, 2024 and 0–4). No
   ensembling.
6. **Evaluation.** Each final model is evaluated once on the held-out test set;
   results are reported as mean ± standard deviation across seeds. Comparisons use
   a paired bootstrap that resamples capture sessions and training runs together,
   with Holm correction within each family of comparisons.

The test set is never used for hyperparameter tuning or model selection.

### Two partitions

The benchmark was built by photographing stored fish repeatedly over six days, so
each (species, grade) spans several photography sessions. An image-level split
therefore lets a model see the background and staging of sessions it is later
tested on. Both partitions are provided in [`split/`](split/) and both are
evaluated: the image-level split for comparability with prior work, and a
**session-disjoint** split that holds out whole sessions. Results differ
substantially between them.

## Repository layout

```
src/
  core.py                  model, losses, data split, prediction rule
  train.py                 fold training and final full-development training
  hpo.py                   per-configuration Optuna search
  phases.py                campaign driver (search / refit / train / evaluate)
  metrics.py               metric computation from saved predictions
  report.py                LaTeX tables and figures
  baseline_hoang.py        matched re-run of the prior-best pipeline
  session_split_run.py     capture-session-disjoint retraining and evaluation
  volume_control.py        training-set-size control for the session result
  alt_branch_run.py        the configuration the selection rule would pick
                           without its parsimony step
  session_bootstrap.py     paired bootstrap over capture sessions and seeds, Holm-corrected
  calibration_analysis.py  decoding convention, temperature scaling, triage, binning
  check_leakage.py         near-duplicate analysis across the split boundary
  occlusion_test.py        accuracy with the eye or its surroundings masked
  session_selection.py     model selection repeated with session-grouped validation
  export_split.py          writes the split out as index files
  make_eigencam.py         EigenCAM saliency figure
  make_dataset_grid.py     dataset sample grid
figures/                   generated figures, plus the stand-alone sources:
  make_prior_works.py      accuracy of published FFE methods against the matched re-run
  make_focal_weight.py     focal weighting curve
  F2_architecture.tex      architecture diagram (TikZ)
  make_architecture_input.py  input thumbnail for the architecture diagram
tests/
  test_pipeline.py         pipeline invariants (normalisation, losses, split, prediction)
  test_pruning.py          confirms Hyperband pruning is active
  test_reporting.py        dry run of every table and figure on synthetic predictions
split/                     published index files for both partitions
configs/                   hyperparameter search records of the paper
results/                   generated tables
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Install a build of PyTorch matching your CUDA version — see <https://pytorch.org>.
Development used PyTorch 2.x, `timm` 1.0.27, Python 3.12.

## Data

The FFE dataset is not redistributed here. Download it from Mendeley Data
(Prasetyo et al., 2022; `doi:10.17632/xzyx7pbr3w.1`) and arrange the images under
`./data/` as one folder per (species, freshness) class, named
`"<Species> - <Freshness>"`:

```
data/
  Chanos Chanos - Highly Fresh/
  Chanos Chanos - Fresh/
  Chanos Chanos - Not Fresh/
  ...
```
There are **24 folders** (8 species × 3 freshness classes) and **4,390 images**.

## Trained models

The ten final models of the paper (the proposed configuration, one per training
seed) are attached to the [v1.0 release](https://github.com/xuan-tu-ai/focalcorn-swin-ffe/releases/tag/v1.0)
as `focalcorn_swin_seed<N>.pth`, with a `SHA256SUMS` file. Each is a PyTorch
state dict:

```python
import torch
from src import core

model = core.build_model({"dual_pool": False, "head": "corn"}, {"drop_path": 0.0},
                         pretrained=False)
model.load_state_dict(torch.load("focalcorn_swin_seed7.pth", map_location=core.DEVICE))
model.eval()
labels, probs, conf = core.predict(model(images.to(core.DEVICE)).cpu(), ordinal=True)
```

`images` is a batch normalised as in `core.build_transforms` (evaluation branch).
On the 878 images of `split/test.txt` the ten models reach 90.01% mean accuracy.

## Verification

Run the checks before any training:

```bash
python tests/test_pipeline.py     # invariants; exits non-zero on failure
python tests/test_pruning.py      # confirms the pruner is active
python tests/test_reporting.py    # builds every table and figure on synthetic data
```

## Running the pipeline

See [REPRODUCE.md](REPRODUCE.md).

## License

Released under the MIT License — see [LICENSE](LICENSE).
