#!/usr/bin/env python3
"""Matched re-run of the prior best FFE pipeline (Hoang et al., 2026).

Their method, our protocol. The pipeline is reproduced from the authors' public
repository (Fish_Eyes_Freshness_Classification): fine-tune Swin-Tiny with
cross-entropy, extract 768-d Stage-4 pooled features, standardise them, select
the top 10% by LightGBM split importance, and classify with a Random Forest or
Extra Trees chosen by randomised search.

What differs from their published setup is only the evaluation protocol, so the
comparison is matched: the species-stratified split from ``core.make_split``
(SPLIT_SEED, 24 strata) instead of their unstratified 64/16/20 split, a single
model trained on the whole development set instead of a fold ensemble, and the
same training seeds as the proposed model, reported as mean +/- SD on the same
locked test set.

Their hyperparameters are left exactly as published and are not re-tuned.

    python baseline_hoang.py --seeds 42 123 999 7 2024
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import socket
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

import core
import metrics as M

# ── Their published hyperparameters (unchanged) ─────────────────────────────
EPOCHS = 100
PATIENCE = 20            # early stopping on validation accuracy
BATCH_SIZE = 64
LR = 1e-4
WD = 1e-2
N_LGBM_FEATS = 77        # top 10% of 768
N_ITER_SEARCH = 50       # RandomizedSearchCV n_iter, per classifier family
LGBM_SEED = 42           # their fixed random_state for the selector

OUT = Path("outputs/baseline")
WEIGHTS = Path("outputs/weights/baseline")

TRAIN_TF = T.Compose([
    T.Resize((core.IMG_SIZE, core.IMG_SIZE), interpolation=T.InterpolationMode.LANCZOS),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=30),
    T.ColorJitter(brightness=0.2),
    T.ToTensor(),
    T.Normalize(core.MEAN, core.STD),
])
EVAL_TF = T.Compose([
    T.Resize((core.IMG_SIZE, core.IMG_SIZE), interpolation=T.InterpolationMode.LANCZOS),
    T.ToTensor(),
    T.Normalize(core.MEAN, core.STD),
])


class _Data(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = paths, labels, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), int(self.labels[i])


class HoangSwin(nn.Module):
    """Swin-Tiny with a linear head; the backbone doubles as the feature extractor."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(core.MODEL_NAME, pretrained=pretrained,
                                          num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, core.NUM_CLASSES)

    def forward(self, x):
        return self.head(self.backbone(x))

    def features(self, x):
        return self.backbone(x)


def _loader(paths, labels, tf, shuffle=False):
    return DataLoader(_Data(paths, labels, tf), batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=core.NUM_WORKERS, pin_memory=True)


def fine_tune(paths, labels, tr_idx, va_idx, seed: int, save_to: Path) -> HoangSwin:
    """Their fine-tuning recipe: CE, AdamW(1e-4, 1e-2), no scheduler, early stop on val acc."""
    core.set_seed(seed)
    model = HoangSwin(pretrained=True).to(core.DEVICE)
    if save_to.exists() and (save_to.parent / (save_to.name + ".done")).exists():
        model.load_state_dict(torch.load(save_to, map_location=core.DEVICE))
        print(f"    backbone: reusing {save_to.name}")
        return model

    tl = _loader(paths[tr_idx], labels[tr_idx], TRAIN_TF, shuffle=True)
    vl = _loader(paths[va_idx], labels[va_idx], EVAL_TF)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    best_acc, best_wts, bad, t0 = -1.0, None, 0, time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        for x, y in tl:
            x, y = x.to(core.DEVICE), y.to(core.DEVICE)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(core.DEVICE), y.to(core.DEVICE)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.numel()
        acc = correct / total
        if acc > best_acc:
            best_acc, best_wts, bad = acc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if ep % 10 == 0 or ep == 1:
            print(f"    ep {ep:3d}/{EPOCHS}  val_acc={acc:.4f}  best={best_acc:.4f}  "
                  f"pat={bad}/{PATIENCE}  {time.time() - t0:.0f}s")
        if bad >= PATIENCE:
            print(f"    early stop at epoch {ep} (best val acc {best_acc:.4f})")
            break

    model.load_state_dict(best_wts)
    save_to.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_wts, save_to)
    (save_to.parent / (save_to.name + ".done")).write_text(json.dumps(
        {"best_val_acc": best_acc, "epochs_run": ep, "seed": seed,
         "minutes": round((time.time() - t0) / 60, 1)}))
    return model


def extract(model: HoangSwin, paths, labels) -> np.ndarray:
    """768-d Stage-4 pooled features from the fine-tuned backbone."""
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in _loader(paths, labels, EVAL_TF):
            out.append(model.features(x.to(core.DEVICE)).cpu().numpy())
    return np.vstack(out)


def select_features(x_dev: np.ndarray, y_dev: np.ndarray) -> np.ndarray:
    """Top-10% feature indices by LightGBM split importance, as in their selector."""
    lgbm = LGBMClassifier(objective="multiclass", random_state=LGBM_SEED,
                          force_col_wise=True, verbose=-1)
    lgbm.fit(x_dev, y_dev)
    return np.sort(np.argsort(lgbm.feature_importances_)[::-1][:N_LGBM_FEATS])


def search_classifier(kind: str, x, y, seed: int):
    """Their randomised search over the published grids, scored by CV accuracy."""
    rng = np.random.RandomState(seed)
    if kind == "rf":
        est = RandomForestClassifier(random_state=seed, n_jobs=-1)
        grid = {"n_estimators": rng.randint(10, 501, size=300),
                "max_depth": rng.randint(2, 50, size=25),
                "criterion": ["gini", "entropy"]}
    else:
        est = ExtraTreesClassifier(random_state=seed, n_jobs=-1)
        grid = {"n_estimators": rng.randint(10, 501, size=300),
                "max_depth": [None, 10, 15, 20, 25, 30, 35, 40, 50],
                "criterion": ["gini", "entropy"],
                "min_samples_split": [2, 3, 5, 7, 9, 11],
                "min_samples_leaf": [1, 3, 5, 8, 9, 11],
                "bootstrap": [True, False],
                "max_leaf_nodes": [None, 2, 3, 4, 5, 6, 8, 9, 10, 11]}
    s = RandomizedSearchCV(est, grid, n_iter=N_ITER_SEARCH, cv=5, scoring="accuracy",
                           random_state=seed, n_jobs=-1, refit=True)
    s.fit(x, y)
    return s.best_estimator_, float(s.best_score_), s.best_params_


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999, 7, 2024])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--grouped", action="store_true",
                    help="hold out whole capture sessions instead of individual images")
    args = ap.parse_args()

    dev_idx, test_idx, paths, labels, strata = core.make_split()
    out_dir, weights_dir = OUT, WEIGHTS
    if args.grouped:
        import session_split_run as S
        dev_idx, test_idx, _, _ = S.grouped_split(paths, labels)
        out_dir = Path("outputs/baseline_session")
        weights_dir = Path("outputs/weights/baseline_session")
    folds = core.make_folds(dev_idx, strata)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("MATCHED BASELINE — Hoang et al. pipeline under our protocol")
    print(f"host      : {socket.gethostname()}  |  {torch.cuda.get_device_name(0)}")
    print(f"versions  : torch {torch.__version__}  timm {timm.__version__}  "
          f"python {platform.python_version()}")
    print(f"split     : SPLIT_SEED={core.SPLIT_SEED}, 24 strata, "
          f"dev={len(dev_idx)} test={len(test_idx)}")
    print(f"recipe    : CE, AdamW(lr={LR}, wd={WD}), no scheduler, {EPOCHS} ep, "
          f"bs={BATCH_SIZE}, early stop on val acc (patience {PATIENCE})")
    print(f"classifier: StandardScaler -> LGBM top-{N_LGBM_FEATS} -> "
          f"RandomizedSearchCV(n_iter={N_ITER_SEARCH}) over RF and ET")
    print(f"split     : {'session-disjoint' if args.grouped else 'image-level'}")
    print(f"seeds     : {args.seeds}")
    print(f"invoked   : baseline_hoang.py --seeds {' '.join(map(str, args.seeds))}")
    print(f"timestamp : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 74)

    # The test set must be byte-identical to the one our own model was scored on.
    ours = Path("outputs/session_split/seed42.npz") if args.grouped \
           else Path("outputs/test/seed42.npz")
    y_test = labels[test_idx]
    if ours.exists():
        ref = np.load(ours)["y_true"]
        assert np.array_equal(ref, y_test), "test labels differ from our evaluation"
        print(f"verified: test set identical to ours (n={len(y_test)}, label sequence matches)")
    else:
        print(f"WARNING: {ours} absent; could not cross-check the test set")

    rows = []
    for seed in args.seeds:
        print(f"\n[seed {seed}] " + "-" * 58)
        # Their protocol selects the backbone on a validation split; we use fold 1
        # of the development folds, so nothing outside the development set is read.
        # make_folds already returns global indices.
        tr_idx, va_idx = folds[0]
        model = fine_tune(paths, labels, tr_idx, va_idx, seed,
                          weights_dir / f"seed{seed}" / "swin.pth")

        x_dev = extract(model, paths[dev_idx], labels[dev_idx])
        x_test = extract(model, paths[test_idx], y_test)
        y_dev = labels[dev_idx]

        scaler = StandardScaler().fit(x_dev)          # fitted on development data only
        x_dev_s, x_test_s = scaler.transform(x_dev), scaler.transform(x_test)

        keep = select_features(x_dev_s, y_dev)
        x_dev_s, x_test_s = x_dev_s[:, keep], x_test_s[:, keep]
        print(f"    features: 768 -> {len(keep)} after LightGBM selection")

        for kind in ("rf", "et"):
            clf, cv_acc, params = search_classifier(kind, x_dev_s, y_dev, seed)
            probs = clf.predict_proba(x_test_s).astype(np.float32)
            y_pred = probs.argmax(1).astype(np.int64)
            conf = probs[np.arange(len(y_pred)), y_pred]
            np.savez(out_dir / f"{kind}_seed{seed}.npz", y_true=y_test, y_pred=y_pred,
                     probs=probs, conf=conf)
            m = M.all_metrics(y_test, y_pred, probs, conf)
            rows.append({"classifier": kind.upper(), "seed": seed,
                         "cv_acc": cv_acc, "params": {k: str(v) for k, v in params.items()},
                         **m})
            print(f"    {kind.upper()}: dev CV acc={cv_acc:.4f} | test acc="
                  f"{m['accuracy'] * 100:.2f}%  qwk={m['qwk']:.4f}  ece={m['ece']:.4f}")

    # extra seeds get their own files so the original five-seed results are never overwritten
    default_seeds = ap.get_default("seeds")
    tag = "" if args.seeds == default_seeds else "_seeds" + "-".join(map(str, args.seeds))
    (out_dir / f"results{tag}.json").write_text(json.dumps(rows, indent=2))

    print("\n" + "=" * 74)
    print("mean +/- SD over seeds (no pooling)")
    print("=" * 74)
    summary = {}
    for kind in ("RF", "ET"):
        sub = [r for r in rows if r["classifier"] == kind]
        if not sub:
            continue
        summary[kind] = {}
        print(f"\n{kind} + LGBM-FS  (n={len(sub)} seeds)")
        for k in ["accuracy", "f1_macro", "mae", "qwk", "ece", "mce", "brier", "nll",
                  "nf_to_hf", "total_jumps"]:
            mu, sd = M.agg([r[k] for r in sub])
            summary[kind][k] = [mu, sd]
            sc = 100 if k in ("accuracy", "f1_macro") else 1
            print(f"  {k:<11} {mu * sc:.4f} +/- {sd * sc:.4f}")
    (out_dir / f"summary{tag}.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{len(args.seeds)} seeds x 2 classifiers evaluated on {len(y_test)} "
          f"held-out test images; written to {out_dir}/results{tag}.json and summary{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
