#!/usr/bin/env python3
"""Campaign driver.

    python phases.py search --n-trials 12 --folds 3     # per-configuration search
    python phases.py refit                             # refit at best HP, save OOF predictions
    python phases.py train                              # final model on the full dev set, per seed
    python phases.py evaluate                              # evaluate final models on the test set

Phase A works only on the development set. The test set is read for the first
time in Phase D, after the configuration has been frozen.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import core
import metrics as M
import train as T

OUT = Path("outputs")
HPO_DIR = OUT / "hpo"
OOF_DIR = OUT / "oof"
TEST_DIR = OUT / "test"
WEIGHTS = "outputs/weights"
SEEDS = [42, 123, 999, 7, 2024]        # five training seeds
MAX_EPOCHS = 100
PATIENCE = 30
REFIT_SEED = 0                          # fixed seed for the A2 refit


def _load(name):
    dev, test, paths, labels, strata = core.make_split()
    return dev, test, paths, labels, strata, core.make_folds(dev, strata)


# ── Phase A1: per-configuration hyperparameter search ───────────────────────
def phase_a1(args):
    for i, name in enumerate(core.CONFIGS, 1):
        out = HPO_DIR / f"best_{name}.json"
        if out.exists() and not args.force:
            print(f"[{i}/{len(core.CONFIGS)}] {name}: already searched, skipping")
            continue
        print(f"\n[{i}/{len(core.CONFIGS)}] searching {name}")
        cmd = [sys.executable, str(Path(__file__).parent / "hpo.py"),
               "--config", name, "--n-trials", str(args.n_trials),
               "--folds", str(args.folds), "--out-dir", str(HPO_DIR)]
        subprocess.run(cmd, check=True)
    print("\nPhase A1 complete.")


# ── Phase A2: refit at the selected hyperparameters, collect OOF predictions ─
def phase_a2(args):
    dev, test, paths, labels, strata, folds = _load(None)
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, name in enumerate(core.CONFIGS, 1):
        cfgfile = HPO_DIR / f"best_{name}.json"
        if not cfgfile.exists():
            print(f"[{i}] {name}: no search result, skipping")
            continue
        blob = json.loads(cfgfile.read_text())
        hp, cfg = blob["best_hp"], core.CONFIGS[name]
        npz = OOF_DIR / f"{name}.npz"
        if npz.exists() and not args.force:
            print(f"[{i}/{len(core.CONFIGS)}] {name}: OOF present, skipping")
            continue

        print(f"\n[{i}/{len(core.CONFIGS)}] refitting {name} over {core.N_FOLDS} folds")
        yt = np.zeros(len(dev), dtype=np.int64)
        yp = np.zeros(len(dev), dtype=np.int64)
        pr = np.zeros((len(dev), core.NUM_CLASSES), dtype=np.float32)
        cf = np.zeros(len(dev), dtype=np.float32)
        pos = {g: k for k, g in enumerate(dev)}     # global index -> row in dev
        qwks, eps = [], []
        for k in range(1, core.N_FOLDS + 1):
            ck = T.ckpt_path(WEIGHTS + "/oof", name, REFIT_SEED, k)
            r = T.train_fold(cfg, hp, k, REFIT_SEED, paths, labels, folds,
                             epochs=MAX_EPOCHS, patience=PATIENCE,
                             save_to=str(ck), collect_val_preds=True)
            rowsi = [pos[g] for g in r["val_idx"]]
            yt[rowsi] = r["y_true"]; yp[rowsi] = r["y_pred"]
            pr[rowsi] = r["probs"]; cf[rowsi] = r["conf"]
            qwks.append(r["best_qwk"]); eps.append(r["best_epoch"])
            print(f"    fold {k}: val_qwk={r['best_qwk']:.4f} "
                  f"best_epoch={r['best_epoch']} ({r['minutes']:.1f} min)")

        np.savez(npz, y_true=yt, y_pred=yp, probs=pr, conf=cf,
                 fold_qwks=np.array(qwks), best_epochs=np.array(eps))
        m = M.all_metrics(yt, yp, pr, cf)
        mu, sd = M.agg(qwks)
        rows.append({"config": name, **cfg, "hp": hp,
                     "fold_qwk_mean": mu, "fold_qwk_sd": sd,
                     "median_best_epoch": int(np.median(eps)),
                     "best_epochs": eps, "oof": m})
        print(f"  -> mean fold QWK {mu:.4f} +/- {sd:.4f} | OOF acc {m['accuracy']*100:.2f}% "
              f"ECE {m['ece']:.4f} | median best epoch {int(np.median(eps))}")

    if rows:
        p = OUT / "selection.json"
        prev = json.loads(p.read_text()) if p.exists() else []
        by = {r["config"]: r for r in prev}
        by.update({r["config"]: r for r in rows})
        p.write_text(json.dumps(list(by.values()), indent=2))
        print(f"\nselection summary -> {p}")
    print("Phase A2 complete.")


def _winner(name=None):
    """The selected configuration, with the criterion that selected it.

    Defaults to the highest mean fold QWK. An explicit name records a documented
    tie-break instead: the configurations fall within one fold standard deviation
    of each other on QWK, so calibration decides. Returns the selected row, the
    criterion, and the QWK argmax, so the two can be compared in the output.
    """
    rows = json.loads((OUT / "selection.json").read_text())
    argmax = max(rows, key=lambda r: r["fold_qwk_mean"])
    if name is None:
        return argmax, "highest mean validation QWK", argmax["config"]
    sel = next(r for r in rows if r["config"] == name)
    crit = ("validation QWK differences within one fold SD treated as tied; "
            "tie broken by development-set ECE")
    return sel, crit, argmax["config"]


# ── Phase C: final single model on the full development set ─────────────────
def phase_c(args):
    dev, test, paths, labels, strata, folds = _load(None)
    w, criterion, qwk_argmax = _winner(args.config)
    cfg, hp = core.CONFIGS[w["config"]], w["hp"]
    epochs = args.epochs or w["median_best_epoch"]
    print(f"selected : {w['config']}  (dev QWK {w['fold_qwk_mean']:.4f} "
          f"+/- {w['fold_qwk_sd']:.4f}, dev ECE {w['oof']['ece']:.4f})")
    print(f"criterion: {criterion}")
    if w["config"] != qwk_argmax:
        print(f"note     : QWK argmax was {qwk_argmax}; this selection is a "
              f"recorded tie-break, not a QWK win")
    print(f"training one model on all {len(dev)} dev images for {epochs} epochs, "
          f"seeds {SEEDS}")
    (OUT).mkdir(parents=True, exist_ok=True)
    (OUT / "final_config.json").write_text(json.dumps(
        {"config": w["config"], "cfg": cfg, "hp": hp, "epochs": epochs,
         "seeds": SEEDS, "dev_qwk_mean": w["fold_qwk_mean"],
         "dev_qwk_sd": w["fold_qwk_sd"], "dev_ece": w["oof"]["ece"],
         "selection_criterion": criterion, "qwk_argmax": qwk_argmax}, indent=2))
    for s in SEEDS:
        ck = T.ckpt_path(WEIGHTS + "/final", w["config"], s, None)
        if T.is_complete(ck) and not args.force:
            print(f"  seed {s}: complete, skipping")
            continue
        t0 = time.time()
        T.train_final(cfg, hp, s, paths, labels, dev, epochs, save_to=str(ck),
                      verbose=False)
        print(f"  seed {s}: done ({(time.time()-t0)/60:.1f} min) -> {ck}")
    print("Phase C complete.")


# ── Phase D: single evaluation on the held-out test set ─────────────────────
def phase_d(args):
    from torch.utils.data import DataLoader
    dev, test, paths, labels, strata, folds = _load(None)
    fc = json.loads((OUT / "final_config.json").read_text())
    cfg, hp, name = fc["cfg"], fc["hp"], fc["config"]
    ordinal = cfg["head"] == "corn"
    _, eval_tf = core.build_transforms(hp)
    tl = DataLoader(core.FishDataset(paths[test], labels[test], eval_tf),
                    batch_size=128, shuffle=False, num_workers=core.NUM_WORKERS)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"evaluating {name} on {len(test)} held-out test images")
    rows = []
    for s in fc["seeds"]:
        ck = T.ckpt_path(WEIGHTS + "/final", name, s, None)
        model = core.build_model(cfg, hp, pretrained=False)
        model.load_state_dict(torch.load(ck, map_location=core.DEVICE))
        r = T.evaluate(model, tl, ordinal)
        np.savez(TEST_DIR / f"seed{s}.npz", y_true=r["y_true"], y_pred=r["y_pred"],
                 probs=r["probs"], conf=r["conf"])
        m = M.all_metrics(r["y_true"], r["y_pred"], r["probs"], r["conf"])
        rows.append({"seed": s, **m})
        print(f"  seed {s}: acc={m['accuracy']*100:.2f}%  qwk={m['qwk']:.4f}  "
              f"ece={m['ece']:.4f}")
    (OUT / "test_results.json").write_text(json.dumps(rows, indent=2))
    print("\nmean +/- SD over seeds (no pooling):")
    for k in ["accuracy", "f1_macro", "mae", "qwk", "ece", "mce", "brier", "nll"]:
        mu, sd = M.agg([r[k] for r in rows])
        scale = 100 if k in ("accuracy", "f1_macro") else 1
        print(f"  {k:<10} {mu*scale:.4f} +/- {sd*scale:.4f}")
    print("Phase D complete.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="phase", required=True)
    p1 = sub.add_parser("search", aliases=["a1"])
    p1.add_argument("--n-trials", type=int, default=12)
    p1.add_argument("--folds", type=int, default=3); p1.add_argument("--force", action="store_true")
    p2 = sub.add_parser("refit", aliases=["a2"]); p2.add_argument("--force", action="store_true")
    p3 = sub.add_parser("train", aliases=["c"]); p3.add_argument("--epochs", type=int, default=None)
    p3.add_argument("--config", default=None, choices=list(core.CONFIGS),
                    help="select this configuration instead of the QWK argmax")
    p3.add_argument("--force", action="store_true")
    p4 = sub.add_parser("evaluate", aliases=["d"])
    a = ap.parse_args()
    {"search": phase_a1, "a1": phase_a1, "refit": phase_a2, "a2": phase_a2,
     "train": phase_c, "c": phase_c, "evaluate": phase_d, "d": phase_d}[a.phase](a)


if __name__ == "__main__":
    main()
