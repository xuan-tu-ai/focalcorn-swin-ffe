#!/usr/bin/env python3
"""Per-configuration hyperparameter search on the development folds.

Every configuration in ``core.CONFIGS`` receives its own Optuna study, so the
architecture comparison is not biased by a configuration tuned for one variant.
The objective is the mean validation QWK over the first ``--folds`` development
folds. The held-out test set is never touched.

    python hpo.py --config focalcorn_gap_ours --n-trials 20 --folds 3
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import torch

import core
import train as T

SEARCH_SEED = 0        # governs the sampler and fold-training seed during search
MAX_EPOCHS = 100
PATIENCE = 30


def suggest_hp(trial: optuna.Trial) -> dict:
    return {
        "lr":            trial.suggest_float("lr", 3e-5, 3e-4, log=True),
        "weight_decay":  trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "drop_path":     trial.suggest_float("drop_path", 0.0, 0.30),
        "gamma":         trial.suggest_float("gamma", 0.5, 4.0),
        "pct_start":     trial.suggest_float("pct_start", 0.03, 0.30),
        "batch_size":    trial.suggest_categorical("batch_size", [32, 64, 96]),
        "aff_degrees":   trial.suggest_int("aff_degrees", 0, 40),
        "aff_translate": trial.suggest_float("aff_translate", 0.0, 0.15),
        "rerase_p":      trial.suggest_float("rerase_p", 0.0, 0.70),
    }


def build_study(name: str, storage: str) -> optuna.Study:
    """TPE sampler with Hyperband pruning, matching the reported protocol."""
    return optuna.create_study(
        study_name=name, storage=storage, direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=SEARCH_SEED),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=20, max_resource=MAX_EPOCHS, reduction_factor=3),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(core.CONFIGS))
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--folds", type=int, default=3, help="use folds 1..N for the objective")
    ap.add_argument("--out-dir", default="outputs/hpo")
    args = ap.parse_args()

    cfg = core.CONFIGS[args.config]
    dev_idx, test_idx, paths, labels, strata = core.make_split()
    folds = core.make_folds(dev_idx, strata)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print(f"HPO  config={args.config}  {cfg}")
    print(f"objective : mean val QWK over folds 1..{args.folds} (species-stratified)")
    print(f"sampler   : TPE(multivariate)   pruner: Hyperband(20,{MAX_EPOCHS},3)")
    print(f"trials    : {args.n_trials}   dev={len(dev_idx)}  test={len(test_idx)} (untouched)")
    print("=" * 68)

    def objective(trial: optuna.Trial) -> float:
        hp = suggest_hp(trial)
        qwks, epochs = [], []
        for k in range(1, args.folds + 1):
            # Hyperband's resource is the epoch count, so pruning is driven from
            # inside the first fold; surviving trials then run the remaining folds.
            r = T.train_fold(cfg, hp, k, SEARCH_SEED, paths, labels, folds,
                             epochs=MAX_EPOCHS, patience=PATIENCE,
                             trial=(trial if k == 1 else None))
            qwks.append(r["best_qwk"]); epochs.append(r["best_epoch"])
        trial.set_user_attr("fold_qwks", qwks)
        trial.set_user_attr("best_epochs", epochs)
        return float(np.mean(qwks))

    storage = f"sqlite:///{args.out_dir}/study_{args.config}.db"
    study = build_study(f"hpo_{args.config}", storage)
    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best_epochs = study.best_trial.user_attrs.get("best_epochs", [])
    out = {
        "config_name": args.config,
        "config": cfg,
        "best_hp": {**core.DEFAULT_HP, **study.best_params},
        "best_mean_val_qwk": float(study.best_value),
        "fold_qwks": study.best_trial.user_attrs.get("fold_qwks", []),
        "best_epochs": best_epochs,
        "median_best_epoch": int(np.median(best_epochs)) if best_epochs else None,
        "search": {
            "objective": f"mean val QWK over folds 1..{args.folds}",
            "sampler": "TPE(multivariate)",
            "pruner": f"Hyperband(20,{MAX_EPOCHS},3)",
            "n_trials": args.n_trials, "n_complete": len(done),
            "n_pruned": len(study.trials) - len(done),
            "split_seed": core.SPLIT_SEED, "stratification": "species x class (24 strata)",
            "n_folds_total": core.N_FOLDS, "folds_used": args.folds,
        },
        "elapsed_gpu_hours": round((time.time() - t0) / 3600, 2),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "host": socket.gethostname(),
    }
    p = Path(args.out_dir) / f"best_{args.config}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nbest mean val QWK {study.best_value:.4f}  ({len(done)} complete, "
          f"{len(study.trials)-len(done)} pruned, {out['elapsed_gpu_hours']}h)")
    print(f"median best epoch: {out['median_best_epoch']}")
    print(f"saved -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
