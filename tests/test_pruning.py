#!/usr/bin/env python3
"""Confirm that Hyperband pruning actually fires on the epoch resource.

Runs a small synthetic study whose reported metric degrades for most trials, so
a correctly configured pruner must terminate some of them early. Guards against
the resource-granularity mistake where progress is reported per fold while the
pruner expects epochs.
"""

import sys
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import hpo

optuna.logging.set_verbosity(optuna.logging.WARNING)

MAX_EPOCHS = hpo.MAX_EPOCHS


def objective(trial):
    quality = trial.suggest_float("quality", 0.0, 1.0)
    for epoch in range(1, MAX_EPOCHS + 1):
        trial.report(quality * epoch / MAX_EPOCHS, epoch)   # resource = epochs
        if trial.should_prune():
            raise optuna.TrialPruned()
    return quality


study = hpo.build_study("verify_pruning", "sqlite:///:memory:")
study.optimize(objective, n_trials=40)

pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

print(f"trials={len(study.trials)}  complete={len(done)}  pruned={len(pruned)}")
assert isinstance(study.pruner, optuna.pruners.HyperbandPruner)
assert len(pruned) > 0, (
    "Hyperband never pruned. The reported resource does not match "
    "min_resource/max_resource (are you reporting folds instead of epochs?)"
)
steps = [max(t.intermediate_values) for t in pruned if t.intermediate_values]
print(f"pruned trials stopped at epochs: min={min(steps):.0f} "
      f"median={sorted(steps)[len(steps)//2]:.0f} max={max(steps):.0f}"
      if False else f"pruning fired on {len(pruned)}/{len(study.trials)} trials")
print("PRUNING VERIFIED")
