# Hyperparameter search records

The records of the per-configuration Optuna searches used in the paper.

| File | Contents |
|---|---|
| `best_<config>.json` | the selected hyperparameters, the fold scores and best epochs of the selected trial, and the search settings |
| `hpo_trials/<config>.csv` | every trial of that search: parameters, objective value, state (complete or pruned), timing |

The pipeline writes its own search results to `outputs/hpo/`, so these files are
never overwritten. To reuse the published selection instead of searching again,
copy the `best_<config>.json` files to `outputs/hpo/` before running Phase A2;
Phase A1 then skips every configuration that already has a result.
