#!/usr/bin/env python3
"""Sensitivity of model selection to capture sessions.

The configuration was selected on development folds drawn at the image level, so
images of one capture session could sit in both the training and the validation
part of a fold. This script repeats the selection with session-grouped validation:
each split holds out one capture session for every species-and-grade pair (24 of the
72 sessions) as validation and trains on the rest, so no session spans the two. Ten
such splits are drawn from SPLIT_SEED. Only the 3,512 development images are used;
the locked test set is never read. Hyperparameters are those already selected for
each configuration and are not re-searched.

Every run uses training seed 0, the seed of the out-of-fold refit, so the variation
across splits comes from the split alone. Training follows the refit protocol: at
most 100 epochs, early stopping on validation QWK with patience 30.

The selection rule of the paper is then applied to the means over splits: a
configuration whose mean validation QWK lies within one standard deviation (across
splits) of the leader's is tied with it; among tied configurations single pooling is
preferred, and validation ECE breaks the remaining tie. Per-split rankings are
reported alongside.

    python session_selection.py --n-splits 10
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
from datetime import datetime
from pathlib import Path

import numpy as np
import timm
import torch

import core
import metrics as M
import session_split_run as S
import train as T

CONFIGS = ["focalcorn_gap_ours", "focalcorn_gap_std", "ce_gapgmp", "focalce_gap"]
SEED = 0
OUT = Path("outputs/session_selection")
WEIGHTS = Path("outputs/weights/session_selection")


def session_splits(dev, sess, n_splits, seed):
    """Validation = one randomly chosen development session per species-and-grade pair."""
    rng = np.random.default_rng(seed)
    pairs = sorted({s.split("|")[0] for s in sess[dev]})
    splits, seen = [], set()
    while len(splits) < n_splits:
        val_sess = frozenset(rng.choice(sorted({s for s in sess[dev] if s.startswith(p + "|")}))
                             for p in pairs)
        if val_sess in seen:
            continue
        seen.add(val_sess)
        is_val = np.array([s in val_sess for s in sess[dev]])
        splits.append((dev[~is_val], dev[is_val], sorted(val_sess)))
    return splits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-splits", type=int, default=10)
    args = ap.parse_args()

    dev, test, paths, labels, _ = core.make_split()
    sess = S.sessions_of(paths)
    splits = session_splits(dev, sess, args.n_splits, core.SPLIT_SEED)
    rows = {r["config"]: r for r in json.loads(Path("outputs/selection.json").read_text())}

    stamp = {"script": "session_selection.py", "host": socket.gethostname(),
             "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
             "cuda": torch.version.cuda, "timm": timm.__version__,
             "python": platform.python_version(), "train_seed": SEED,
             "split_seed": core.SPLIT_SEED, "args": vars(args),
             "timestamp": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}
    print("=" * 78)
    print("MODEL SELECTION UNDER SESSION-GROUPED VALIDATION")
    print(json.dumps(stamp))
    print(f"development images {len(dev)} | sessions {len(set(sess[dev]))} | "
          f"test images {len(test)} (not read)")
    for k, (tr, va, vs) in enumerate(splits, 1):
        assert not set(sess[tr]) & set(sess[va]), f"split {k}: a session spans the split"
        assert not set(tr) & set(test) and not set(va) & set(test)
        print(f"  split {k:>2}: train {len(tr)} / val {len(va)} images, {len(vs)} val sessions, "
              f"val classes {np.bincount(labels[va], minlength=3).tolist()}")
    print("=" * 78)

    OUT.mkdir(parents=True, exist_ok=True)
    folds = [(tr, va) for tr, va, _ in splits]
    res = {c: [] for c in CONFIGS}
    for c in CONFIGS:
        cfg, hp = core.CONFIGS[c], rows[c]["hp"]
        for k in range(1, len(splits) + 1):
            f = OUT / f"{c}_split{k}.npz"
            if not f.exists():          # resumable: a finished run is never retrained
                ck = T.ckpt_path(str(WEIGHTS), c, SEED, k)
                r = T.train_fold(cfg, hp, k, SEED, paths, labels, folds, epochs=100,
                                 patience=30, save_to=str(ck), collect_val_preds=True)
                np.savez(f, y_true=r["y_true"], y_pred=r["y_pred"], probs=r["probs"],
                         conf=r["conf"], best_qwk=r["best_qwk"], best_epoch=r["best_epoch"])
            d = dict(np.load(f))
            assert np.array_equal(d["y_true"], labels[splits[k - 1][1]])
            m = M.all_metrics(d["y_true"], d["y_pred"], d["probs"], d["conf"])
            res[c].append({"split": k, "qwk": float(d["best_qwk"]), "accuracy": m["accuracy"],
                           "ece": m["ece"], "best_epoch": int(d["best_epoch"])})
            print(f"[{c} | split {k}] qwk {float(d['best_qwk']):.4f}  acc "
                  f"{m['accuracy']*100:.2f}%  ece {m['ece']:.4f}  epoch {int(d['best_epoch'])}")

    # aggregate rule, as in the paper
    agg = {c: {k: M.agg([r[k] for r in res[c]]) for k in ("qwk", "accuracy", "ece")}
           for c in CONFIGS}
    leader = max(CONFIGS, key=lambda c: agg[c]["qwk"][0])
    band = agg[leader]["qwk"][0] - agg[leader]["qwk"][1]
    tied = [c for c in CONFIGS if agg[c]["qwk"][0] >= band]
    single = [c for c in tied if not core.CONFIGS[c]["dual_pool"]] or tied
    selected = min(single, key=lambda c: agg[c]["ece"][0])

    # per-split rankings
    n = len(splits)
    per = {c: {"qwk_first": 0, "ece_first": 0, "acc_first": 0, "qwk_rank": []} for c in CONFIGS}
    for k in range(n):
        for key, name in (("qwk", "qwk_first"), ("ece", "ece_first"), ("accuracy", "acc_first")):
            vals = {c: res[c][k][key] for c in CONFIGS}
            best = (min if key == "ece" else max)(vals, key=vals.get)
            per[best][name] += 1
        order = sorted(CONFIGS, key=lambda c: -res[c][k]["qwk"])
        for i, c in enumerate(order, 1):
            per[c]["qwk_rank"].append(i)

    print("\nconfiguration          val QWK            val acc (%)       val ECE         "
          "QWK-first ECE-first acc-first  mean QWK rank")
    for c in CONFIGS:
        q, a, e = agg[c]["qwk"], agg[c]["accuracy"], agg[c]["ece"]
        print(f"{c:<22} {q[0]:.4f} +/- {q[1]:.4f}  {a[0]*100:5.2f} +/- {a[1]*100:4.2f}  "
              f"{e[0]:.4f} +/- {e[1]:.4f}  {per[c]['qwk_first']:>5}/{n} {per[c]['ece_first']:>5}/{n} "
              f"{per[c]['acc_first']:>5}/{n}   {np.mean(per[c]['qwk_rank']):.1f}")
    print(f"\nrule: leader {leader} ({agg[leader]['qwk'][0]:.4f} +/- {agg[leader]['qwk'][1]:.4f}); "
          f"tied {tied}; single pooling {single}; selected {selected}")

    Path("results").mkdir(exist_ok=True)
    f = Path(f"results/session_selection_{datetime.now():%Y-%m-%d}.json")
    f.write_text(json.dumps({"stamp": stamp, "splits": [vs for _, _, vs in splits],
                             "runs": res, "aggregate": agg, "per_split": per,
                             "rule": {"leader": leader, "tied": tied, "single_pooling": single,
                                      "selected": selected}}, indent=2, default=float))
    print(f"\nverified: {len(CONFIGS)} configurations x {n} session-grouped splits, no session "
          f"spans a split, test set untouched; written {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
