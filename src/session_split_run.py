#!/usr/bin/env python3
"""Robustness check under a capture-session-disjoint split.

The images of one (species, grade) were taken across two or four photography
sessions, so a split drawn at the image level lets a model see the background,
lighting, and staging of a session it is later tested on. This script rebuilds
the split so that whole sessions fall on one side or the other, retrains the
selected configuration on it, and evaluates on the held-out sessions.

The gap between these numbers and the ones from the image-level split measures
how much of the reported accuracy depends on session-level context.

    python session_split_run.py --seeds 42 123 999 7 2024
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import socket
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import core
import metrics as M
import train as T

OUT = Path("outputs/session_split")
WEIGHTS = Path("outputs/weights/session")
SPLIT_SEED = core.SPLIT_SEED


def sessions_of(paths) -> np.ndarray:
    """One session per (species, grade, capture date)."""
    out = []
    for p in paths:
        folder = Path(p).parent.name
        m = re.search(r"(20\d{6})", Path(p).name)
        out.append(f"{folder}|{m.group(1) if m else 'unknown'}")
    return np.array(out)


def grouped_split(paths, labels):
    """Hold out one capture session per (species, grade); no session spans the split."""
    sess = sessions_of(paths)
    rng = np.random.default_rng(SPLIT_SEED)
    test_sessions = set()
    groups = sorted({s.split("|")[0] for s in sess})
    for g in groups:
        dates = sorted({s for s in sess if s.startswith(g + "|")})
        test_sessions.add(dates[int(rng.integers(len(dates)))])
    is_test = np.array([s in test_sessions for s in sess])
    return np.where(~is_test)[0], np.where(is_test)[0], sess, sorted(test_sessions)


def session_folds(dev, labels, sess, n_folds=core.N_FOLDS):
    """Folds over the development sessions: no session spans train and validation."""
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SPLIT_SEED)
    return [(dev[tr], dev[va]) for tr, va in sgkf.split(dev, labels[dev], groups=sess[dev])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999, 7, 2024])
    ap.add_argument("--config", default=None, choices=list(core.CONFIGS),
                    help="configuration to run (default: the selected final model)")
    ap.add_argument("--oof", action="store_true",
                    help="out-of-fold predictions over session-grouped development folds, "
                         "for fitting a temperature on this split")
    args = ap.parse_args()

    _, _, paths, labels, strata = core.make_split()
    dev, test, sess, held = grouped_split(paths, labels)
    fc = json.loads(Path("outputs/final_config.json").read_text())
    name = args.config or fc["config"]
    if name == fc["config"]:
        cfg, hp, epochs = fc["cfg"], fc["hp"], fc["epochs"]
        out = OUT                     # the selected model keeps its original location
    else:
        rows = {r["config"]: r for r in
                json.loads(Path("outputs/selection.json").read_text())}
        cfg, hp, epochs = core.CONFIGS[name], rows[name]["hp"], rows[name]["median_best_epoch"]
        out = OUT / name

    out.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print("SESSION-DISJOINT ROBUSTNESS CHECK" + (" -- session-grouped OOF" if args.oof else ""))
    print(f"host      : {socket.gethostname()}  |  {torch.cuda.get_device_name(0)}")
    print(f"torch     : {torch.__version__}  |  python {platform.python_version()}")
    print(f"config    : {name}  ({epochs} epochs, hyperparameters unchanged)")
    print(f"sessions  : {len(set(sess))} total, {len(held)} held out")
    print(f"split     : dev={len(dev)}  test={len(test)}  "
          f"({len(test)/len(paths)*100:.1f}% held out)")
    print(f"overlap   : {len(set(sess[dev]) & set(sess[test]))} sessions on both sides")
    for c in range(core.NUM_CLASSES):
        print(f"    {core.CLASS_NAMES[c]:<14} dev={int((labels[dev]==c).sum()):>5}  "
              f"test={int((labels[test]==c).sum()):>4}")
    print(f"invoked   : session_split_run.py --config {name} --seeds "
          f"{' '.join(map(str, args.seeds))}{' --oof' if args.oof else ''}")
    print(f"timestamp : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 74)
    assert len(set(sess[dev]) & set(sess[test])) == 0, "a session spans the split"
    ordinal = cfg["head"] == "corn"

    if args.oof:
        # same refit protocol as phases.py refit (early stopping on validation QWK),
        # but the folds hold out whole development sessions
        folds = session_folds(dev, labels, sess)
        for k, (tr, va) in enumerate(folds, 1):
            assert not set(sess[tr]) & set(sess[va]), f"fold {k}: a session spans the fold"
        pos = {g: i for i, g in enumerate(dev)}
        yt = np.zeros(len(dev), np.int64); yp = np.zeros(len(dev), np.int64)
        pr = np.zeros((len(dev), core.NUM_CLASSES), np.float32); cf = np.zeros(len(dev), np.float32)
        seed = args.seeds[0]
        for k in range(1, len(folds) + 1):
            ck = T.ckpt_path(str(WEIGHTS / "oof"), name, seed, k)
            r = T.train_fold(cfg, hp, k, seed, paths, labels, folds, epochs=100, patience=30,
                             save_to=str(ck), collect_val_preds=True)
            rows_ = [pos[g] for g in r["val_idx"]]
            yt[rows_] = r["y_true"]; yp[rows_] = r["y_pred"]
            pr[rows_] = r["probs"]; cf[rows_] = r["conf"]
            print(f"  fold {k}: {len(folds[k-1][1])} val images, "
                  f"{len(set(sess[folds[k-1][1]]))} sessions, val QWK {r['best_qwk']:.4f}, "
                  f"best epoch {r['best_epoch']}")
        f = out / "oof.npz"
        np.savez(f, y_true=yt, y_pred=yp, probs=pr, conf=cf, dev_idx=dev)
        assert np.array_equal(yt, labels[dev])
        print(f"\nverified: {len(folds)} session-grouped folds, {len(dev)} OOF predictions -> {f}")
        return 0

    _, eval_tf = core.build_transforms(hp)
    tl = DataLoader(core.FishDataset(paths[test], labels[test], eval_tf),
                    batch_size=128, shuffle=False, num_workers=core.NUM_WORKERS)

    rows = []
    for s in args.seeds:
        ck = T.ckpt_path(str(WEIGHTS), name, s, None)
        if not T.is_complete(ck):
            t0 = time.time()
            T.train_final(cfg, hp, s, paths, labels, dev, epochs, save_to=str(ck),
                          verbose=False)
            print(f"  seed {s}: trained ({(time.time()-t0)/60:.1f} min)")
        model = core.build_model(cfg, hp, pretrained=False)
        model.load_state_dict(torch.load(ck, map_location=core.DEVICE))
        r = T.evaluate(model, tl, ordinal)
        np.savez(out / f"seed{s}.npz", y_true=r["y_true"], y_pred=r["y_pred"],
                 probs=r["probs"], conf=r["conf"])
        m = M.all_metrics(r["y_true"], r["y_pred"], r["probs"], r["conf"])
        rows.append({"seed": s, **m})
        print(f"  seed {s}: acc={m['accuracy']*100:.2f}%  qwk={m['qwk']:.4f}  "
              f"ece={m['ece']:.4f}")

    (out / "results.json").write_text(json.dumps(rows, indent=2))
    print("\nmean +/- SD over seeds (session-disjoint split)")
    for k in ["accuracy", "f1_macro", "mae", "qwk", "ece", "brier", "nll",
              "nf_to_hf", "total_jumps"]:
        mu, sd = M.agg([r[k] for r in rows])
        sc = 100 if k in ("accuracy", "f1_macro") else 1
        print(f"  {k:<12} {mu*sc:.4f} +/- {sd*sc:.4f}")
    print(f"\n{len(args.seeds)} seeds on {len(test)} held-out-session images -> {out}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
