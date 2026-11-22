#!/usr/bin/env python3
"""Train and evaluate the configuration the selection rule would pick without
the single-pooling preference.

The tie-break in the selection rule has two steps: configurations within one
fold standard deviation of the best validation kappa are treated as tied, the
simpler single-pooling head is preferred, and calibration decides among what
remains. Dropping the parsimony step would select the dual-pooling variant
instead. This runs that branch so the reader can see what it gives, rather than
having to take the rule on trust.

Writes to its own directory; the reported final model is untouched.

    python alt_branch_run.py --config focalcorn_gapgmp_std
"""

from __future__ import annotations

import argparse
import json
import platform
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

OUT_ROOT = Path("outputs/alt_branch")
WEIGHTS = Path("outputs/weights/alt")
SEEDS = [42, 123, 999, 7, 2024]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="focalcorn_gapgmp_std", choices=list(core.CONFIGS))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    dev, test, paths, labels, strata = core.make_split()
    rows = {r["config"]: r for r in json.loads(Path("outputs/selection.json").read_text())}
    r = rows[args.config]
    cfg, hp, epochs = core.CONFIGS[args.config], r["hp"], r["median_best_epoch"]

    OUT = OUT_ROOT / args.config
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print("ALTERNATIVE SELECTION BRANCH")
    print(f"host      : {socket.gethostname()}  |  {torch.cuda.get_device_name(0)}")
    print(f"torch     : {torch.__version__}  |  python {platform.python_version()}")
    print(f"config    : {args.config}  (dev QWK {r['fold_qwk_mean']:.4f}, "
          f"dev ECE {r['oof']['ece']:.4f}, {epochs} epochs)")
    print(f"split     : the same locked split, dev={len(dev)} test={len(test)}")
    print(f"invoked   : alt_branch_run.py --config {args.config}")
    print(f"timestamp : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 74)

    _, eval_tf = core.build_transforms(hp)
    tl = DataLoader(core.FishDataset(paths[test], labels[test], eval_tf),
                    batch_size=128, shuffle=False, num_workers=core.NUM_WORKERS)
    ordinal = cfg["head"] == "corn"

    out = []
    for s in args.seeds:
        ck = T.ckpt_path(str(WEIGHTS), args.config, s, None)
        if not T.is_complete(ck):
            t0 = time.time()
            T.train_final(cfg, hp, s, paths, labels, dev, epochs, save_to=str(ck),
                          verbose=False)
            print(f"  seed {s}: trained ({(time.time()-t0)/60:.1f} min)")
        model = core.build_model(cfg, hp, pretrained=False)
        model.load_state_dict(torch.load(ck, map_location=core.DEVICE))
        e = T.evaluate(model, tl, ordinal)
        np.savez(OUT / f"seed{s}.npz", y_true=e["y_true"], y_pred=e["y_pred"],
                 probs=e["probs"], conf=e["conf"])
        m = M.all_metrics(e["y_true"], e["y_pred"], e["probs"], e["conf"])
        out.append({"seed": s, **m})
        print(f"  seed {s}: acc={m['accuracy']*100:.2f}%  qwk={m['qwk']:.4f}  "
              f"ece={m['ece']:.4f}")

    # extra seeds get their own file so the original five-seed results are never overwritten
    tag = "" if args.seeds == SEEDS else "_seeds" + "-".join(map(str, args.seeds))
    (OUT / f"results{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"\nmean +/- SD over {len(args.seeds)} seeds")
    for k in ["accuracy", "f1_macro", "mae", "qwk", "ece", "brier", "nll",
              "nf_to_hf", "total_jumps"]:
        mu, sd = M.agg([x[k] for x in out])
        sc = 100 if k in ("accuracy", "f1_macro") else 1
        print(f"  {k:<12} {mu*sc:.4f} +/- {sd*sc:.4f}")
    print(f"\nwritten to {OUT}/results{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
