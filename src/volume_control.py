#!/usr/bin/env python3
"""Control for training-set size when reading the session-disjoint result.

Holding out whole capture sessions removes about a sixth of the development set
as well as the sessions themselves, so a drop measured against the full-data
model confounds two causes. This trains the selected configuration on a random
subset of the development set matched in size to the session-disjoint one, and
evaluates it on the ordinary image-level test set. Whatever it loses is
attributable to data volume; the remainder of the session-disjoint drop is not.

    python volume_control.py --n 2988 --seeds 42 123 999
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

OUT = Path("outputs/volume_control")
WEIGHTS = Path("outputs/weights/volume")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2988, help="development images to keep")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999])
    args = ap.parse_args()

    dev, test, paths, labels, strata = core.make_split()
    fc = json.loads(Path("outputs/final_config.json").read_text())
    cfg, hp, epochs = fc["cfg"], fc["hp"], fc["epochs"]

    # keep the class balance of the full development set
    rng = np.random.default_rng(core.SPLIT_SEED)
    keep = []
    for c in range(core.NUM_CLASSES):
        idx = dev[labels[dev] == c]
        k = int(round(len(idx) * args.n / len(dev)))
        keep.append(rng.choice(idx, size=k, replace=False))
    sub = np.sort(np.concatenate(keep))

    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print("TRAINING-SET SIZE CONTROL")
    print(f"host      : {socket.gethostname()}  |  {torch.cuda.get_device_name(0)}")
    print(f"torch     : {torch.__version__}  |  python {platform.python_version()}")
    print(f"config    : {fc['config']}  ({epochs} epochs, hyperparameters unchanged)")
    print(f"training  : {len(sub)} of {len(dev)} development images "
          f"({len(sub)/len(dev)*100:.1f}%), class balance preserved")
    print(f"evaluated : the ordinary image-level test set, n={len(test)}")
    print(f"invoked   : volume_control.py --n {args.n} --seeds "
          f"{' '.join(map(str, args.seeds))}")
    print(f"timestamp : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 74)

    _, eval_tf = core.build_transforms(hp)
    tl = DataLoader(core.FishDataset(paths[test], labels[test], eval_tf),
                    batch_size=128, shuffle=False, num_workers=core.NUM_WORKERS)
    ordinal = cfg["head"] == "corn"

    rows = []
    for s in args.seeds:
        ck = T.ckpt_path(str(WEIGHTS), fc["config"], s, None)
        if not T.is_complete(ck):
            t0 = time.time()
            T.train_final(cfg, hp, s, paths, labels, sub, epochs, save_to=str(ck),
                          verbose=False)
            print(f"  seed {s}: trained ({(time.time()-t0)/60:.1f} min)")
        model = core.build_model(cfg, hp, pretrained=False)
        model.load_state_dict(torch.load(ck, map_location=core.DEVICE))
        e = T.evaluate(model, tl, ordinal)
        np.savez(OUT / f"seed{s}.npz", y_true=e["y_true"], y_pred=e["y_pred"],
                 probs=e["probs"], conf=e["conf"])
        m = M.all_metrics(e["y_true"], e["y_pred"], e["probs"], e["conf"])
        rows.append({"seed": s, **m})
        print(f"  seed {s}: acc={m['accuracy']*100:.2f}%  qwk={m['qwk']:.4f}  "
              f"ece={m['ece']:.4f}")

    (OUT / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"\nmean +/- SD over {len(args.seeds)} seeds")
    for k in ["accuracy", "f1_macro", "qwk", "ece"]:
        mu, sd = M.agg([r[k] for r in rows])
        sc = 100 if k in ("accuracy", "f1_macro") else 1
        print(f"  {k:<10} {mu*sc:.4f} +/- {sd*sc:.4f}")
    print(f"\nwritten to {OUT}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
