#!/usr/bin/env python3
"""Occlusion check: how much of the decision rests on the eye, and how much on its surroundings.

The FFE images are cropped around the eye, so the eye sits near the centre of each
image. Every test image is evaluated under four inputs: unchanged; only a central disc
kept (radius 30, 40, or 50% of the image side) with everything outside replaced by the
ImageNet mean (zero after normalisation); and the converse, the central 30% disc masked
with the surroundings kept. A model that grades from the eye should lose little when the
surroundings are removed and much when the eye is removed.

Run for the final model and for CE + GAP, each at every training seed. Inference only.

    python occlusion_test.py
"""

from __future__ import annotations

import json
import platform
import socket
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import core
import metrics as M
import train as T

ORIG_SEEDS = [42, 123, 999, 7, 2024]
SEEDS = ORIG_SEEDS + [0, 1, 2, 3, 4]
W = Path("outputs/weights")
# (config, checkpoint root for the original seeds, root for the extra seeds)
SYSTEMS = {"proposed": ("focalcorn_gap_ours", W / "final", W / "alt"),
           "ce_gap": ("ce_gap", W / "alt", W / "alt")}
KEEP_RADII = [0.30, 0.40, 0.50]
MASK_RADIUS = 0.30


def disc(size: int, radius: float) -> torch.Tensor:
    """Boolean mask of a centred disc whose radius is a fraction of the image side."""
    c = (size - 1) / 2
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    return ((yy - c) ** 2 + (xx - c) ** 2) <= (radius * size) ** 2


@torch.no_grad()
def run(model, loader, ordinal, mask):
    L, P, C, Y = [], [], [], []
    for x, y in loader:
        if mask is not None:
            x = x * mask.to(x.dtype)              # zero = ImageNet mean after normalisation
        lab, probs, conf = core.predict(model(x.to(core.DEVICE)).cpu(), ordinal)
        L.append(lab); P.append(probs); C.append(conf); Y.append(y)
    return (torch.cat(Y).numpy(), torch.cat(L).numpy(), torch.cat(P).numpy(),
            torch.cat(C).numpy())


def main() -> int:
    core.set_seed(0)
    _, test, paths, labels, _ = core.make_split()
    rows = {r["config"]: r for r in json.loads(Path("outputs/selection.json").read_text())}
    conds = {"full": None}
    conds.update({f"keep_centre_{int(r*100)}": disc(224, r) for r in KEEP_RADII})
    conds[f"mask_centre_{int(MASK_RADIUS*100)}"] = ~disc(224, MASK_RADIUS)
    stamp = {"host": socket.gethostname(), "device": torch.cuda.get_device_name(0),
             "torch": torch.__version__, "python": platform.python_version(),
             "seeds": SEEDS, "conditions": list(conds),
             "timestamp": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}
    print("=" * 74); print("OCCLUSION CHECK"); print(json.dumps(stamp)); print("=" * 74)

    out = {"stamp": stamp, "results": {}}
    for sysname, (cfgname, root_orig, root_extra) in SYSTEMS.items():
        cfg, hp = core.CONFIGS[cfgname], rows[cfgname]["hp"]
        _, eval_tf = core.build_transforms(hp)
        tl = DataLoader(core.FishDataset(paths[test], labels[test], eval_tf),
                        batch_size=128, shuffle=False, num_workers=core.NUM_WORKERS)
        per = {c: [] for c in conds}
        for s in SEEDS:
            ck = T.ckpt_path(str(root_orig if s in ORIG_SEEDS else root_extra), cfgname, s, None)
            assert T.is_complete(ck), ck
            model = core.build_model(cfg, hp, pretrained=False)
            model.load_state_dict(torch.load(ck, map_location=core.DEVICE))
            model.eval()
            for c, m in conds.items():
                yt, yp, pr, cf = run(model, tl, cfg["head"] == "corn", m)
                assert np.array_equal(yt, labels[test])
                per[c].append(M.all_metrics(yt, yp, pr, cf))
        res = {}
        for c in conds:
            res[c] = {k: M.agg([m[k] for m in per[c]]) for k in ("accuracy", "qwk", "ece")}
            a, q, e = res[c]["accuracy"], res[c]["qwk"], res[c]["ece"]
            print(f"{sysname:<9}{c:<18} acc {a[0]*100:6.2f} +/- {a[1]*100:4.2f}  "
                  f"qwk {q[0]:.4f}  ece {e[0]:.4f}")
        out["results"][sysname] = res

    Path("results").mkdir(exist_ok=True)
    f = Path(f"results/occlusion_test_{datetime.now():%Y-%m-%d}.json")
    f.write_text(json.dumps(out, indent=2))
    print(f"\nverified: {len(SYSTEMS)} systems x {len(SEEDS)} seeds x {len(conds)} conditions "
          f"on {len(test)} test images -> {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
