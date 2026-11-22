#!/usr/bin/env python3
"""EigenCAM saliency maps for the final model.

Projects the final-stage activations onto their first principal component, which
gives a class-agnostic saliency map without gradients (Muhammad and Yeasin,
2020). Panels are drawn from the held-out test set and span species and
freshness grades, so the figure shows whether the model attends to the eye
itself rather than to background or packaging.

Uses the final model of the representative seed (median test kappa), the same one
behind the reported confusion matrix. One correctly classified test image is shown
per species, all eight; the grade cycles Highly Fresh, Fresh, Not Fresh down the
species list, and within that species and grade the image is drawn at random with
PICK_SEED, so no panel is hand-picked.

    python make_eigencam.py --seed <representative seed>
"""

from __future__ import annotations

import argparse
import platform
import socket
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import core

OUT = Path("outputs/figures")
ORIG_SEEDS = [42, 123, 999, 7, 2024]
# the original five seeds were trained by phases.py, the extra five by alt_branch_run.py
WEIGHTS = {True: Path("outputs/weights/final"), False: Path("outputs/weights/alt")}
PREDS = {True: Path("outputs/test/seed{}.npz"),
         False: Path("outputs/alt_branch/focalcorn_gap_ours/seed{}.npz")}
N_COLS = 4          # four panels per row fit the text block at legible size
PICK_SEED = 0
# printed sizes in inches: the figure is included unscaled, so text stays at 8 pt
IMG, GUT, ROWLAB, HEAD, GAP = 1.0, 0.05, 0.2, 0.32, 0.08
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})


def eigencam(model: core.FreshnessModel, x: torch.Tensor) -> np.ndarray:
    """First-principal-component saliency of the final-stage activations."""
    with torch.no_grad():
        feat = model.backbone.forward_features(x)      # B, H, W, C
    if feat.dim() == 3:                                 # B, N, C -> B, H, W, C
        n = feat.shape[1]
        s = int(round(n ** 0.5))
        feat = feat.reshape(feat.shape[0], s, s, feat.shape[2])
    a = feat[0].detach().cpu().numpy()                  # H, W, C
    h, w, c = a.shape
    flat = a.reshape(h * w, c)
    flat = flat - flat.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    sal = (flat @ vt[0]).reshape(h, w)
    if sal.sum() < 0:                                   # sign of a PC is arbitrary
        sal = -sal
    sal = np.maximum(sal, 0)
    return sal / (sal.max() + 1e-12)


def upscale(sal: np.ndarray, size: int) -> np.ndarray:
    """Nearest-then-smooth resize, so no extra dependency is needed."""
    r = np.kron(sal, np.ones((size // sal.shape[0] + 1, size // sal.shape[1] + 1)))[:size, :size]
    k = 9
    pad = np.pad(r, k // 2, mode="edge")
    out = np.zeros_like(r)
    for i in range(k):
        for j in range(k):
            out += pad[i:i + r.shape[0], j:j + r.shape[1]]
    return out / (k * k)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dev_idx, test_idx, paths, labels, strata = core.make_split()
    import json
    fc = json.loads(Path("outputs/final_config.json").read_text())
    cfg, hp, name = fc["cfg"], fc["hp"], fc["config"]

    orig = args.seed in ORIG_SEEDS
    ck = WEIGHTS[orig] / name / f"seed{args.seed}" / "final.pth"
    model = core.build_model(cfg, hp, pretrained=False)
    model.load_state_dict(torch.load(ck, map_location=core.DEVICE))
    model.eval()

    print("=" * 70)
    print("EIGENCAM for the final model")
    print(f"host      : {socket.gethostname()}  |  python {platform.python_version()}")
    print(f"torch     : {torch.__version__}  |  device {core.DEVICE}")
    print(f"config    : {name}   checkpoint {ck}")
    print(f"invoked   : make_eigencam.py --seed {args.seed}")
    print(f"timestamp : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    # one correctly classified test image per species; grade cycles HF, F, NF
    pred = np.load(str(PREDS[orig]).format(args.seed))
    assert np.array_equal(pred["y_true"], labels[test_idx])
    correct = pred["y_pred"] == pred["y_true"]
    species = np.array([Path(p).parent.name.split(" - ")[0] for p in paths[test_idx]])
    _, eval_tf = core.build_transforms(hp)

    rng = np.random.default_rng(PICK_SEED)
    picks = []
    for j, sp in enumerate(sorted(set(species))):
        want = j % core.NUM_CLASSES
        pool = np.where(correct & (species == sp) & (pred["y_true"] == want))[0]
        picks.append(int(rng.choice(pool)))

    n_blocks = int(np.ceil(len(picks) / N_COLS))
    block_h = HEAD + 2 * IMG + GUT
    fw, fh = ROWLAB + N_COLS * IMG + (N_COLS - 1) * GUT, n_blocks * block_h + (n_blocks - 1) * GAP
    fig = plt.figure(figsize=(fw, fh))

    def panel(block, row, col):
        x = ROWLAB + col * (IMG + GUT)
        y = block * (block_h + GAP) + HEAD + row * (IMG + GUT)
        ax = fig.add_axes([x / fw, 1 - (y + IMG) / fh, IMG / fw, IMG / fh])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        return ax

    for k, i in enumerate(picks):
        b, col = k // N_COLS, k % N_COLS
        p = paths[test_idx][i]
        img = Image.open(p).convert("RGB").resize((640, 640))
        x = eval_tf(Image.open(p).convert("RGB")).unsqueeze(0).to(core.DEVICE)
        sal = upscale(eigencam(model, x), 640)

        parts = species[i].split()
        short = f"{parts[0][0]}. {parts[1].lower()}" if len(parts) > 1 else species[i]
        top, bottom = panel(b, 0, col), panel(b, 1, col)
        top.imshow(img)
        italic = short.replace(" ", r"\,")             # thin space keeps the label narrow
        top.set_title(f"$\\it{{{italic}}}$\n{core.CLASS_NAMES[pred['y_true'][i]]}",
                      fontsize=8, pad=2, linespacing=1.1)
        bottom.imshow(img)
        bottom.imshow(sal, cmap="jet", alpha=0.45)
        if col == 0:
            top.set_ylabel("Input", fontsize=8, labelpad=3)
            bottom.set_ylabel("EigenCAM", fontsize=8, labelpad=3)
        print(f"  panel {k + 1}: {species[i]:<28} {core.CLASS_NAMES[pred['y_true'][i]]:<13} {p}")
    used = set(species[picks])

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F_eigencam.{ext}", bbox_inches="tight", pad_inches=0.02,
                    dpi=400)
    plt.close(fig)
    print(f"{len(picks)} panels over {len(used)} species -> {OUT}/F_eigencam.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
