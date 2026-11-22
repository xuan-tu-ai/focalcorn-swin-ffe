#!/usr/bin/env python3
"""Training routines: cross-validation folds and the final full-development model.

A fold is considered complete only when a ``.done`` marker is written beside its
checkpoint, so an interrupted run never leaves a partial checkpoint that a later
run would silently reuse.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import cohen_kappa_score, accuracy_score
from torch.utils.data import DataLoader

import core


# ── checkpoint bookkeeping (F5) ─────────────────────────────────────────────
def ckpt_path(root: str, tag: str, seed: int, fold) -> Path:
    f = "final" if fold is None else f"fold{fold}"
    return Path(root) / tag / f"seed{seed}" / f"{f}.pth"


def done_path(p: Path) -> Path:
    return Path(str(p) + ".done")


def is_complete(p: Path) -> bool:
    """A checkpoint counts only if its completion marker exists."""
    return Path(p).exists() and done_path(p).exists()


def mark_complete(p: Path, meta: dict) -> None:
    done_path(p).write_text(json.dumps(meta, indent=2))


def assert_fresh(root: str) -> None:
    """Refuse to start if the target directory already holds weights."""
    r = Path(root)
    if r.exists():
        existing = list(r.rglob("*.pth"))
        if existing:
            raise RuntimeError(
                f"{root} already contains {len(existing)} checkpoint(s). "
                f"Refusing to start: architecture, loss, or split may have changed. "
                f"Move or delete the directory, or pass --allow-existing."
            )


# ── evaluation helper ───────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, ordinal: bool):
    model.eval()
    L, P, C, Y = [], [], [], []
    for x, y in loader:
        lab, probs, conf = core.predict(model(x.to(core.DEVICE)).cpu(), ordinal)
        L.append(lab); P.append(probs); C.append(conf); Y.append(y)
    lab = torch.cat(L).numpy(); probs = torch.cat(P).numpy()
    conf = torch.cat(C).numpy(); y = torch.cat(Y).numpy()
    return dict(
        y_true=y, y_pred=lab, probs=probs, conf=conf,
        accuracy=accuracy_score(y, lab),
        qwk=cohen_kappa_score(y, lab, weights="quadratic"),
    )


# ── core training loop ──────────────────────────────────────────────────────
def _run(model, loss_fn, ordinal, hp, train_loader, val_loader,
         epochs, patience, trial=None, verbose=False):
    opt = optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=hp["lr"], steps_per_epoch=len(train_loader),
        epochs=epochs, pct_start=hp["pct_start"])

    best_qwk, best_epoch, no_imp, best_state = -1.0, 0, 0, None
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(core.DEVICE), y.to(core.DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), y, core.NUM_CLASSES, hp["gamma"])
            loss.backward(); opt.step(); sched.step()

        if val_loader is None:                     # final model: no validation
            history.append(float("nan"))
            continue

        r = evaluate(model, val_loader, ordinal)
        history.append(r["qwk"])
        if r["qwk"] > best_qwk:
            best_qwk, best_epoch, no_imp = r["qwk"], ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
        if verbose and (ep % 10 == 0 or no_imp == 0):
            print(f"      ep {ep:3d} val_qwk={r['qwk']:.4f} best={best_qwk:.4f}")
        if trial is not None:
            trial.report(r["qwk"], ep)
            import optuna
            if trial.should_prune():
                raise optuna.TrialPruned()
        if patience and no_imp >= patience:
            break
    return best_qwk, best_epoch, best_state, history


def train_fold(cfg: dict, hp: dict, fold: int, seed: int, paths, labels,
               folds, epochs=100, patience=30, trial=None, save_to=None,
               verbose=False, collect_val_preds=False):
    """Train one cross-validation fold; selection signal is validation QWK."""
    core.set_seed(seed)
    tr_idx, va_idx = folds[fold - 1]
    train_tf, eval_tf = core.build_transforms(hp)
    bs = int(hp["batch_size"])
    tl = DataLoader(core.FishDataset(paths[tr_idx], labels[tr_idx], train_tf),
                    batch_size=bs, shuffle=True, num_workers=core.NUM_WORKERS)
    vl = DataLoader(core.FishDataset(paths[va_idx], labels[va_idx], eval_tf),
                    batch_size=bs, shuffle=False, num_workers=core.NUM_WORKERS)
    model = core.build_model(cfg, hp)
    ordinal = cfg["head"] == "corn"
    t0 = time.time()
    best_qwk, best_ep, state, hist = _run(
        model, core.LOSSES[cfg["loss"]], ordinal, hp, tl, vl,
        epochs, patience, trial, verbose)
    if save_to is not None and state is not None:
        p = Path(save_to); p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, p)
        mark_complete(p, dict(best_qwk=best_qwk, best_epoch=best_ep,
                              minutes=round((time.time() - t0) / 60, 1), hp=hp))

    out = dict(best_qwk=best_qwk, best_epoch=best_ep,
               minutes=(time.time() - t0) / 60, history=hist)

    if collect_val_preds:
        # Restore the selected checkpoint and record its validation predictions,
        # so development-set tables and figures need no further training.
        if state is not None:
            model.load_state_dict(state)
        r = evaluate(model, vl, ordinal)
        out.update(val_idx=va_idx, y_true=r["y_true"], y_pred=r["y_pred"],
                   probs=r["probs"], conf=r["conf"])
    return out


def train_final(cfg: dict, hp: dict, seed: int, paths, labels, dev_idx,
                epochs: int, save_to=None, verbose=True):
    """Train the single final model on the whole development set.

    There is no validation split at this stage, so training runs for a fixed
    number of epochs with the one-cycle schedule completing over them, and the
    final-epoch weights are kept.
    """
    core.set_seed(seed)
    train_tf, _ = core.build_transforms(hp)
    tl = DataLoader(core.FishDataset(paths[dev_idx], labels[dev_idx], train_tf),
                    batch_size=int(hp["batch_size"]), shuffle=True,
                    num_workers=core.NUM_WORKERS)
    model = core.build_model(cfg, hp)
    t0 = time.time()
    _run(model, core.LOSSES[cfg["loss"]], cfg["head"] == "corn", hp, tl, None,
         epochs, patience=0, verbose=verbose)
    if save_to is not None:
        p = Path(save_to); p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), p)
        mark_complete(p, dict(epochs=epochs, seed=seed,
                              minutes=round((time.time() - t0) / 60, 1), hp=hp))
    return model
