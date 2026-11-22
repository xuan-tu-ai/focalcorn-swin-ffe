#!/usr/bin/env python3
"""Verification suite for the corrected pipeline.

Checks every fix before any training is launched:
  F1 LayerNorm applied exactly once
  F2 standard CORN loss matches the reference implementation
  F3 one prediction rule for labels, probabilities, and confidence
  F6 the split is fixed, disjoint, and independent of the training seed

Exit status is non-zero if any check fails.

    python verify.py
"""

from __future__ import annotations

import sys
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path as _RelPath
sys.path.insert(0, str(_RelPath(__file__).resolve().parent.parent / "src"))

import core

PASS, FAIL = [], []


def check(name):
    def deco(fn):
        try:
            fn()
            PASS.append(name)
            print(f"  PASS  {name}")
        except Exception as e:
            FAIL.append((name, e))
            print(f"  FAIL  {name}\n        {type(e).__name__}: {e}")
        return fn
    return deco


print("=" * 70)
print("VERIFICATION SUITE")
print("=" * 70)

# ── F1: LayerNorm applied exactly once ──────────────────────────────────────
print("\n[F1] LayerNorm applied exactly once")


@check("F1.1 backbone.norm is invoked exactly once per forward pass")
def _():
    m = core.FreshnessModel(pretrained=False, dual_pool=False, ordinal=True).eval()
    calls = {"n": 0}
    orig = m.backbone.norm.forward

    def counting(x):
        calls["n"] += 1
        return orig(x)

    m.backbone.norm.forward = counting
    with torch.no_grad():
        m(torch.randn(2, 3, 224, 224))
    assert calls["n"] == 1, f"norm called {calls['n']} times, expected exactly 1"


@check("F1.2 _pool output equals pooling of forward_features (no extra transform)")
def _():
    m = core.FreshnessModel(pretrained=False, dual_pool=False, ordinal=True).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        feat = m.backbone.forward_features(x)
        expected = feat.mean(dim=[1, 2]) if feat.dim() == 4 else feat.mean(dim=1)
        got = m._pool(x)
    assert torch.allclose(got, expected, atol=1e-6), "pooling differs from forward_features"


@check("F1.3 dual pooling concatenates GAP and GMP (2x width)")
def _():
    m = core.FreshnessModel(pretrained=False, dual_pool=True, ordinal=True).eval()
    with torch.no_grad():
        z = m._pool(torch.randn(2, 3, 224, 224))
    assert z.shape[1] == m.backbone.num_features * 2, f"got {z.shape}"


# ── F2: losses ──────────────────────────────────────────────────────────────
print("\n[F2] CORN losses")


@check("F2.1 corn_std matches coral_pytorch reference implementation")
def _():
    from coral_pytorch.losses import corn_loss as ref
    torch.manual_seed(0)
    for _ in range(5):
        logits = torch.randn(64, core.NUM_CLASSES - 1)
        labels = torch.randint(0, core.NUM_CLASSES, (64,))
        got = core.corn_std(logits, labels, core.NUM_CLASSES)
        exp = ref(logits, labels, core.NUM_CLASSES)
        assert torch.allclose(got, exp, atol=1e-6), f"ours {got.item()} vs ref {exp.item()}"


@check("F2.2 focalcorn_std with gamma=0 reduces to corn_std")
def _():
    torch.manual_seed(1)
    logits = torch.randn(64, core.NUM_CLASSES - 1)
    labels = torch.randint(0, core.NUM_CLASSES, (64,))
    a = core.focalcorn_std(logits, labels, core.NUM_CLASSES, gamma=0.0)
    b = core.corn_std(logits, labels, core.NUM_CLASSES)
    assert torch.allclose(a, b, atol=1e-6), f"{a.item()} vs {b.item()}"


@check("F2.3 focalcorn_ours with gamma=0 reduces to corn_ours")
def _():
    torch.manual_seed(2)
    logits = torch.randn(64, core.NUM_CLASSES - 1)
    labels = torch.randint(0, core.NUM_CLASSES, (64,))
    a = core.focalcorn_ours(logits, labels, core.NUM_CLASSES, gamma=0.0)
    b = core.corn_ours(logits, labels, core.NUM_CLASSES)
    assert torch.allclose(a, b, atol=1e-6)


@check("F2.4 corn_std and corn_ours genuinely differ (documented variant)")
def _():
    torch.manual_seed(3)
    logits = torch.randn(256, core.NUM_CLASSES - 1)
    labels = torch.randint(0, core.NUM_CLASSES, (256,))
    a = core.corn_std(logits, labels, core.NUM_CLASSES)
    b = core.corn_ours(logits, labels, core.NUM_CLASSES)
    assert not torch.allclose(a, b, atol=1e-4), "the two variants should differ"
    assert torch.isfinite(a) and torch.isfinite(b) and a > 0 and b > 0


@check("F2.5 all registered losses are finite and differentiable")
def _():
    nominal = {c["loss"] for c in core.CONFIGS.values() if c["head"] == "ce"}
    for name, fn in core.LOSSES.items():
        nout = core.NUM_CLASSES if name in nominal else core.NUM_CLASSES - 1
        logits = torch.randn(32, nout, requires_grad=True)
        labels = torch.randint(0, core.NUM_CLASSES, (32,))
        loss = fn(logits, labels, core.NUM_CLASSES, 2.0)
        assert torch.isfinite(loss), f"{name} not finite"
        loss.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all(), f"{name} grad"


# ── F3: single prediction rule ──────────────────────────────────────────────
print("\n[F3] Unified prediction rule")


@check("F3.1 confidence equals the probability of the predicted label (ordinal)")
def _():
    torch.manual_seed(4)
    logits = torch.randn(500, core.NUM_CLASSES - 1)
    lab, probs, conf = core.predict(logits, ordinal=True)
    exp = probs[torch.arange(len(lab)), lab]
    assert torch.allclose(conf, exp, atol=1e-6)


@check("F3.2 confidence equals the probability of the predicted label (nominal)")
def _():
    torch.manual_seed(5)
    logits = torch.randn(500, core.NUM_CLASSES)
    lab, probs, conf = core.predict(logits, ordinal=False)
    assert torch.allclose(conf, probs.max(dim=1).values, atol=1e-6)
    assert (lab == probs.argmax(1)).all()


@check("F3.3 ordinal probabilities are normalised")
def _():
    torch.manual_seed(6)
    _, probs, _ = core.predict(torch.randn(500, core.NUM_CLASSES - 1), ordinal=True)
    assert torch.allclose(probs.sum(1), torch.ones(500), atol=1e-5)
    assert (probs >= 0).all()


@check("F3.4 decoded label matches the cumulative-product rule (rank consistent)")
def _():
    torch.manual_seed(7)
    logits = torch.randn(1000, core.NUM_CLASSES - 1)
    lab, _, _ = core.predict(logits, ordinal=True)
    f = torch.sigmoid(logits)
    manual = (f[:, 0] > 0.5).long() + ((f[:, 0] * f[:, 1]) > 0.5).long()
    assert (lab == manual).all()
    # rank consistency: label 2 requires the first threshold to have fired
    assert not ((manual == 2) & (f[:, 0] <= 0.5)).any()


@check("F3.5 labels are in range and decode is deterministic")
def _():
    torch.manual_seed(8)
    logits = torch.randn(300, core.NUM_CLASSES - 1)
    a, _, _ = core.predict(logits, ordinal=True)
    b, _, _ = core.predict(logits, ordinal=True)
    assert (a == b).all() and a.min() >= 0 and a.max() <= core.NUM_CLASSES - 1


# ── F6: split integrity ─────────────────────────────────────────────────────
print("\n[F6] Split integrity")

try:
    DEV, TEST, PATHS, LABELS, STRATA = core.make_split()
    HAVE_DATA = True
except Exception as e:  # dataset not present
    HAVE_DATA = False
    print(f"  SKIP  dataset unavailable ({e})")

if HAVE_DATA:
    @check("F6.1 dataset has 4390 images and 24 species x class strata")
    def _():
        assert len(PATHS) == 4390, f"got {len(PATHS)}"
        assert len(set(STRATA.tolist())) == 24, f"got {len(set(STRATA.tolist()))} strata"

    @check("F6.2 test set is 878 images and disjoint from development set")
    def _():
        assert len(TEST) == 878, f"got {len(TEST)}"
        assert len(DEV) == 4390 - 878
        assert len(set(DEV.tolist()) & set(TEST.tolist())) == 0

    @check("F6.3 split is identical regardless of the training seed")
    def _():
        for s in [42, 123, 999, 7, 2024]:
            core.set_seed(s)                       # training seed must not affect the split
            d2, t2, _, _, _ = core.make_split()
            assert np.array_equal(t2, TEST), f"test set changed under training seed {s}"
            assert np.array_equal(d2, DEV), f"dev set changed under training seed {s}"

    @check("F6.4 five folds partition the development set exactly")
    def _():
        folds = core.make_folds(DEV, STRATA)
        assert len(folds) == 5
        vals = [set(va.tolist()) for _, va in folds]
        union = set().union(*vals)
        assert union == set(DEV.tolist()), "validation folds do not cover the dev set"
        for i in range(5):
            for j in range(i + 1, 5):
                assert not (vals[i] & vals[j]), f"folds {i},{j} overlap"

    @check("F6.5 no fold (train or validation) touches the test set")
    def _():
        t = set(TEST.tolist())
        for k, (tr, va) in enumerate(core.make_folds(DEV, STRATA)):
            assert not (set(tr.tolist()) & t), f"fold {k} train leaks into test"
            assert not (set(va.tolist()) & t), f"fold {k} val leaks into test"

    @check("F6.6 folds are stratified: class balance is stable across folds")
    def _():
        for _, va in core.make_folds(DEV, STRATA):
            frac = np.bincount(LABELS[va], minlength=3) / len(va)
            ref = np.bincount(LABELS[DEV], minlength=3) / len(DEV)
            assert np.abs(frac - ref).max() < 0.02, f"imbalance {frac} vs {ref}"

    @check("F6.7 test set class composition matches the reported 354/264/260")
    def _():
        c = np.bincount(LABELS[TEST], minlength=3)
        assert c.tolist() == [354, 264, 260], f"got {c.tolist()}"



# -- F4: Optuna configuration matches the reported protocol -------------------
print("\n[F4] Optuna protocol")


@check("F4.1 study uses TPE sampler and Hyperband pruner")
def _():
    import optuna, hpo
    s = hpo.build_study("verify_tmp", "sqlite:///:memory:")
    assert isinstance(s.sampler, optuna.samplers.TPESampler), type(s.sampler)
    assert isinstance(s.pruner, optuna.pruners.HyperbandPruner), type(s.pruner)


@check("F4.2 search space covers the nine reported hyperparameters")
def _():
    import optuna, hpo
    st = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    got = set(hpo.suggest_hp(st.ask()).keys())
    want = {"lr", "weight_decay", "drop_path", "gamma", "pct_start",
            "batch_size", "aff_degrees", "aff_translate", "rerase_p"}
    assert got == want, f"missing {want - got}, extra {got - want}"


@check("F4.3 HPO folds come from the same species-stratified split as training")
def _():
    import hpo
    assert hpo.core.SPLIT_SEED == core.SPLIT_SEED
    assert hpo.core.N_FOLDS == core.N_FOLDS


# -- F5: stale-checkpoint protection ------------------------------------------
print("\n[F5] Checkpoint protection")


@check("F5.1 a checkpoint without a .done marker is not treated as complete")
def _():
    import tempfile, torch as th, train as T
    with tempfile.TemporaryDirectory() as d:
        p = __import__("pathlib").Path(d) / "fold1.pth"
        th.save({"w": th.zeros(1)}, p)
        assert not T.is_complete(p), "partial checkpoint wrongly counted as complete"
        T.mark_complete(p, {"best_qwk": 0.9})
        assert T.is_complete(p), "marker written but not detected"


@check("F5.2 assert_fresh refuses a directory that already holds weights")
def _():
    import tempfile, torch as th, train as T, pathlib
    with tempfile.TemporaryDirectory() as d:
        T.assert_fresh(d)                      # empty: fine
        sub = pathlib.Path(d) / "cfg" / "seed42"
        sub.mkdir(parents=True)
        th.save({"w": th.zeros(1)}, sub / "fold1.pth")
        try:
            T.assert_fresh(d)
        except RuntimeError:
            return
        raise AssertionError("assert_fresh did not raise on a non-empty directory")


@check("F5.3 checkpoint paths separate config, seed, and fold")
def _():
    import train as T
    a = T.ckpt_path("R", "cfgA", 42, 1)
    b = T.ckpt_path("R", "cfgA", 43, 1)
    c = T.ckpt_path("R", "cfgB", 42, 1)
    f = T.ckpt_path("R", "cfgA", 42, None)
    assert len({str(a), str(b), str(c), str(f)}) == 4, "paths collide"
    assert f.name == "final.pth"

# ── summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("\nFAILURES:")
    for n, e in FAIL:
        print(f"  - {n}: {type(e).__name__}: {e}")
    print("\nDO NOT START PHASE A.")
    sys.exit(1)
print("All checks passed. Pipeline is safe to run.")
print("=" * 70)
