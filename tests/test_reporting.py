#!/usr/bin/env python3
"""Dry run of the whole reporting pipeline on synthetic predictions.

Fabricates development and test artefacts with exactly the schema the real
phases produce, then builds every table and figure. This proves that no table or
figure is missing code before any GPU time is spent. Nothing here touches the
dataset or any trained weights; outputs go to a scratch directory that is removed
afterwards.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import core  # noqa: E402

N_DEV, N_TEST = 3512, 878
SEEDS = [42, 123, 999, 7, 2024]
rng = np.random.default_rng(0)


def fake_preds(n, quality=0.9):
    """Predictions with a realistic ordinal structure and calibrated-ish confidence."""
    y = rng.integers(0, 3, size=n)
    keep = rng.random(n) < quality
    step = rng.choice([-1, 1], size=n)
    yp = np.where(keep, y, np.clip(y + step, 0, 2)).astype(np.int64)
    logits = rng.normal(size=(n, 2)) * 1.5
    f = 1 / (1 + np.exp(-logits))
    probs = np.stack([1 - f[:, 0], f[:, 0] * (1 - f[:, 1]), f[:, 0] * f[:, 1]], 1)
    probs = np.clip(probs, 1e-12, None)
    probs /= probs.sum(1, keepdims=True)
    conf = probs[np.arange(n), yp]
    return y.astype(np.int64), yp, probs.astype(np.float32), conf.astype(np.float32)


def main() -> int:
    scratch = Path(tempfile.mkdtemp(prefix="reportgate_"))
    out = scratch / "outputs"   # mirrors report.OUT
    (out / "oof").mkdir(parents=True)
    (out / "test").mkdir(parents=True)
    print(f"scratch: {scratch}")

    # --- fabricate Phase A2 artefacts (one per configuration) ---------------
    selection = []
    for i, name in enumerate(core.CONFIGS):
        y, yp, pr, cf = fake_preds(N_DEV, quality=0.86 + 0.01 * i)
        folds = rng.normal(0.88 + 0.005 * i, 0.01, size=core.N_FOLDS)
        epochs = rng.integers(40, 90, size=core.N_FOLDS)
        np.savez(out / "oof" / f"{name}.npz", y_true=y, y_pred=yp, probs=pr,
                 conf=cf, fold_qwks=folds, best_epochs=epochs)
        import metrics as _M
        selection.append({
            "config": name, **core.CONFIGS[name], "hp": dict(core.DEFAULT_HP),
            "fold_qwk_mean": float(folds.mean()),
            "fold_qwk_sd": float(folds.std(ddof=1)),
            "median_best_epoch": int(np.median(epochs)),
            "best_epochs": epochs.tolist(),
            "oof": _M.all_metrics(y, yp, pr, cf),
        })
    (out / "selection.json").write_text(json.dumps(selection, indent=2))

    win = max(selection, key=lambda r: r["fold_qwk_mean"])
    (out / "final_config.json").write_text(json.dumps({
        "config": win["config"], "cfg": core.CONFIGS[win["config"]],
        "hp": dict(core.DEFAULT_HP), "epochs": win["median_best_epoch"],
        "seeds": SEEDS, "dev_qwk_mean": win["fold_qwk_mean"],
        "dev_qwk_sd": win["fold_qwk_sd"], "dev_ece": win["oof"]["ece"],
        "selection_criterion": "validation QWK differences within one fold SD "
                               "treated as tied; tie broken by development-set ECE",
        "qwk_argmax": win["config"]}, indent=2))

    # --- fabricate Phase D artefacts (one per seed) -------------------------
    for s in SEEDS:
        y, yp, pr, cf = fake_preds(N_TEST, quality=0.89)
        np.savez(out / "test" / f"seed{s}.npz", y_true=y, y_pred=yp, probs=pr, conf=cf)

    # --- fabricate matched-baseline, comparator, and bootstrap artefacts ----
    (out / "baseline").mkdir(parents=True, exist_ok=True)
    (out / "archive_ce_gapgmp").mkdir(parents=True, exist_ok=True)
    for kind in ("rf", "et"):
        for s in SEEDS:
            y, yp, pr, cf = fake_preds(N_TEST, quality=0.86)
            np.savez(out / "baseline" / f"{kind}_seed{s}.npz",
                     y_true=y, y_pred=yp, probs=pr, conf=cf)
    (out / "baseline" / "results.json").write_text(json.dumps([{"classifier": "RF"}]))
    for s in SEEDS:
        y, yp, pr, cf = fake_preds(N_TEST, quality=0.885)
        np.savez(out / "archive_ce_gapgmp" / f"seed{s}.npz",
                 y_true=y, y_pred=yp, probs=pr, conf=cf)
    (scratch / "results").mkdir(parents=True, exist_ok=True)
    (scratch / "results" / "bootstrap_test_2026-01-01.json").write_text(json.dumps({
        "seed": 123, "n_boot": 10, "n_obs": N_TEST, "systems": ["a", "b"], "ref": "a",
        "comparisons": [{"vs": "ce_gapgmp", "metric": "QWK", "delta": 0.009,
                         "lo": -0.002, "hi": 0.02, "p_better": 0.95,
                         "significant": False}]}))

    # --- run the real report generator against them -------------------------
    ok = True
    for what in ["dev", "test", "compute", "extra"]:
        print(f"\n--- report.py {what} ---")
        r = subprocess.run([sys.executable, str(HERE.parent / "src" / "report.py"), what],
                           cwd=scratch, capture_output=True, text=True)
        print(r.stdout.strip()[-1500:] or "(no stdout)")
        if r.returncode != 0:
            ok = False
            print("STDERR:\n" + r.stderr.strip()[-2500:])

    tables = sorted(p.name for p in (out / "tables").glob("*.tex")) if (out / "tables").exists() else []
    figures = sorted(p.name for p in (out / "figures").glob("*.pdf")) if (out / "figures").exists() else []

    expect_t = {"T_selection.tex", "T_calibration_dev.tex", "T_safety_dev.tex",
                "T_test_per_seed.tex", "T_vs_prior.tex", "T_calibration_test.tex",
                "T_compute.tex", "T_ablation_2x2.tex", "T_baseline_full.tex",
                "T_significance.tex"}
    expect_f = {"F_selection.pdf", "F_cm_by_config_dev.pdf", "F_confusion_matrix.pdf",
                "F_cm_vs_prior.pdf", "F_ordinal_error.pdf", "F_reliability.pdf",
                "F_ablation_2x2.pdf", "F_safety.pdf"}

    print("\n" + "=" * 66)
    print(f"tables produced ({len(tables)}): {tables}")
    print(f"figures produced ({len(figures)}): {figures}")
    missing_t, missing_f = expect_t - set(tables), expect_f - set(figures)
    if missing_t:
        ok = False; print(f"MISSING TABLES : {sorted(missing_t)}")
    if missing_f:
        ok = False; print(f"MISSING FIGURES: {sorted(missing_f)}")

    # LaTeX sanity: balanced tabular environments
    for t in tables:
        txt = (out / "tables" / t).read_text()
        if txt.count("begin{tabular}") != txt.count("end{tabular}"):
            ok = False; print(f"UNBALANCED tabular in {t}")

    shutil.rmtree(scratch, ignore_errors=True)
    print("=" * 66)
    if not ok:
        print("REPORT GATE FAILED — do not start Phase A.")
        return 1
    print("REPORT GATE PASSED — every table and figure has working code.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
