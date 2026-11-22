#!/usr/bin/env python3
"""Metric computation from saved predictions.

Every metric is derived from the same predicted label and its own confidence, as
produced by ``core.predict``, so accuracy, ordinal agreement, and calibration all
refer to the identical decision.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

N_BINS = 15


def _bins(y_true, y_pred, conf, n_bins=N_BINS):
    correct = (y_pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            yield int(m.sum()), float(correct[m].mean()), float(conf[m].mean())


def ece(y_true, y_pred, conf, n_bins=N_BINS):
    n = len(y_true)
    return float(sum(c / n * abs(a - f) for c, a, f in _bins(y_true, y_pred, conf, n_bins)))


def mce(y_true, y_pred, conf, n_bins=N_BINS):
    gaps = [abs(a - f) for _, a, f in _bins(y_true, y_pred, conf, n_bins)]
    return float(max(gaps)) if gaps else 0.0


def brier(y_true, probs):
    onehot = np.eye(probs.shape[1])[y_true]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def nll(y_true, probs):
    p = np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def safety_counts(y_true, y_pred):
    """Two-class ordinal errors; NF->HF is the food-safety hazard."""
    nf_hf = int(np.sum((y_true == 2) & (y_pred == 0)))
    hf_nf = int(np.sum((y_true == 0) & (y_pred == 2)))
    return nf_hf, hf_nf, nf_hf + hf_nf


def ordinal_errors(y_true, y_pred):
    e = np.abs(y_true - y_pred)
    return [int((e == k).sum()) for k in (0, 1, 2)]


def all_metrics(y_true, y_pred, probs, conf) -> dict:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    probs = np.asarray(probs); conf = np.asarray(conf)
    nf_hf, hf_nf, jumps = safety_counts(y_true, y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "ece": ece(y_true, y_pred, conf),
        "mce": mce(y_true, y_pred, conf),
        "brier": brier(y_true, probs),
        "nll": nll(y_true, probs),
        "nf_to_hf": nf_hf, "hf_to_nf": hf_nf, "total_jumps": jumps,
        "n": int(len(y_true)),
    }


def cm(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=[0, 1, 2])


def agg(values):
    """Mean and sample standard deviation (ddof=1)."""
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
