"""Probability / forecast calibration methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass
class Calibrator:
    method: str
    params: dict[str, Any]

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.method == "temperature":
            t = float(self.params.get("temperature", 1.0))
            # treat x as logits or raw scores
            return 1.0 / (1.0 + np.exp(-x / max(t, 1e-6)))
        if self.method == "platt":
            a = float(self.params.get("a", 1.0))
            b = float(self.params.get("b", 0.0))
            return 1.0 / (1.0 + np.exp(-(a * x + b)))
        if self.method == "isotonic":
            xs = np.asarray(self.params["x"], dtype=np.float64)
            ys = np.asarray(self.params["y"], dtype=np.float64)
            return np.interp(x.reshape(-1), xs, ys, left=ys[0], right=ys[-1]).reshape(x.shape)
        if self.method == "dirichlet":
            # simplified: temperature on softmax probs
            raw = np.asarray(x, dtype=np.float64)
            if raw.ndim == 1:
                # interpret as positive-class probability or logit
                if np.all((raw >= 0) & (raw <= 1)):
                    p = np.column_stack([1.0 - raw, raw])
                else:
                    p = 1.0 / (1.0 + np.exp(-np.column_stack([-raw, raw])))
            else:
                p = np.clip(raw, 1e-6, None)
                p = p / p.sum(axis=-1, keepdims=True)
            p = np.clip(p, 1e-6, 1.0)
            p = p / p.sum(axis=-1, keepdims=True)
            logp = np.log(p)
            t = float(self.params.get("temperature", 1.0))
            logp = logp / max(t, 1e-6)
            logp -= logp.max(axis=-1, keepdims=True)
            out = np.exp(logp)
            return out / out.sum(axis=-1, keepdims=True)
        return x


def fit_calibrator(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    method: Literal["temperature", "platt", "isotonic", "dirichlet", "none"] = "platt",
) -> Calibrator | None:
    if method in {"none", None}:
        return None
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim == 2:
        s = s[:, -1]
    s = s.reshape(-1)
    n = min(y.size, s.size)
    y, s = y[:n], s[:n]
    # binarize y if needed
    if not np.isin(y, [0, 1]).all():
        y = (y > np.median(y)).astype(np.float64)
    if method == "temperature":
        best_t, best = 1.0, float("inf")
        for t in np.linspace(0.5, 5.0, 20):
            p = 1.0 / (1.0 + np.exp(-s / t))
            loss = float(np.mean((p - y) ** 2))
            if loss < best:
                best, best_t = loss, float(t)
        return Calibrator("temperature", {"temperature": best_t})
    if method == "platt":
        # logistic via Newton-ish closed form approx: a,b from logit regression OLS on clipped
        p = np.clip(s, 1e-4, 1 - 1e-4)
        # if scores look like logits, use as-is; else logit transform
        if np.all((s >= 0) & (s <= 1)):
            z = np.log(p / (1 - p))
        else:
            z = s
        X = np.column_stack([z, np.ones(n)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return Calibrator("platt", {"a": float(beta[0]), "b": float(beta[1])})
    if method == "isotonic":
        order = np.argsort(s)
        xs, ys = s[order], y[order]
        # pool adjacent violators (simple isotonic)
        y_iso = ys.astype(np.float64).copy()
        for _ in range(n):
            changed = False
            i = 0
            while i < n - 1:
                if y_iso[i] > y_iso[i + 1]:
                    avg = 0.5 * (y_iso[i] + y_iso[i + 1])
                    y_iso[i] = y_iso[i + 1] = avg
                    changed = True
                i += 1
            if not changed:
                break
        # compress to unique x
        ux, uy = [xs[0]], [y_iso[0]]
        for i in range(1, n):
            if xs[i] == ux[-1]:
                uy[-1] = 0.5 * (uy[-1] + y_iso[i])
            else:
                ux.append(xs[i])
                uy.append(y_iso[i])
        return Calibrator("isotonic", {"x": ux, "y": uy})
    if method == "dirichlet":
        best_t, best = 1.0, float("inf")
        for t in np.linspace(0.5, 5.0, 15):
            p = 1.0 / (1.0 + np.exp(-s / t))
            loss = float(np.mean((p - y) ** 2))
            if loss < best:
                best, best_t = loss, float(t)
        return Calibrator("dirichlet", {"temperature": best_t})
    return None


def apply_calibration(calibrator: Calibrator | None, scores: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return np.asarray(scores, dtype=np.float64)
    return calibrator.transform(scores)
