"""Parallel orchestration of ensemble member fit / predict."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import polars as pl

from iqrp.app.regimes.ensemble.registry import EnsembleMember


def fit_members(
    members: list[EnsembleMember],
    frame: pl.DataFrame,
    feature_columns: list[str] | None = None,
    *,
    parallel: bool = True,
) -> list[EnsembleMember]:
    """Fit all members; failures are marked in metadata and skipped later."""

    def _fit_one(m: EnsembleMember) -> EnsembleMember:
        try:
            m.model.fit(frame, feature_columns)
            m.metadata["fitted"] = True
            m.metadata["error"] = None
        except Exception as exc:  # noqa: BLE001 - isolate member failures
            m.metadata["fitted"] = False
            m.metadata["error"] = str(exc)
        return m

    if not parallel or len(members) <= 1:
        return [_fit_one(m) for m in members]
    out: list[EnsembleMember] = []
    with ThreadPoolExecutor(max_workers=min(8, len(members))) as pool:
        futs = {pool.submit(_fit_one, m): m.name for m in members}
        for fut in as_completed(futs):
            out.append(fut.result())
    # preserve original order
    by_name = {m.name: m for m in out}
    return [by_name[m.name] for m in members if m.name in by_name]


def predict_members(
    members: list[EnsembleMember],
    frame: pl.DataFrame,
    feature_columns: list[str] | None = None,
    *,
    n_canonical: int,
    parallel: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """
    Returns ``(mapped_probas, hard_preds, active_names)`` for fitted members.
    """

    def _one(m: EnsembleMember) -> tuple[str, np.ndarray, np.ndarray] | None:
        if not m.metadata.get("fitted", m.model.is_fitted):
            return None
        try:
            proba = m.model.predict_proba(frame, feature_columns)
            mapped = m.map_proba(proba, n_canonical)
            hard = np.argmax(mapped, axis=1).astype(np.int64)
            return m.name, mapped, hard
        except Exception:  # noqa: BLE001
            return None

    results: list[tuple[str, np.ndarray, np.ndarray]] = []
    if not parallel or len(members) <= 1:
        for m in members:
            r = _one(m)
            if r is not None:
                results.append(r)
    else:
        with ThreadPoolExecutor(max_workers=min(8, len(members))) as pool:
            futs = [pool.submit(_one, m) for m in members]
            for fut in as_completed(futs):
                r = fut.result()
                if r is not None:
                    results.append(r)
        # restore member order
        order = {m.name: i for i, m in enumerate(members)}
        results.sort(key=lambda x: order.get(x[0], 10**9))

    if not results:
        from iqrp.app.core.exceptions import ValidationError

        raise ValidationError("No ensemble members produced predictions", code="ENS_NO_PREDS")
    names = [r[0] for r in results]
    probas = [r[1] for r in results]
    hards = [r[2] for r in results]
    return probas, hards, names


def member_log_likelihoods(
    members: list[EnsembleMember],
    frame: pl.DataFrame,
    feature_columns: list[str] | None = None,
) -> dict[str, float]:
    """Approximate LL as sum log max-proba (model-agnostic proxy)."""
    out: dict[str, float] = {}
    for m in members:
        if not m.metadata.get("fitted", m.model.is_fitted):
            continue
        try:
            p = m.model.predict_proba(frame, feature_columns)
            out[m.name] = float(np.sum(np.log(np.clip(np.max(p, axis=1), 1e-300, None))))
        except Exception:  # noqa: BLE001
            out[m.name] = -1e9
    return out


def collect_transition(members: list[EnsembleMember], n_canonical: int) -> np.ndarray:
    """Average available member transition matrices mapped into canonical space."""
    mats: list[np.ndarray] = []
    for m in members:
        if not m.metadata.get("fitted", m.model.is_fitted):
            continue
        try:
            tm = m.model.transition_matrix()
            # map via state_map: P_c ≈ M.T @ P_m @ M (row-normalize)
            sm = m.state_map
            if sm is None or sm.shape[0] != tm.shape[0]:
                k = min(tm.shape[0], n_canonical)
                out = np.eye(n_canonical)
                out[:k, :k] = tm[:k, :k]
                mats.append(out)
            else:
                mapped = sm.T @ tm @ sm
                mapped = np.clip(mapped, 0, None)
                row = np.clip(mapped.sum(axis=1, keepdims=True), 1e-300, None)
                mats.append(mapped / row)
        except Exception:  # noqa: BLE001
            continue
    if not mats:
        # sticky default
        eye = np.eye(n_canonical)
        return 0.9 * eye + 0.1 * (np.ones((n_canonical, n_canonical)) - eye) / max(
            n_canonical - 1, 1
        )
    avg = np.mean(np.stack(mats, axis=0), axis=0)
    avg = np.clip(avg, 0, None)
    row = np.clip(avg.sum(axis=1, keepdims=True), 1e-300, None)
    return avg / row
