"""Overtrading / signal-churn diagnostics (detect & report, do not auto-suppress)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from iqrp.app.backtesting.horizon.trade_analytics import classify_side


def overtrading_diagnostics(
    sides: Sequence[Any],
    *,
    timestamps: Sequence[Any] | None = None,
    reentry_window: int = 1,
) -> dict[str, Any]:
    """Detect repeated signals, excessive reversals, churning, immediate re-entry.

    Does **not** suppress behavior — strategies may define their own cooldowns.
    """
    seq = [classify_side(s).value for s in sides]
    n = len(seq)
    if n == 0:
        return {
            "repeated_signals": 0,
            "excessive_reversals": 0,
            "signal_churning_rate": 0.0,
            "immediate_reentries": 0,
            "same_move_reentries": 0,
            "note": "Diagnostics only; no automatic suppression.",
        }

    repeated = 0
    for a, b in zip(seq, seq[1:], strict=False):
        if a == b and a != "FLAT":
            repeated += 1

    reversals = 0
    for a, b in zip(seq, seq[1:], strict=False):
        if {a, b} == {"LONG", "SHORT"}:
            reversals += 1

    changes = sum(1 for a, b in zip(seq, seq[1:], strict=False) if a != b)
    churn = changes / max(n - 1, 1)

    # Immediate re-entry: FLAT → same side within reentry_window after exit
    immediate = 0
    same_move = 0
    last_side = None
    flat_streak = 0
    for s in seq:
        if s == "FLAT":
            flat_streak += 1
            continue
        if last_side == s and 0 < flat_streak <= int(reentry_window):
            immediate += 1
            same_move += 1
        elif last_side is not None and last_side != s and flat_streak == 0:
            # direct reverse counted above
            pass
        last_side = s
        flat_streak = 0

    return {
        "n_bars": n,
        "repeated_signals": int(repeated),
        "excessive_reversals": int(reversals),
        "signal_churning_rate": float(churn),
        "immediate_reentries": int(immediate),
        "same_move_reentries": int(same_move),
        "change_count": int(changes),
        "note": (
            "Diagnostics only; do not auto-suppress. "
            "Strategies may define cooldown/re-entry rules explicitly."
        ),
    }


def position_path_sides(positions: Any) -> list[str]:
    p = np.asarray(positions, dtype=np.float64).reshape(-1)
    out: list[str] = []
    for x in p:
        if x > 1e-12:
            out.append("LONG")
        elif x < -1e-12:
            out.append("SHORT")
        else:
            out.append("FLAT")
    return out


__all__ = ["overtrading_diagnostics", "position_path_sides"]
