"""Hierarchical risk aggregation."""

from __future__ import annotations

from typing import Any

from iqrp.app.risk.aggregation.risk_aggregator import aggregate_risks


def hierarchical_aggregate(
    tree: dict[str, Any],
    *,
    method: str = "weighted_sum",
) -> dict[str, Any]:
    """Recursively aggregate a nested risk tree.

    Leaf nodes are numeric / RiskMeasure / dict-with-value.
    Internal nodes may include optional ``weights`` and ``children``::

        {
          "children": {"equity": {...}, "rates": {...}},
          "weights": {"equity": 0.6, "rates": 0.4},
        }
    """

    def _is_leaf(node: Any) -> bool:
        return not (isinstance(node, dict) and "children" in node)

    def _walk(node: Any, name: str = "root") -> dict[str, Any]:
        if _is_leaf(node):
            # Normalize leaf
            agg = aggregate_risks({name: node}, method=method)
            return {
                "name": name,
                "value": agg["value"],
                "risk_state": agg["risk_state"],
                "leaf": True,
                "measure": agg["measure"],
            }

        children = node.get("children", {})
        child_results = {k: _walk(v, name=k) for k, v in children.items()}
        measures = {k: v["value"] for k, v in child_results.items()}
        weights = node.get("weights")
        node_method = node.get("method", method)
        agg = aggregate_risks(measures, weights=weights, method=node_method)
        return {
            "name": name,
            "value": agg["value"],
            "risk_state": agg["risk_state"],
            "leaf": False,
            "method": node_method,
            "children": child_results,
            "measure": agg["measure"],
        }

    return {"name": "hierarchical_aggregate", "tree": _walk(tree, name="root")}
