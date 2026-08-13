"""Markdown / JSON research report generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class ResearchReportDocument:
    summary: dict[str, Any]
    statistics: list[dict[str, Any]]
    rankings: dict[str, list[Any]]
    recommendations: list[str]
    accepted_features: list[dict[str, Any]]
    rejected_features: list[dict[str, Any]]
    weak_features: list[dict[str, Any]]
    correlated_groups: list[list[str]]
    charts: dict[str, str] = field(default_factory=dict)
    reasoning: dict[str, str] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "summary": self.summary,
            "statistics": self.statistics,
            "rankings": self.rankings,
            "recommendations": self.recommendations,
            "accepted_features": self.accepted_features,
            "rejected_features": self.rejected_features,
            "weak_features": self.weak_features,
            "correlated_groups": self.correlated_groups,
            "charts": self.charts,
            "reasoning": self.reasoning,
        }

    def to_markdown(self) -> str:
        lines: list[str] = [
            "# Feature Research Validation Report",
            "",
            f"Generated: `{self.generated_at}`",
            "",
            "## Summary",
            "",
        ]
        for k, v in self.summary.items():
            lines.append(f"- **{k}**: {v}")
        lines.extend(["", "## Feature Rankings", ""])
        for section, items in self.rankings.items():
            lines.append(f"### {section.replace('_', ' ').title()}")
            lines.append("")
            if not items:
                lines.append("_None_")
            else:
                for item in items[:50]:
                    if isinstance(item, dict):
                        name = item.get("feature", item.get("name", "?"))
                        score = item.get("score", item.get("value", ""))
                        lines.append(f"- `{name}` — {score}")
                    else:
                        lines.append(f"- `{item}`")
            lines.append("")

        lines.extend(["## Accepted Features", ""])
        for item in self.accepted_features:
            lines.append(
                f"- `{item.get('feature')}` score={item.get('score')} — {item.get('reason', '')}"
            )
        lines.extend(["", "## Rejected Features", ""])
        for item in self.rejected_features:
            lines.append(
                f"- `{item.get('feature')}` score={item.get('score')} — {item.get('reason', '')}"
            )
        lines.extend(["", "## Weak Features", ""])
        for item in self.weak_features:
            lines.append(
                f"- `{item.get('feature')}` score={item.get('score')} — {item.get('reason', '')}"
            )

        lines.extend(["", "## Highly Correlated Groups", ""])
        if not self.correlated_groups:
            lines.append("_None_")
        else:
            for g in self.correlated_groups:
                lines.append("- " + ", ".join(f"`{x}`" for x in g))

        lines.extend(["", "## Recommendations", ""])
        for rec in self.recommendations:
            lines.append(f"- {rec}")

        lines.extend(["", "## Reasoning", ""])
        for feat, reason in self.reasoning.items():
            lines.append(f"- `{feat}`: {reason}")

        if self.charts:
            lines.extend(["", "## Charts", ""])
            for name, path in self.charts.items():
                lines.append(f"- **{name}**: `{path}`")

        lines.extend(["", "## Statistics (sample)", ""])
        for row in self.statistics[:25]:
            lines.append(
                f"- `{row.get('name')}` mean={row.get('mean')} "
                f"std={row.get('std')} skew={row.get('skewness')} "
                f"missing%={row.get('missing_pct')} dist={row.get('distribution_type')}"
            )
        lines.append("")
        return "\n".join(lines)


class ReportWriter:
    def write(
        self,
        document: ResearchReportDocument,
        output_dir: Path,
        *,
        write_markdown: bool = True,
        write_json: bool = True,
    ) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        if write_json:
            jp = output_dir / "feature_research_report.json"
            jp.write_text(json.dumps(document.to_dict(), indent=2, default=str), encoding="utf-8")
            paths["json"] = jp
        if write_markdown:
            mp = output_dir / "feature_research_report.md"
            mp.write_text(document.to_markdown(), encoding="utf-8")
            paths["markdown"] = mp
        return paths
