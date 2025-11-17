"""
Data models for code change documentation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CodeChange:
    """Represents a single code change"""

    title: str
    location: str  # e.g., "Lines 42-45"
    old_code: str
    new_code: str
    reason: str  # Why this change was made
    impact: str  # How this improved the metrics

    def to_markdown(self) -> str:
        """Convert to markdown format"""
        md = f"### {self.title}\n\n"
        md += f"**Location:** {self.location}\n\n"
        md += "**What changed:**\n```diff\n"

        # Format as diff
        for line in self.old_code.strip().split('\n'):
            md += f"- {line}\n"
        for line in self.new_code.strip().split('\n'):
            md += f"+ {line}\n"

        md += "```\n\n"
        md += f"**Why:** {self.reason}\n\n"
        md += f"**Impact:** {self.impact}\n\n"

        return md


@dataclass
class MetricChange:
    """Represents a change in a metric"""

    name: str
    before: float
    after: float

    @property
    def change(self) -> float:
        """Calculate the change"""
        return self.after - self.before

    @property
    def percent_change(self) -> float:
        """Calculate the percent change"""
        if self.before == 0:
            return 0.0
        return ((self.after - self.before) / abs(self.before)) * 100


@dataclass
class ChangeDocumentation:
    """Complete documentation of code changes"""

    summary: str
    changes: List[CodeChange] = field(default_factory=list)
    metric_changes: List[MetricChange] = field(default_factory=list)
    overall_impact: str = ""

    def to_markdown(self) -> str:
        """Convert to markdown format"""
        md = "# Code Evolution Changes\n\n"

        # Summary
        md += "## Summary\n\n"
        md += f"{self.summary}\n\n"

        # Metrics table
        if self.metric_changes:
            md += "## Metrics Improvement\n\n"
            md += "| Metric | Before | After | Change |\n"
            md += "|--------|--------|-------|--------|\n"

            for metric in self.metric_changes:
                change_str = f"{metric.percent_change:+.1f}%"
                md += f"| {metric.name} | {metric.before:.6f} | {metric.after:.6f} | {change_str} |\n"

            md += "\n"

        # Detailed changes
        if self.changes:
            md += "## Detailed Changes\n\n"

            for i, change in enumerate(self.changes, 1):
                md += change.to_markdown()

        # Overall impact
        if self.overall_impact:
            md += "## Overall Impact\n\n"
            md += f"{self.overall_impact}\n"

        return md
