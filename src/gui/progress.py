"""
Progress tracking dataclasses for the pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class StepInfo:
    """Information about a single pipeline step."""

    name: str
    status: str = "pending"  # pending | running | done | error
    start_time: float | None = None
    end_time: float | None = None
    details: str = ""
    output: str = ""


@dataclass
class PipelineProgress:
    """Tracks the overall pipeline progress."""

    steps: list[StepInfo] = field(default_factory=list)
    current_step: int = 0
    total_steps: int = 0
    overall_status: str = "idle"  # idle | running | done | error
    error_message: str = ""

    def add_step(self, name: str) -> StepInfo:
        step = StepInfo(name=name)
        self.steps.append(step)
        self.total_steps = len(self.steps)
        return step

    def start_step(self, index: int, details: str = "") -> None:
        if 0 <= index < len(self.steps):
            self.steps[index].status = "running"
            self.steps[index].start_time = time.time()
            self.steps[index].details = details
            self.current_step = index

    def complete_step(self, index: int, output: str = "") -> None:
        if 0 <= index < len(self.steps):
            step = self.steps[index]
            step.status = "done"
            step.end_time = time.time()
            step.output = output

    def fail_step(self, index: int, error: str) -> None:
        if 0 <= index < len(self.steps):
            step = self.steps[index]
            step.status = "error"
            step.end_time = time.time()
            step.output = f"❌ {error}"
            self.error_message = error

    def get_elapsed(self, index: int) -> str:
        if 0 <= index < len(self.steps):
            step = self.steps[index]
            if step.start_time and step.end_time:
                return f"{step.end_time - step.start_time:.1f}s"
            elif step.start_time:
                return f"{time.time() - step.start_time:.1f}s (running)"
        return ""

    def format_progress(self) -> str:
        """Format the progress as a human-readable markdown string."""
        lines = []
        lines.append(f"## Pipeline Progress ({self.overall_status})")
        lines.append("")

        for i, step in enumerate(self.steps):
            icon = {
                "pending": "⏳",
                "running": "🔄",
                "done": "✅",
                "error": "❌",
            }.get(step.status, "⏳")

            elapsed = self.get_elapsed(i)
            detail_str = f" – {step.details}" if step.details else ""
            output_str = (
                f"\n   > {step.output}"
                if step.output and step.status != "running"
                else ""
            )

            lines.append(
                f"{icon} **Step {i + 1}: {step.name}** {elapsed}"
                f"{detail_str}{output_str}"
            )

        if self.error_message:
            lines.append("")
            lines.append(f"**Error:** {self.error_message}")

        return "\n".join(lines)
