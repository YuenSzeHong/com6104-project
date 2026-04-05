from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict, cast

from langgraph.graph import END, StateGraph


StageName = Literal[
    "midi_analysis",
    "melody_mapping",
    "candidate_recall",
    "compose",
    "validate",
    "completed",
    "error",
]


class WorkflowState(TypedDict, total=False):
    stage: StageName
    attempt: int
    max_attempts: int
    score: float
    min_quality_score: float
    accepted: bool
    error: str


@dataclass(frozen=True)
class WorkflowDecision:
    next_stage: StageName
    accepted: bool


def _identity(state: WorkflowState) -> WorkflowState:
    return state


def decide_after_validation(state: WorkflowState) -> WorkflowDecision:
    score = float(state.get("score", 0.0) or 0.0)
    threshold = float(state.get("min_quality_score", 1.0) or 1.0)
    attempt = int(state.get("attempt", 0) or 0)
    max_attempts = int(state.get("max_attempts", 0) or 0)

    if score >= threshold:
        return WorkflowDecision(next_stage="completed", accepted=True)
    if attempt + 1 >= max_attempts:
        return WorkflowDecision(next_stage="completed", accepted=False)
    return WorkflowDecision(next_stage="compose", accepted=False)


def _route_after_validate(state: WorkflowState) -> str:
    decision = decide_after_validation(state)
    return decision.next_stage


def build_workflow_graph():
    graph = StateGraph(cast(Any, WorkflowState))

    graph.add_node("midi_analysis", _identity)
    graph.add_node("melody_mapping", _identity)
    graph.add_node("candidate_recall", _identity)
    graph.add_node("compose", _identity)
    graph.add_node("validate", _identity)
    graph.add_node("completed", _identity)

    graph.set_entry_point("midi_analysis")
    graph.add_edge("midi_analysis", "melody_mapping")
    graph.add_edge("melody_mapping", "candidate_recall")
    graph.add_edge("candidate_recall", "compose")
    graph.add_edge("compose", "validate")
    graph.add_conditional_edges(
        "validate",
        _route_after_validate,
        {
            "compose": "compose",
            "completed": "completed",
        },
    )
    graph.add_edge("completed", END)

    return graph.compile()
