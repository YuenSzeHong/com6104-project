from __future__ import annotations

from agent.workflow_graph import build_workflow_graph, decide_after_validation


def test_workflow_graph_compiles():
    graph = build_workflow_graph()
    assert graph is not None


def test_decide_after_validation_accepts_when_score_high():
    decision = decide_after_validation(
        {
            "score": 0.9,
            "min_quality_score": 0.75,
            "attempt": 0,
            "max_attempts": 4,
        }
    )
    assert decision.accepted is True
    assert decision.next_stage == "completed"


def test_decide_after_validation_loops_when_under_threshold_and_attempts_left():
    decision = decide_after_validation(
        {
            "score": 0.6,
            "min_quality_score": 0.75,
            "attempt": 1,
            "max_attempts": 4,
        }
    )
    assert decision.accepted is False
    assert decision.next_stage == "compose"
