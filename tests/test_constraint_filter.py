from __future__ import annotations

from agent.utils.constraint_filter import CandidateConstraintConfig, CandidateConstraintEngine


def test_constraint_engine_filters_non_single_cjk_and_dedupes():
    engine = CandidateConstraintEngine()
    result = engine.apply(["海", "天空", "海", "A", "山"])
    assert result == ["海", "山"]


def test_constraint_engine_respects_max_candidates():
    engine = CandidateConstraintEngine(CandidateConstraintConfig(max_candidates=2))
    result = engine.apply(["天", "地", "人"])
    assert result == ["天", "地"]
