from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def _is_single_cjk(text: str) -> bool:
    return len(text) == 1 and "\u4e00" <= text <= "\u9fff"


@dataclass
class CandidateConstraintConfig:
    require_single_cjk: bool = True
    max_candidates: int = 30


class CandidateConstraintEngine:
    def __init__(self, config: CandidateConstraintConfig | None = None) -> None:
        self._config = config or CandidateConstraintConfig()

    def apply(self, candidates: Iterable[str]) -> list[str]:
        deduped: list[str] = []
        for candidate in candidates:
            word = str(candidate).strip()
            if not word:
                continue
            if self._config.require_single_cjk and not _is_single_cjk(word):
                continue
            if word in deduped:
                continue
            deduped.append(word)
            if len(deduped) >= self._config.max_candidates:
                break
        return deduped
