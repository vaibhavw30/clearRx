from __future__ import annotations

import json
import re
from typing import Callable, Optional, Protocol


class JudgeError(Exception):
    pass


class Judge(Protocol):
    def score_facts(self, answer: str, facts: list[str]) -> list[bool]: ...
    def check_forbidden(self, answer: str, must_not_say: list[str]) -> list[bool]: ...


def build_fact_prompt(answer: str, facts: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts))
    return (
        "You are grading whether an ANSWER covers each FACT.\n"
        "Return ONLY a JSON array of booleans, one per fact, in order.\n\n"
        f"ANSWER:\n{answer}\n\nFACTS:\n{numbered}\n\nJSON:"
    )


def _parse_bools(text: str) -> list[bool]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise JudgeError(f"no JSON array in judge output: {text!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise JudgeError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, list) or not all(isinstance(x, bool) for x in data):
        raise JudgeError(f"expected list of booleans, got {data!r}")
    return data


class LLMJudge:
    def __init__(self, llm: Callable[[str], str], max_retries: int = 1) -> None:
        self.llm = llm
        self.max_retries = max_retries

    def score_facts(self, answer: str, facts: list[str]) -> list[bool]:
        if not facts:
            return []
        prompt = build_fact_prompt(answer, facts)
        last_result: Optional[list[bool]] = None
        for _ in range(self.max_retries + 1):
            try:
                result = _parse_bools(self.llm(prompt))
            except JudgeError:
                continue
            if len(result) == len(facts):
                return result
            last_result = result  # parseable but miscounted; remember for coercion
        # A single flaky judge response must not abort the whole eval run. Coerce
        # the best available parse to the expected length (clamp extras, pad
        # missing facts as not-covered). If nothing parsed, treat all facts as
        # not covered. The judge calibration report is what surfaces whether the
        # judge itself is trustworthy.
        best = last_result or []
        return (best + [False] * len(facts))[: len(facts)]

    def check_forbidden(self, answer: str, must_not_say: list[str]) -> list[bool]:
        low = answer.lower()
        return [phrase.lower() in low for phrase in must_not_say]
