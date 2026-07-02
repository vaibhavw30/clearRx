from __future__ import annotations

import pytest

from app.eval.judge import JudgeError, LLMJudge


def test_score_facts_parses_json():
    judge = LLMJudge(llm=lambda p: "[true, false, true]")
    assert judge.score_facts("ans", ["f1", "f2", "f3"]) == [True, False, True]


def test_score_facts_retries_then_succeeds():
    calls = {"n": 0}

    def flaky(prompt: str) -> str:
        calls["n"] += 1
        return "nonsense" if calls["n"] == 1 else "[true]"

    judge = LLMJudge(llm=flaky, max_retries=1)
    assert judge.score_facts("ans", ["f1"]) == [True]
    assert calls["n"] == 2


def test_score_facts_raises_on_wrong_length():
    judge = LLMJudge(llm=lambda p: "[true]", max_retries=0)
    with pytest.raises(JudgeError):
        judge.score_facts("ans", ["f1", "f2"])


def test_check_forbidden_is_substring():
    judge = LLMJudge(llm=lambda p: "[]")
    out = judge.check_forbidden("It is safe to combine these.", ["safe to combine", "no risk"])
    assert out == [True, False]


def test_empty_facts_returns_empty_without_calling_llm():
    def boom(prompt: str) -> str:
        raise AssertionError("should not be called")

    assert LLMJudge(llm=boom).score_facts("ans", []) == []
