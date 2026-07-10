from __future__ import annotations

from app.eval.judge import LLMJudge


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


def test_score_facts_clamps_too_many_bools():
    # the live failure mode: local judge returned 2 bools for a 1-fact query
    judge = LLMJudge(llm=lambda p: "[true, false]", max_retries=0)
    assert judge.score_facts("ans", ["f1"]) == [True]


def test_score_facts_pads_too_few_bools_as_not_covered():
    judge = LLMJudge(llm=lambda p: "[true]", max_retries=0)
    assert judge.score_facts("ans", ["f1", "f2"]) == [True, False]


def test_score_facts_all_false_when_unparseable():
    judge = LLMJudge(llm=lambda p: "nonsense", max_retries=0)
    assert judge.score_facts("ans", ["f1", "f2"]) == [False, False]


def test_check_forbidden_is_substring():
    judge = LLMJudge(llm=lambda p: "[]")
    out = judge.check_forbidden("It is safe to combine these.", ["safe to combine", "no risk"])
    assert out == [True, False]


def test_empty_facts_returns_empty_without_calling_llm():
    def boom(prompt: str) -> str:
        raise AssertionError("should not be called")

    assert LLMJudge(llm=boom).score_facts("ans", []) == []
