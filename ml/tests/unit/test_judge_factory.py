from __future__ import annotations

from app.config import Settings
from app.rag.judge_clients import OpenAIChat, build_judge


class FakeOllama:
    def generate(self, prompt):
        return "[true, false]"


class FakeChat:
    def __call__(self, prompt):
        return "[true, false]"


def test_build_judge_uses_ollama_by_default():
    s = Settings(judge_provider="ollama")
    judge = build_judge(s, ollama=FakeOllama())
    assert judge.score_facts("answer", ["fact one", "fact two"]) == [True, False]


def test_build_judge_uses_openai_when_selected():
    s = Settings(judge_provider="openai")
    judge = build_judge(s, openai_chat=FakeChat())
    assert judge.score_facts("answer", ["fact one", "fact two"]) == [True, False]


def test_build_judge_rejects_unknown_provider():
    s = Settings(judge_provider="bogus")
    try:
        build_judge(s)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown provider")


class FakeOpenAIClient:
    class chat:
        class completions:
            @staticmethod
            def create(model, messages):
                class M:
                    class choices:
                        pass
                obj = type("R", (), {})()
                obj.choices = [type("C", (), {"message": type("Msg", (), {"content": "[true]"})()})()]
                return obj


def test_openai_chat_extracts_message_content():
    chat = OpenAIChat("k", "gpt-4o-mini", client=FakeOpenAIClient())
    assert chat("grade this") == "[true]"
