from __future__ import annotations

from app.eval.judge import LLMJudge
from app.rag.generator import OllamaClient


class OpenAIChat:
    """Callable judge backend using OpenAI chat completions. Inject `client`
    in tests to avoid the SDK/network."""

    def __init__(self, api_key: str, model: str, *, client=None) -> None:
        self.model = model
        if client is None:
            from openai import OpenAI  # lazy

            client = OpenAI(api_key=api_key)
        self.client = client

    def __call__(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content


def build_judge(settings, *, ollama=None, openai_chat=None) -> LLMJudge:
    provider = settings.judge_provider.lower()
    if provider == "ollama":
        client = ollama or OllamaClient(settings.ollama_base_url, settings.judge_model)
        return LLMJudge(client.generate, max_retries=2)
    if provider == "openai":
        chat = openai_chat or OpenAIChat(settings.openai_api_key, settings.judge_model)
        return LLMJudge(chat, max_retries=2)
    raise ValueError(f"unknown judge_provider: {settings.judge_provider!r}")
