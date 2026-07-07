from __future__ import annotations

from app.rag.generator import OllamaClient


def test_generate_posts_prompt_and_returns_response():
    seen = {}

    def fake_post(url, payload):
        seen["url"] = url
        seen["payload"] = payload
        return {"response": "Increased bleeding risk."}

    client = OllamaClient("http://localhost:11434", "llama3.1", post=fake_post)
    out = client.generate("Context...\n\nQuery: warfarin ibuprofen?")
    assert out == "Increased bleeding risk."
    assert seen["url"] == "http://localhost:11434/api/generate"
    assert seen["payload"]["model"] == "llama3.1"
    assert seen["payload"]["stream"] is False
    assert "warfarin ibuprofen" in seen["payload"]["prompt"]
