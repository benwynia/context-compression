"""Tests for the in-house LLM summarizer (local-7B-style OpenAI endpoint)."""

import json

import httpx
import pytest

from ctxc.compressor import CompressConfig, compress
from ctxc.models import is_digest, validate_chain
from ctxc.summarize import LlmSummarizer
from ctxc.synth import synth_session


def _mock_llm(reply: str, seen: list[dict], status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        if status != 200:
            return httpx.Response(status, json={"error": "boom"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": reply}}]},
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def _forcing_session():
    return synth_session(rounds=30, seed=11)


def test_llm_digest_is_used(counter):
    seen: list[dict] = []
    summ = LlmSummarizer(
        "http://localhost:11434", "qwen2.5:7b",
        client=_mock_llm("LLM-DIGEST: fixed retry, edited src/pay.py, tests green.", seen),
    )
    res = compress(_forcing_session(), budget=6_000,
                   config=CompressConfig(summarizer=summ), counter=counter)
    assert res.compressed_tokens <= 6_000
    assert validate_chain(res.messages) == []
    digest = next(m for m in res.messages if is_digest(m))
    assert "LLM-DIGEST" in digest["content"]
    # called exactly once, on the succeeding level only
    assert summ.calls == 1
    # url normalization: bare host -> /v1/chat/completions
    assert seen and seen[0]["model"] == "qwen2.5:7b"
    assert seen[0]["temperature"] == 0


def test_dead_endpoint_falls_back_to_deterministic(counter):
    seen: list[dict] = []
    summ = LlmSummarizer(
        "http://localhost:11434/v1", "qwen2.5:7b",
        client=_mock_llm("", seen, status=500),
    )
    res = compress(_forcing_session(), budget=6_000,
                   config=CompressConfig(summarizer=summ), counter=counter)
    # the budget guarantee survives a dead summarizer
    assert res.compressed_tokens <= 6_000
    digest = next(m for m in res.messages if is_digest(m))
    # deterministic digest body (extractive lines), not summarizer output
    assert "keep:" in digest["content"] or "- " in digest["content"]


def test_empty_reply_falls_back(counter):
    summ = LlmSummarizer("http://x", "m", client=_mock_llm("   ", []))
    res = compress(_forcing_session(), budget=6_000,
                   config=CompressConfig(summarizer=summ), counter=counter)
    assert res.compressed_tokens <= 6_000
    digest = next(m for m in res.messages if is_digest(m))
    assert "keep:" in digest["content"] or "- " in digest["content"]


def test_input_is_capped_for_small_context_models():
    seen: list[dict] = []
    summ = LlmSummarizer("http://x", "m", max_input_chars=500,
                         client=_mock_llm("ok summary", seen))
    summ([f"line {i} " + "x" * 100 for i in range(50)])
    user_msg = seen[0]["messages"][1]["content"]
    assert len(user_msg) <= 500 + 40  # cap plus the omission note
    assert "omitted" in user_msg
    assert user_msg.endswith("x" * 100)  # most recent lines kept


def test_api_key_env_is_used(monkeypatch):
    headers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        headers.append(request.headers.get("authorization", ""))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "s"}}]}
        )

    monkeypatch.setenv("LOCAL_LLM_KEY", "sekrit")
    summ = LlmSummarizer("http://x", "m", api_key_env="LOCAL_LLM_KEY",
                         client=httpx.Client(transport=httpx.MockTransport(handler)))
    summ(["a line"])
    assert headers == ["Bearer sekrit"]


def test_cli_summarizer_flags_must_pair(tmp_path):
    from ctxc.cli import main

    f = tmp_path / "s.json"
    f.write_text(json.dumps({"messages": [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"},
        {"role": "user", "content": "x"}, {"role": "assistant", "content": "y"},
    ]}))
    with pytest.raises(SystemExit, match="together"):
        main(["verify", str(f), "--summarizer-url", "http://localhost:11434"])
