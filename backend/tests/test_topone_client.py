import json
import httpx
import pytest

from app.services.topone_client import ToponeClient


@pytest.mark.asyncio
async def test_generate_content_builds_payload():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["json"] = json.loads(request.content.decode())
        mock_response = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "thought text", "thought": True},
                            {"text": "final text"},
                        ],
                    },
                    "finishReason": "STOP",
                }
            ]
        }
        return httpx.Response(200, json=mock_response)

    transport = httpx.MockTransport(handler)
    client = ToponeClient(
        api_key="test-key",
        base_url="https://api.toponeapi.top",
        default_model="gemini-3-pro-preview-11-2025",
        secondary_model="gemini-3-flash-preview",
    )

    data = await client.generate_content(
        messages=[{"role": "user", "text": "hello"}],
        system_instruction="sys",
        generation_config={"temperature": 0.5},
        model="gemini-3-flash-preview",
        transport=transport,
    )

    assert data["candidates"][0]["content"]["parts"] == [{"text": "final text"}]
    assert "gemini-3-flash-preview" in captured["url"]
    assert captured["json"]["contents"][0]["role"] == "user"
    assert captured["json"]["contents"][0]["parts"][0]["text"] == "hello"
    assert captured["json"]["systemInstruction"]["parts"][0]["text"] == "sys"
    assert captured["json"]["generationConfig"]["temperature"] == 0.5


@pytest.mark.asyncio
async def test_generate_content_rejects_unsupported_model():
    client = ToponeClient(api_key="k")
    with pytest.raises(ValueError):
        await client.generate_content(
            messages=[{"role": "user", "text": "hi"}],
            model="unknown-model",
        )


@pytest.mark.asyncio
async def test_generate_content_requires_api_key():
    client = ToponeClient(api_key="")  # force missing key even if .env provides placeholder
    with pytest.raises(ValueError):
        await client.generate_content(messages=[{"role": "user", "text": "hi"}])
