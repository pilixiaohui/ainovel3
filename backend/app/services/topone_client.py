"""TopOne Gemini 接口封装，支持真实大模型调用."""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import httpx

from app.config import (
    TOPONE_API_KEY,
    TOPONE_BASE_URL,
    TOPONE_DEFAULT_MODEL,
    TOPONE_SECONDARY_MODEL,
    TOPONE_TIMEOUT_SECONDS,
)


class ToponeClient:
    """轻量封装 TopOne API，默认使用 env 配置。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        secondary_model: str | None = None,
        timeout_seconds: float | None = None,
        allowed_models: Sequence[str] | None = None,
    ) -> None:
        self.api_key = TOPONE_API_KEY if api_key is None else api_key
        self.base_url = base_url or TOPONE_BASE_URL
        self.default_model = default_model or TOPONE_DEFAULT_MODEL
        self.secondary_model = secondary_model or TOPONE_SECONDARY_MODEL
        self.timeout_seconds = timeout_seconds or TOPONE_TIMEOUT_SECONDS
        self.allowed_models = tuple(allowed_models or (self.default_model, self.secondary_model))

    @staticmethod
    def _strip_thoughts(payload: dict[str, Any]) -> dict[str, Any]:
        """去除 thought 标记的内容块，避免上层收到思考过程."""
        candidates = payload.get("candidates", [])
        for candidate in candidates:
            content = candidate.get("content")
            if not content:
                continue
            parts = content.get("parts", [])
            filtered = [part for part in parts if not part.get("thought")]
            content["parts"] = filtered
        return payload

    def _validate_model(self, model: str) -> str:
        if model not in self.allowed_models:
            raise ValueError(f"Unsupported model: {model}")
        return model

    def _ensure_key(self) -> str:
        if not self.api_key:
            raise ValueError("TOPONE_API_KEY is required")
        return self.api_key

    @staticmethod
    def _to_parts(text: str) -> list[Mapping[str, str]]:
        return [{"text": text}]

    def _build_payload(
        self,
        *,
        messages: Iterable[Mapping[str, str]],
        system_instruction: str | None,
        generation_config: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        contents = [
            {"role": msg["role"], "parts": self._to_parts(msg["text"])}
            for msg in messages
        ]
        payload: dict[str, Any] = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = {"parts": self._to_parts(system_instruction)}
        if generation_config:
            payload["generationConfig"] = generation_config
        return payload

    async def generate_content(
        self,
        *,
        messages: Iterable[Mapping[str, str]],
        system_instruction: str | None = None,
        generation_config: Mapping[str, Any] | None = None,
        model: str | None = None,
        timeout: float | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> dict[str, Any]:
        """调用 TopOne generateContent 接口并返回 JSON 响应."""
        model_name = self._validate_model(model or self.default_model)
        api_key = self._ensure_key()
        payload = self._build_payload(
            messages=messages,
            system_instruction=system_instruction,
            generation_config=generation_config,
        )

        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout or self.timeout_seconds,
            transport=transport,
        ) as client:
            response = await client.post(
                f"/v1beta/models/{model_name}:generateContent",
                params={"key": api_key},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return self._strip_thoughts(data)
