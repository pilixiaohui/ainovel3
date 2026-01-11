import pytest

from app.llm.schemas import SceneRenderPayload
from app.llm.topone_gateway import ToponeGateway


class StubToponeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, str | list[dict[str, str]] | None]] = []
        self.default_model = "gemini-default"
        self.secondary_model = "gemini-secondary"

    async def generate_content(
        self,
        *,
        messages,
        system_instruction=None,
        generation_config=None,
        model=None,
    ):
        self.calls.append(
            {
                "system_instruction": system_instruction,
                "model": model,
                "messages": messages,
            }
        )
        return {
            "candidates": [
                {"content": {"parts": [{"text": "Rendered scene text"}]}}
            ]
        }


@pytest.mark.asyncio
async def test_render_scene_force_adds_drama_instruction():
    client = StubToponeClient()
    gateway = ToponeGateway(client)
    payload = SceneRenderPayload(
        voice_dna="冷静",
        conflict_type="internal",
        outline_requirement="保持原计划",
        user_intent="强制推进剧情",
        expected_outcome="主角继续前进",
        world_state={"hp": "50%"},
        logic_exception=True,
        force_reason="必须现在推进",
    )

    result = await gateway.render_scene(payload)

    assert result == "Rendered scene text"
    assert "戏剧性优先" in client.calls[0]["system_instruction"]


@pytest.mark.asyncio
async def test_render_scene_standard_keeps_prompt_clean():
    client = StubToponeClient()
    gateway = ToponeGateway(client)
    payload = SceneRenderPayload(
        voice_dna="冷静",
        conflict_type="internal",
        outline_requirement="保持原计划",
        user_intent="继续发展",
        expected_outcome="主角前进",
        world_state={"hp": "100%"},
    )

    await gateway.render_scene(payload)

    assert "戏剧性优先" not in client.calls[0]["system_instruction"]
