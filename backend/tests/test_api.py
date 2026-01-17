import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.llm.schemas import ImpactLevel, LogicCheckResult, StateProposal
from app.main import (
    app,
    get_graph_storage,
    get_llm_engine,
    get_snowflake_manager,
    get_topone_client,
    get_topone_gateway,
)
from app.models import CharacterSheet, SceneNode, SnowflakeRoot
from app.services.topone_client import ToponeClient
from app.storage.graph import DEFAULT_BRANCH_ID, GraphStorage


class DummyEngine:
    async def generate_root_structure(self, logline: str) -> SnowflakeRoot:
        return SnowflakeRoot(
            logline="Test story",
            three_disasters=["D1", "D2", "D3"],
            ending="End",
            theme="Testing",
        )

    async def generate_scene_list(self, root, characters):
        return [
            SceneNode(
                branch_id=DEFAULT_BRANCH_ID,
                expected_outcome="Outcome 1",
                conflict_type="internal",
                actual_outcome="",
                parent_act_id=None,
                is_dirty=False,
            )
        ]


def _seed_storage(storage: GraphStorage) -> tuple[str, CharacterSheet, SceneNode]:
    root = SnowflakeRoot(
        logline="Test story",
        three_disasters=["D1", "D2", "D3"],
        ending="End",
        theme="Testing",
    )
    character = CharacterSheet(
        name="Hero",
        ambition="A",
        conflict="B",
        epiphany="C",
        voice_dna="D",
    )
    scene = SceneNode(
        branch_id=DEFAULT_BRANCH_ID,
        expected_outcome="Outcome 1",
        conflict_type="internal",
        actual_outcome="",
        parent_act_id=None,
        is_dirty=False,
        pov_character_id=character.entity_id,
    )
    root_id = storage.save_snowflake(root, [character], [scene])
    return root_id, character, scene


def _seed_storage_with_scenes(
    storage: GraphStorage, count: int
) -> tuple[str, CharacterSheet, list[SceneNode]]:
    root = SnowflakeRoot(
        logline="Test story",
        three_disasters=["D1", "D2", "D3"],
        ending="End",
        theme="Testing",
    )
    character = CharacterSheet(
        name="Hero",
        ambition="A",
        conflict="B",
        epiphany="C",
        voice_dna="D",
    )
    if count < 2:
        raise ValueError("count must be >= 2 for multi-scene seed")
    scenes: list[SceneNode] = []
    for idx in range(count):
        scenes.append(
            SceneNode(
                branch_id=DEFAULT_BRANCH_ID,
                expected_outcome=f"Outcome {idx + 1}",
                conflict_type="internal" if idx % 2 == 0 else "external",
                actual_outcome="",
                parent_act_id=None,
                is_dirty=False,
                pov_character_id=character.entity_id,
            )
        )
    root_id = storage.save_snowflake(root, [character], scenes)
    return root_id, character, scenes


def test_generate_structure_endpoint(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "local")
    app.dependency_overrides[get_llm_engine] = lambda: DummyEngine()
    client = TestClient(app)

    response = client.post("/api/v1/snowflake/step2", json={"logline": "raw idea"})
    assert response.status_code == 200

    data = response.json()
    assert data["logline"] == "Test story"
    assert len(data["three_disasters"]) == 3

    app.dependency_overrides.clear()


def test_generate_scene_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "local")
    dummy_engine = DummyEngine()
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    app.dependency_overrides[get_llm_engine] = lambda: dummy_engine
    # override manager with relaxed scene bounds
    from app.logic.snowflake_manager import SnowflakeManager

    app.dependency_overrides[get_snowflake_manager] = lambda: SnowflakeManager(
        engine=dummy_engine, min_scenes=1, max_scenes=5, storage=storage
    )
    client = TestClient(app)

    payload = {
        "root": {
            "logline": "Test story",
            "three_disasters": ["D1", "D2", "D3"],
            "ending": "End",
            "theme": "Testing",
        },
        "characters": [
            {
                "name": "Hero",
                "ambition": "A",
                "conflict": "B",
                "epiphany": "C",
                "voice_dna": "D",
            }
        ],
    }
    try:
        response = client.post("/api/v1/snowflake/step4", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, dict)
        assert data["branch_id"] == DEFAULT_BRANCH_ID
        assert data["root_id"]
        assert data["scenes"][0]["expected_outcome"] == "Outcome 1"
        assert data["scenes"][0]["actual_outcome"] == ""
        assert data["scenes"][0]["branch_id"] == DEFAULT_BRANCH_ID
        assert data["scenes"][0]["is_dirty"] is False
    finally:
        storage.close()

    app.dependency_overrides.clear()


def test_scene_context_includes_expected_outcome_semantic_states_summary(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "local")
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    root_id, character, scenes = _seed_storage_with_scenes(storage, count=2)
    entity_id = str(character.entity_id)
    storage.set_entity_semantic_states(
        root_id=root_id,
        branch_id=DEFAULT_BRANCH_ID,
        entity_id=entity_id,
        semantic_states={"hp": "100%"},
    )
    storage.complete_scene(
        scene_id=str(scenes[0].id),
        branch_id=DEFAULT_BRANCH_ID,
        actual_outcome="Outcome 1 actual",
        summary="Scene 1 summary",
    )

    app.dependency_overrides[get_graph_storage] = lambda: storage
    client = TestClient(app)
    try:
        response = client.get(
            f"/api/v1/scenes/{scenes[1].id}/context",
            params={"branch_id": DEFAULT_BRANCH_ID},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["expected_outcome"] == "Outcome 2"
        assert data["summary"] == "Scene 1 summary"
        assert data["semantic_states"][entity_id] == {"hp": "100%"}
    finally:
        storage.close()
        app.dependency_overrides.clear()


def test_scene_context_requires_previous_summary(monkeypatch, tmp_path):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "local")
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    _root_id, _character, scenes = _seed_storage_with_scenes(storage, count=2)

    app.dependency_overrides[get_graph_storage] = lambda: storage
    client = TestClient(app)
    try:
        response = client.get(
            f"/api/v1/scenes/{scenes[1].id}/context",
            params={"branch_id": DEFAULT_BRANCH_ID},
        )
        assert response.status_code == 400
        assert "summary" in response.json()["detail"]
    finally:
        storage.close()
        app.dependency_overrides.clear()


def test_negotiation_websocket_removed(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "local")
    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/ws/negotiation"):
            pass
    assert excinfo.value.code in {1000, 404, 410}


def test_topone_generate_endpoint(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "local")
    class StubTopone(ToponeClient):
        async def generate_content(self, **kwargs):
            self.kwargs = kwargs
            return {"ok": True}

    stub = StubTopone(api_key="k")
    app.dependency_overrides[get_snowflake_manager] = lambda: None
    app.dependency_overrides[get_llm_engine] = lambda: DummyEngine()
    app.dependency_overrides[get_topone_client] = lambda: stub  # type: ignore[name-defined]

    client = TestClient(app)
    payload = {
        "model": "gemini-3-flash-preview",
        "system_instruction": "sys",
        "messages": [{"role": "user", "text": "hi"}],
        "generation_config": {"temperature": 0.5},
        "timeout": 15,
    }
    response = client.post("/api/v1/llm/topone/generate", json=payload)
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert stub.kwargs["model"] == "gemini-3-flash-preview"
    assert stub.kwargs["system_instruction"] == "sys"
    app.dependency_overrides.clear()


def test_topone_generate_rejects_missing_content_type(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "local")
    app.dependency_overrides[get_llm_engine] = lambda: DummyEngine()
    client = TestClient(app)

    response = client.post(
        "/api/v1/llm/topone/generate",
        data="not-json",
        headers={"Content-Type": "text/plain"},
    )
    assert response.status_code == 422
    app.dependency_overrides.clear()


def test_topone_generate_rejects_missing_message_text(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "local")
    app.dependency_overrides[get_llm_engine] = lambda: DummyEngine()
    client = TestClient(app)

    payload = {"messages": [{"role": "user"}]}
    response = client.post("/api/v1/llm/topone/generate", json=payload)
    assert response.status_code == 422
    app.dependency_overrides.clear()


def test_topone_generate_rejects_blank_message_text(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "local")
    app.dependency_overrides[get_llm_engine] = lambda: DummyEngine()
    client = TestClient(app)

    payload = {"messages": [{"role": "user", "text": ""}]}
    response = client.post("/api/v1/llm/topone/generate", json=payload)
    assert response.status_code == 422
    app.dependency_overrides.clear()



def test_logic_check_rejects_non_gemini(monkeypatch, tmp_path):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "local")
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    app.dependency_overrides[get_graph_storage] = lambda: storage
    client = TestClient(app)
    try:
        payload = {
            "outline_requirement": "outline",
            "world_state": {},
            "user_intent": "intent",
            "mode": "standard",
        }
        response = client.post("/api/v1/logic/check", json=payload)
        assert response.status_code == 400
        assert "gemini" in response.json()["detail"]
    finally:
        storage.close()
        app.dependency_overrides.clear()


def test_state_extract_rejects_non_gemini(monkeypatch, tmp_path):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "local")
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    app.dependency_overrides[get_graph_storage] = lambda: storage
    client = TestClient(app)
    try:
        payload = {"content": "text", "entity_ids": ["e1"]}
        response = client.post("/api/v1/state/extract", json=payload)
        assert response.status_code == 400
        assert "gemini" in response.json()["detail"]
    finally:
        storage.close()
        app.dependency_overrides.clear()


def test_logic_check_force_execute_marks_scene(monkeypatch, tmp_path):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "gemini")
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    root_id, _character, scene = _seed_storage(storage)

    class StubGateway:
        async def logic_check(self, payload):
            return LogicCheckResult(
                ok=True,
                mode=payload.mode,
                decision="execute",
                impact_level=ImpactLevel.NEGLIGIBLE,
            )

    app.dependency_overrides[get_graph_storage] = lambda: storage
    app.dependency_overrides[get_topone_gateway] = lambda: StubGateway()
    client = TestClient(app)
    try:
        payload = {
            "outline_requirement": "outline",
            "world_state": {},
            "user_intent": "intent",
            "mode": "force_execute",
            "root_id": root_id,
            "branch_id": DEFAULT_BRANCH_ID,
            "scene_id": str(scene.id),
            "force_reason": "force it",
        }
        response = client.post("/api/v1/logic/check", json=payload)
        assert response.status_code == 200

        snapshot = storage.get_root_snapshot(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        snapshot_scene = next(
            item for item in snapshot["scenes"] if item["id"] == str(scene.id)
        )
        assert snapshot_scene["logic_exception"] is True
        assert snapshot_scene["logic_exception_reason"] == "force it"
    finally:
        storage.close()
        app.dependency_overrides.clear()


def test_logic_check_skips_impact_for_logic_exception(monkeypatch, tmp_path):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "gemini")
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    root_id, _character, scenes = _seed_storage_with_scenes(storage, count=3)
    storage.mark_scene_logic_exception(
        root_id=root_id,
        branch_id=DEFAULT_BRANCH_ID,
        scene_id=str(scenes[0].id),
        reason="force",
    )

    class StubGateway:
        async def logic_check(self, payload):
            return LogicCheckResult(
                ok=True,
                mode=payload.mode,
                decision="execute",
                impact_level=ImpactLevel.CASCADING,
            )

    app.dependency_overrides[get_graph_storage] = lambda: storage
    app.dependency_overrides[get_topone_gateway] = lambda: StubGateway()
    client = TestClient(app)
    try:
        payload = {
            "outline_requirement": "outline",
            "world_state": {},
            "user_intent": "intent",
            "mode": "standard",
            "root_id": root_id,
            "branch_id": DEFAULT_BRANCH_ID,
            "scene_id": str(scenes[0].id),
        }
        response = client.post("/api/v1/logic/check", json=payload)
        assert response.status_code == 200
        dirty = storage.list_dirty_scenes(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        assert dirty == []
    finally:
        storage.close()
        app.dependency_overrides.clear()


def test_state_extract_enriches_diff_and_is_readonly(monkeypatch, tmp_path):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "gemini")
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    root_id, character, _scene = _seed_storage(storage)
    entity_id = str(character.entity_id)
    storage.set_entity_semantic_states(
        root_id=root_id,
        branch_id=DEFAULT_BRANCH_ID,
        entity_id=entity_id,
        semantic_states={"hp": "100%"},
    )

    class StubGateway:
        async def state_extract(self, payload):
            return [
                StateProposal(
                    entity_id=entity_id,
                    confidence=0.8,
                    semantic_states_patch={"hp": "20%"},
                )
            ]

    app.dependency_overrides[get_graph_storage] = lambda: storage
    app.dependency_overrides[get_topone_gateway] = lambda: StubGateway()
    client = TestClient(app)
    try:
        payload = {
            "content": "text",
            "entity_ids": [entity_id],
            "root_id": root_id,
            "branch_id": DEFAULT_BRANCH_ID,
        }
        response = client.post("/api/v1/state/extract", json=payload)
        assert response.status_code == 200

        result = response.json()[0]
        assert result["semantic_states_before"] == {"hp": "100%"}
        assert result["semantic_states_after"]["hp"] == "20%"

        current = storage.get_entity_semantic_states(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID, entity_id=entity_id
        )
        assert current == {"hp": "100%"}
    finally:
        storage.close()
        app.dependency_overrides.clear()


def test_state_commit_endpoint_updates_semantic_states(monkeypatch, tmp_path):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "local")
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    root_id, character, _scene = _seed_storage(storage)
    app.dependency_overrides[get_graph_storage] = lambda: storage
    client = TestClient(app)
    try:
        payload = [
            {
                "entity_id": str(character.entity_id),
                "confidence": 0.9,
                "semantic_states_patch": {"mood": "nervous"},
            }
        ]
        response = client.post(
            "/api/v1/state/commit",
            params={"root_id": root_id, "branch_id": DEFAULT_BRANCH_ID},
            json=payload,
        )
        assert response.status_code == 200
        assert response.json()["applied"] == 1
        updated = storage.get_entity_semantic_states(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            entity_id=str(character.entity_id),
        )
        assert updated["mood"] == "nervous"
    finally:
        storage.close()
        app.dependency_overrides.clear()


def test_complete_scene_orchestrated_endpoint_runs_flow(monkeypatch, tmp_path):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "gemini")
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    root_id, character, scenes = _seed_storage_with_scenes(storage, count=6)
    entity_id = str(character.entity_id)
    storage.set_entity_semantic_states(
        root_id=root_id,
        branch_id=DEFAULT_BRANCH_ID,
        entity_id=entity_id,
        semantic_states={"hp": "100%"},
    )
    original_snapshots = {
        str(scene.id): storage.get_scene_snapshot(
            scene_id=str(scene.id),
            branch_id=DEFAULT_BRANCH_ID,
        )
        for scene in scenes[1:]
    }

    class StubGateway:
        async def logic_check(self, payload):
            return LogicCheckResult(
                ok=True,
                mode=payload.mode,
                decision="execute",
                impact_level=ImpactLevel.LOCAL,
            )

        async def state_extract(self, payload):
            return [
                StateProposal(
                    entity_id=entity_id,
                    confidence=0.9,
                    semantic_states_patch={"hp": "50%"},
                )
            ]

    app.dependency_overrides[get_graph_storage] = lambda: storage
    app.dependency_overrides[get_topone_gateway] = lambda: StubGateway()
    client = TestClient(app)
    try:
        payload = {
            "root_id": root_id,
            "branch_id": DEFAULT_BRANCH_ID,
            "outline_requirement": "keep the plan",
            "world_state": {"hp": "100%"},
            "user_intent": "advance the plot",
            "mode": "standard",
            "content": "Hero loses strength.",
            "entity_ids": [entity_id],
            "confirmed_proposals": [
                {
                    "entity_id": entity_id,
                    "confidence": 0.9,
                    "semantic_states_patch": {"hp": "50%"},
                }
            ],
            "actual_outcome": "Hero survives",
            "summary": "Scene 1 summary",
        }
        response = client.post(
            f"/api/v1/scenes/{scenes[0].id}/complete/orchestrated",
            json=payload,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["applied"] == 1

        updated = storage.get_entity_semantic_states(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID, entity_id=entity_id
        )
        assert updated["hp"] == "50%"

        dirty = storage.list_dirty_scenes(root_id=root_id, branch_id=DEFAULT_BRANCH_ID)
        expected_dirty = {str(scene.id) for scene in scenes[1:4]}
        assert expected_dirty.issubset(set(dirty))

        for idx, scene in enumerate(scenes[1:], start=1):
            snapshot = storage.get_scene_snapshot(
                scene_id=str(scene.id),
                branch_id=DEFAULT_BRANCH_ID,
            )
            before = original_snapshots[str(scene.id)]
            if idx <= 3:
                assert "local_fix" in snapshot["expected_outcome"]
                assert snapshot["summary"].startswith("local_fix")
                assert snapshot["is_dirty"] is True
            else:
                assert snapshot["expected_outcome"] == before["expected_outcome"]
                assert snapshot["summary"] == before["summary"]
                assert snapshot["is_dirty"] == before["is_dirty"]

        context = client.get(
            f"/api/v1/scenes/{scenes[1].id}/context",
            params={"branch_id": DEFAULT_BRANCH_ID},
        )
        assert context.status_code == 200
        assert context.json()["summary"] == "Scene 1 summary"
    finally:
        storage.close()
        app.dependency_overrides.clear()


def test_complete_scene_orchestrated_schedules_impact_background_task(monkeypatch, tmp_path):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "gemini")
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    root_id, character, scenes = _seed_storage_with_scenes(storage, count=3)
    entity_id = str(character.entity_id)
    storage.set_entity_semantic_states(
        root_id=root_id,
        branch_id=DEFAULT_BRANCH_ID,
        entity_id=entity_id,
        semantic_states={"hp": "100%"},
    )

    called = {"func": None}
    original_add_task = BackgroundTasks.add_task

    def spy_add_task(self, func, *args, **kwargs):
        called["func"] = func
        return original_add_task(self, func, *args, **kwargs)

    class StubGateway:
        async def logic_check(self, payload):
            return LogicCheckResult(
                ok=True,
                mode=payload.mode,
                decision="execute",
                impact_level=ImpactLevel.LOCAL,
            )

        async def state_extract(self, payload):
            return [
                StateProposal(
                    entity_id=entity_id,
                    confidence=0.9,
                    semantic_states_patch={"hp": "50%"},
                )
            ]

    monkeypatch.setattr(BackgroundTasks, "add_task", spy_add_task)
    app.dependency_overrides[get_graph_storage] = lambda: storage
    app.dependency_overrides[get_topone_gateway] = lambda: StubGateway()
    client = TestClient(app)
    try:
        payload = {
            "root_id": root_id,
            "branch_id": DEFAULT_BRANCH_ID,
            "outline_requirement": "keep the plan",
            "world_state": {"hp": "100%"},
            "user_intent": "advance the plot",
            "mode": "standard",
            "content": "Hero loses strength.",
            "entity_ids": [entity_id],
            "confirmed_proposals": [
                {
                    "entity_id": entity_id,
                    "confidence": 0.9,
                    "semantic_states_patch": {"hp": "50%"},
                }
            ],
            "actual_outcome": "Hero survives",
            "summary": "Scene 1 summary",
        }
        response = client.post(
            f"/api/v1/scenes/{scenes[0].id}/complete/orchestrated",
            json=payload,
        )
        assert response.status_code == 200
        assert called["func"] is not None
        assert called["func"].__name__ == "_apply_impact_level"
    finally:
        storage.close()
        app.dependency_overrides.clear()


def test_complete_scene_orchestrated_skips_impact_for_logic_exception(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "gemini")
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    root_id, character, scenes = _seed_storage_with_scenes(storage, count=4)
    entity_id = str(character.entity_id)
    storage.set_entity_semantic_states(
        root_id=root_id,
        branch_id=DEFAULT_BRANCH_ID,
        entity_id=entity_id,
        semantic_states={"hp": "100%"},
    )
    storage.mark_scene_logic_exception(
        root_id=root_id,
        branch_id=DEFAULT_BRANCH_ID,
        scene_id=str(scenes[0].id),
        reason="force",
    )
    original_snapshots = {
        str(scene.id): storage.get_scene_snapshot(
            scene_id=str(scene.id),
            branch_id=DEFAULT_BRANCH_ID,
        )
        for scene in scenes[1:]
    }

    class StubGateway:
        async def logic_check(self, payload):
            return LogicCheckResult(
                ok=True,
                mode=payload.mode,
                decision="execute",
                impact_level=ImpactLevel.CASCADING,
            )

        async def state_extract(self, payload):
            return [
                StateProposal(
                    entity_id=entity_id,
                    confidence=0.9,
                    semantic_states_patch={"hp": "50%"},
                )
            ]

    app.dependency_overrides[get_graph_storage] = lambda: storage
    app.dependency_overrides[get_topone_gateway] = lambda: StubGateway()
    client = TestClient(app)
    try:
        payload = {
            "root_id": root_id,
            "branch_id": DEFAULT_BRANCH_ID,
            "outline_requirement": "keep the plan",
            "world_state": {"hp": "100%"},
            "user_intent": "advance the plot",
            "mode": "standard",
            "content": "Hero loses strength.",
            "entity_ids": [entity_id],
            "confirmed_proposals": [
                {
                    "entity_id": entity_id,
                    "confidence": 0.9,
                    "semantic_states_patch": {"hp": "50%"},
                }
            ],
            "actual_outcome": "Hero survives",
            "summary": "Scene 1 summary",
        }
        response = client.post(
            f"/api/v1/scenes/{scenes[0].id}/complete/orchestrated",
            json=payload,
        )
        assert response.status_code == 200

        dirty = storage.list_dirty_scenes(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        assert dirty == []

        for scene in scenes[1:]:
            snapshot = storage.get_scene_snapshot(
                scene_id=str(scene.id),
                branch_id=DEFAULT_BRANCH_ID,
            )
            before = original_snapshots[str(scene.id)]
            assert snapshot["expected_outcome"] == before["expected_outcome"]
            assert snapshot["summary"] == before["summary"]
            assert snapshot["is_dirty"] == before["is_dirty"]
    finally:
        storage.close()
        app.dependency_overrides.clear()


def test_complete_scene_orchestrated_allows_empty_confirmed_proposals(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "gemini")
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    root_id, character, scenes = _seed_storage_with_scenes(storage, count=3)
    entity_id = str(character.entity_id)
    storage.set_entity_semantic_states(
        root_id=root_id,
        branch_id=DEFAULT_BRANCH_ID,
        entity_id=entity_id,
        semantic_states={"hp": "100%"},
    )

    class StubGateway:
        async def logic_check(self, payload):
            return LogicCheckResult(
                ok=True,
                mode=payload.mode,
                decision="execute",
                impact_level=ImpactLevel.NEGLIGIBLE,
            )

        async def state_extract(self, payload):
            return [
                StateProposal(
                    entity_id=entity_id,
                    confidence=0.6,
                    semantic_states_patch={"hp": "50%"},
                )
            ]

    app.dependency_overrides[get_graph_storage] = lambda: storage
    app.dependency_overrides[get_topone_gateway] = lambda: StubGateway()
    client = TestClient(app)
    try:
        payload = {
            "root_id": root_id,
            "branch_id": DEFAULT_BRANCH_ID,
            "outline_requirement": "keep the plan",
            "world_state": {"hp": "100%"},
            "user_intent": "advance the plot",
            "mode": "standard",
            "content": "Hero loses strength.",
            "entity_ids": [entity_id],
            "confirmed_proposals": [],
            "actual_outcome": "Hero survives",
            "summary": "Scene 1 summary",
        }
        response = client.post(
            f"/api/v1/scenes/{scenes[0].id}/complete/orchestrated",
            json=payload,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["confirmed_count"] == 0
        assert data["applied"] == 0
        assert data["updated_entities"] == []

        current = storage.get_entity_semantic_states(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID, entity_id=entity_id
        )
        assert current == {"hp": "100%"}

        snapshot = storage.get_scene_snapshot(
            scene_id=str(scenes[0].id),
            branch_id=DEFAULT_BRANCH_ID,
        )
        assert snapshot["summary"] == "Scene 1 summary"
    finally:
        storage.close()
        app.dependency_overrides.clear()



def test_complete_scene_orchestrated_force_execute_allows_rejected_logic(monkeypatch, tmp_path):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "gemini")
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    root_id, character, scenes = _seed_storage_with_scenes(storage, count=4)
    entity_id = str(character.entity_id)
    storage.set_entity_semantic_states(
        root_id=root_id,
        branch_id=DEFAULT_BRANCH_ID,
        entity_id=entity_id,
        semantic_states={"hp": "100%"},
    )

    class StubGateway:
        async def logic_check(self, payload):
            return LogicCheckResult(
                ok=False,
                mode=payload.mode,
                decision="review",
                impact_level=ImpactLevel.LOCAL,
            )

        async def state_extract(self, payload):
            return [
                StateProposal(
                    entity_id=entity_id,
                    confidence=0.9,
                    semantic_states_patch={"hp": "50%"},
                )
            ]

    app.dependency_overrides[get_graph_storage] = lambda: storage
    app.dependency_overrides[get_topone_gateway] = lambda: StubGateway()
    client = TestClient(app)
    try:
        payload = {
            "root_id": root_id,
            "branch_id": DEFAULT_BRANCH_ID,
            "outline_requirement": "keep the plan",
            "world_state": {"hp": "100%"},
            "user_intent": "advance the plot",
            "mode": "force_execute",
            "force_reason": "force it",
            "content": "Hero loses strength.",
            "entity_ids": [entity_id],
            "confirmed_proposals": [
                {
                    "entity_id": entity_id,
                    "confidence": 0.9,
                    "semantic_states_patch": {"hp": "50%"},
                }
            ],
            "actual_outcome": "Hero survives",
            "summary": "Scene 1 summary",
        }
        response = client.post(
            f"/api/v1/scenes/{scenes[0].id}/complete/orchestrated",
            json=payload,
        )
        assert response.status_code == 200
        assert response.json()["logic_check"]["ok"] is False

        snapshot = storage.get_root_snapshot(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        snapshot_scene = next(
            item for item in snapshot["scenes"] if item["id"] == str(scenes[0].id)
        )
        assert snapshot_scene["logic_exception"] is True
        assert snapshot_scene["logic_exception_reason"] == "force it"
    finally:
        storage.close()
        app.dependency_overrides.clear()


def test_scene_render_endpoint_persists_content(monkeypatch, tmp_path):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "gemini")
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    _root_id, character, scene = _seed_storage(storage)

    class StubGateway:
        async def render_scene(self, payload):
            return "Rendered text"

    app.dependency_overrides[get_graph_storage] = lambda: storage
    app.dependency_overrides[get_topone_gateway] = lambda: StubGateway()
    client = TestClient(app)
    try:
        payload = {
            "voice_dna": character.voice_dna,
            "conflict_type": scene.conflict_type,
            "outline_requirement": "keep the plan",
            "user_intent": "advance the plot",
            "expected_outcome": scene.expected_outcome,
            "world_state": {"hp": "100%"},
        }
        response = client.post(
            f"/api/v1/scenes/{scene.id}/render",
            params={"branch_id": DEFAULT_BRANCH_ID},
            json=payload,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "Rendered text"
        snapshot = storage.get_scene_snapshot(
            scene_id=str(scene.id),
            branch_id=DEFAULT_BRANCH_ID,
        )
        assert snapshot["rendered_content"] == "Rendered text"
    finally:
        storage.close()
        app.dependency_overrides.clear()


def test_complete_scene_orchestrated_marks_cascading_dirty(monkeypatch, tmp_path):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "gemini")
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    root_id, character, scenes = _seed_storage_with_scenes(storage, count=5)
    entity_id = str(character.entity_id)
    storage.set_entity_semantic_states(
        root_id=root_id,
        branch_id=DEFAULT_BRANCH_ID,
        entity_id=entity_id,
        semantic_states={"hp": "100%"},
    )

    called = {"state_extract": False}

    class StubGateway:
        async def logic_check(self, payload):
            return LogicCheckResult(
                ok=True,
                mode=payload.mode,
                decision="execute",
                impact_level=ImpactLevel.CASCADING,
            )

        async def state_extract(self, payload):
            called["state_extract"] = True
            return [
                StateProposal(
                    entity_id=entity_id,
                    confidence=0.9,
                    semantic_states_patch={"hp": "50%"},
                )
            ]

    app.dependency_overrides[get_graph_storage] = lambda: storage
    app.dependency_overrides[get_topone_gateway] = lambda: StubGateway()
    client = TestClient(app)
    try:
        payload = {
            "root_id": root_id,
            "branch_id": DEFAULT_BRANCH_ID,
            "outline_requirement": "keep the plan",
            "world_state": {"hp": "100%"},
            "user_intent": "advance the plot",
            "mode": "standard",
            "content": "Hero loses strength.",
            "entity_ids": [entity_id],
            "confirmed_proposals": [
                {
                    "entity_id": entity_id,
                    "confidence": 0.9,
                    "semantic_states_patch": {"hp": "50%"},
                }
            ],
            "actual_outcome": "Hero survives",
            "summary": "Scene 1 summary",
        }
        response = client.post(
            f"/api/v1/scenes/{scenes[0].id}/complete/orchestrated",
            json=payload,
        )
        assert response.status_code == 200
        assert called["state_extract"] is True

        dirty = storage.list_dirty_scenes(root_id=root_id, branch_id=DEFAULT_BRANCH_ID)
        expected_dirty = {str(scene.id) for scene in scenes[1:]}
        assert expected_dirty.issubset(set(dirty))
    finally:
        storage.close()
        app.dependency_overrides.clear()



def test_relation_upsert_endpoint_creates_and_updates(monkeypatch, tmp_path):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "local")
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    root_id, character, _scene = _seed_storage(storage)
    other_entity_id = storage.create_entity(
        root_id=root_id,
        branch_id=DEFAULT_BRANCH_ID,
        name="Villain",
        entity_type="npc",
        tags=[],
        arc_status=None,
        semantic_states={},
    )

    app.dependency_overrides[get_graph_storage] = lambda: storage
    client = TestClient(app)
    try:
        payload = {
            "from_entity_id": str(character.entity_id),
            "to_entity_id": other_entity_id,
            "relation_type": "enemy",
            "tension": 30,
        }
        response = client.post(
            f"/api/v1/roots/{root_id}/relations",
            params={"branch_id": DEFAULT_BRANCH_ID},
            json=payload,
        )
        assert response.status_code == 200
        assert response.json()["tension"] == 30

        snapshot = storage.get_root_snapshot(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        relation = next(
            item
            for item in snapshot["relations"]
            if item["from_entity_id"] == str(character.entity_id)
            and item["to_entity_id"] == other_entity_id
            and item["relation_type"] == "enemy"
        )
        assert relation["tension"] == 30

        payload["tension"] = 70
        response = client.post(
            f"/api/v1/roots/{root_id}/relations",
            params={"branch_id": DEFAULT_BRANCH_ID},
            json=payload,
        )
        assert response.status_code == 200
        assert response.json()["tension"] == 70

        snapshot = storage.get_root_snapshot(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        relation = next(
            item
            for item in snapshot["relations"]
            if item["from_entity_id"] == str(character.entity_id)
            and item["to_entity_id"] == other_entity_id
            and item["relation_type"] == "enemy"
        )
        assert relation["tension"] == 70
    finally:
        storage.close()
        app.dependency_overrides.clear()


def test_dirty_endpoints_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "local")
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    root_id, _character, scene = _seed_storage(storage)
    app.dependency_overrides[get_graph_storage] = lambda: storage
    client = TestClient(app)
    try:
        mark = client.post(
            f"/api/v1/scenes/{scene.id}/dirty",
            params={"branch_id": DEFAULT_BRANCH_ID},
        )
        assert mark.status_code == 200

        listing = client.get(
            f"/api/v1/roots/{root_id}/dirty_scenes",
            params={"branch_id": DEFAULT_BRANCH_ID},
        )
        assert listing.status_code == 200
        assert str(scene.id) in listing.json()
    finally:
        storage.close()
        app.dependency_overrides.clear()


def test_branch_endpoints_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "local")
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    root_id, _character, _scene = _seed_storage(storage)
    app.dependency_overrides[get_graph_storage] = lambda: storage
    client = TestClient(app)
    try:
        listing = client.get(f"/api/v1/roots/{root_id}/branches")
        assert listing.status_code == 200
        assert listing.json() == [DEFAULT_BRANCH_ID]

        created = client.post(
            f"/api/v1/roots/{root_id}/branches", json={"branch_id": "dev"}
        )
        assert created.status_code == 200
        assert created.json()["branch_id"] == "dev"

        listing = client.get(f"/api/v1/roots/{root_id}/branches")
        assert listing.status_code == 200
        assert set(listing.json()) == {DEFAULT_BRANCH_ID, "dev"}

        switched = client.post(f"/api/v1/roots/{root_id}/branches/dev/switch")
        assert switched.status_code == 200
        assert switched.json()["branch_id"] == "dev"
    finally:
        storage.close()
        app.dependency_overrides.clear()


def test_branch_duplicate_and_switch_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "local")
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    root_id, _character, _scene = _seed_storage(storage)
    app.dependency_overrides[get_graph_storage] = lambda: storage
    client = TestClient(app)
    try:
        created = client.post(
            f"/api/v1/roots/{root_id}/branches", json={"branch_id": "dev"}
        )
        assert created.status_code == 200

        duplicate = client.post(
            f"/api/v1/roots/{root_id}/branches", json={"branch_id": "dev"}
        )
        assert duplicate.status_code == 409
        assert "already exists" in duplicate.json()["detail"]

        missing = client.post(f"/api/v1/roots/{root_id}/branches/ghost/switch")
        assert missing.status_code == 404
        assert "not found" in missing.json()["detail"]
    finally:
        storage.close()
        app.dependency_overrides.clear()


def test_branch_merge_endpoint_applies_snapshot(monkeypatch, tmp_path):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "local")
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    root_id, _character, scene = _seed_storage(storage)
    app.dependency_overrides[get_graph_storage] = lambda: storage
    client = TestClient(app)
    try:
        created = client.post(
            f"/api/v1/roots/{root_id}/branches", json={"branch_id": "dev"}
        )
        assert created.status_code == 200

        branch_snapshot = client.get(
            f"/api/v1/roots/{root_id}", params={"branch_id": "dev"}
        )
        assert branch_snapshot.status_code == 200
        branch_scene_id = branch_snapshot.json()["scenes"][0]["id"]
        completed = client.post(
            f"/api/v1/scenes/{branch_scene_id}/complete",
            params={"branch_id": "dev"},
            json={"actual_outcome": "branch outcome", "summary": "branch summary"},
        )
        assert completed.status_code == 200

        merged = client.post(f"/api/v1/roots/{root_id}/branches/dev/merge")
        assert merged.status_code == 200

        listing = client.get(f"/api/v1/roots/{root_id}/branches")
        assert listing.status_code == 200
        assert listing.json() == [DEFAULT_BRANCH_ID]

        main_snapshot = client.get(
            f"/api/v1/roots/{root_id}", params={"branch_id": DEFAULT_BRANCH_ID}
        )
        assert main_snapshot.status_code == 200
        main_scene = next(
            item
            for item in main_snapshot.json()["scenes"]
            if item["id"] == str(scene.id)
        )
        assert main_scene["actual_outcome"] == "branch outcome"
    finally:
        storage.close()
        app.dependency_overrides.clear()


def test_branch_revert_endpoint_discards_branch(monkeypatch, tmp_path):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "local")
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    root_id, _character, scene = _seed_storage(storage)
    app.dependency_overrides[get_graph_storage] = lambda: storage
    client = TestClient(app)
    try:
        created = client.post(
            f"/api/v1/roots/{root_id}/branches", json={"branch_id": "dev"}
        )
        assert created.status_code == 200

        branch_snapshot = client.get(
            f"/api/v1/roots/{root_id}", params={"branch_id": "dev"}
        )
        assert branch_snapshot.status_code == 200
        branch_scene_id = branch_snapshot.json()["scenes"][0]["id"]
        completed_branch = client.post(
            f"/api/v1/scenes/{branch_scene_id}/complete",
            params={"branch_id": "dev"},
            json={"actual_outcome": "branch outcome", "summary": "branch summary"},
        )
        assert completed_branch.status_code == 200
        completed_main = client.post(
            f"/api/v1/scenes/{scene.id}/complete",
            params={"branch_id": DEFAULT_BRANCH_ID},
            json={"actual_outcome": "main outcome", "summary": "main summary"},
        )
        assert completed_main.status_code == 200

        reverted = client.post(f"/api/v1/roots/{root_id}/branches/dev/revert")
        assert reverted.status_code == 200

        listing = client.get(f"/api/v1/roots/{root_id}/branches")
        assert listing.status_code == 200
        assert listing.json() == [DEFAULT_BRANCH_ID]

        main_snapshot = client.get(
            f"/api/v1/roots/{root_id}", params={"branch_id": DEFAULT_BRANCH_ID}
        )
        assert main_snapshot.status_code == 200
        main_scene = next(
            item
            for item in main_snapshot.json()["scenes"]
            if item["id"] == str(scene.id)
        )
        assert main_scene["actual_outcome"] == "main outcome"
    finally:
        storage.close()
        app.dependency_overrides.clear()


def test_commit_graph_endpoints_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "local")
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    root_id, _character, scene = _seed_storage(storage)
    app.dependency_overrides[get_graph_storage] = lambda: storage
    client = TestClient(app)
    try:
        initial_head = storage.get_branch_head(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        commit_payload = {
            "scene_origin_id": str(scene.id),
            "content": {
                "actual_outcome": "Outcome A",
                "summary": "Summary A",
                "status": "committed",
            },
            "message": "commit A",
        }
        commit_resp = client.post(
            f"/api/v1/roots/{root_id}/branches/{DEFAULT_BRANCH_ID}/commit",
            json=commit_payload,
        )
        assert commit_resp.status_code == 200
        commit_id = commit_resp.json()["commit_id"]

        history_resp = client.get(
            f"/api/v1/roots/{root_id}/branches/{DEFAULT_BRANCH_ID}/history",
            params={"limit": 1},
        )
        assert history_resp.status_code == 200
        assert history_resp.json()[0]["id"] == commit_id

        diff_resp = client.get(
            f"/api/v1/scenes/{scene.id}/diff",
            params={
                "from_commit_id": initial_head["head_commit_id"],
                "to_commit_id": commit_id,
            },
        )
        assert diff_resp.status_code == 200
        assert diff_resp.json()["actual_outcome"]["to"] == "Outcome A"

        reset_resp = client.post(
            f"/api/v1/roots/{root_id}/branches/{DEFAULT_BRANCH_ID}/reset",
            json={"commit_id": initial_head["head_commit_id"]},
        )
        assert reset_resp.status_code == 200
        head_after = storage.get_branch_head(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        assert head_after["head_commit_id"] == initial_head["head_commit_id"]

        fork_resp = client.post(
            f"/api/v1/roots/{root_id}/branches/fork_from_commit",
            json={"source_commit_id": initial_head["head_commit_id"], "new_branch_id": "dev"},
        )
        assert fork_resp.status_code == 200
        branch_head = storage.get_branch_head(root_id=root_id, branch_id="dev")
        assert branch_head["head_commit_id"] == initial_head["head_commit_id"]
    finally:
        storage.close()
        app.dependency_overrides.clear()


def test_scene_origin_create_and_delete_endpoints(monkeypatch, tmp_path):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "local")
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    root_id, character, _scene = _seed_storage(storage)
    app.dependency_overrides[get_graph_storage] = lambda: storage
    client = TestClient(app)
    try:
        tree = storage.list_structure_tree(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        act_id = tree["acts"][0]["act_id"]
        create_payload = {
            "title": "New scene",
            "parent_act_id": act_id,
            "content": {
                "pov_character_id": str(character.entity_id),
                "expected_outcome": "Outcome X",
                "conflict_type": "internal",
                "actual_outcome": "",
            },
        }
        created = client.post(
            f"/api/v1/roots/{root_id}/scene_origins",
            params={"branch_id": DEFAULT_BRANCH_ID},
            json=create_payload,
        )
        assert created.status_code == 200
        scene_origin_id = created.json()["scene_origin_id"]

        deleted = client.post(
            f"/api/v1/roots/{root_id}/scenes/{scene_origin_id}/delete",
            params={"branch_id": DEFAULT_BRANCH_ID},
            json={"message": "archive scene"},
        )
        assert deleted.status_code == 200
        head = storage.get_branch_head(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        version = storage.get_scene_at_commit(
            scene_origin_id=scene_origin_id, commit_id=head["head_commit_id"]
        )
        assert version["status"] == "archived"
    finally:
        storage.close()
        app.dependency_overrides.clear()


def test_commit_gc_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("SNOWFLAKE_ENGINE", "local")
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    root_id, _character, scene = _seed_storage(storage)
    app.dependency_overrides[get_graph_storage] = lambda: storage
    client = TestClient(app)
    try:
        initial_head = storage.get_branch_head(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        commit_a = storage.commit_scene(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            scene_origin_id=str(scene.id),
            content={"actual_outcome": "A", "summary": "A", "status": "committed"},
            message="commit A",
        )
        commit_b = storage.commit_scene(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            scene_origin_id=str(scene.id),
            content={"actual_outcome": "B", "summary": "B", "status": "committed"},
            message="commit B",
        )
        storage.reset_branch_head(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            commit_id=initial_head["head_commit_id"],
        )

        gc_resp = client.post(
            "/api/v1/commits/gc", json={"retention_days": 0}
        )
        assert gc_resp.status_code == 200
        deleted = set(gc_resp.json()["deleted_commit_ids"])
        assert commit_a["commit_id"] in deleted
        assert commit_b["commit_id"] in deleted
    finally:
        storage.close()
        app.dependency_overrides.clear()
