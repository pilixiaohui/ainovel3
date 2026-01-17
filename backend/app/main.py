"""FastAPI 入口，暴露雪花流程接口。"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, List
from pathlib import Path

from fastapi import BackgroundTasks, Body, Depends, FastAPI, HTTPException, Query, Path
from pydantic import BaseModel, Field

from app.llm.schemas import (
    ImpactLevel,
    LogicCheckPayload,
    LogicCheckResult,
    SceneRenderPayload,
    StateExtractPayload,
    StateProposal,
    ToponeGeneratePayload,
)
from app.llm.topone_gateway import ToponeGateway
from app.config import SCENE_MAX_COUNT, SCENE_MIN_COUNT
from app.logic.snowflake_manager import SnowflakeManager
from app.models import CharacterSheet, Commit, SceneNode, SnowflakeRoot
from app.services.llm_engine import LLMEngine, LocalStoryEngine
from app.services.topone_client import ToponeClient
from app.constants import DEFAULT_BRANCH_ID
from app.storage.graph import GraphStorage

app = FastAPI(title="Snowflake Engine API", version="0.1.0")

_ALLOWED_SNOWFLAKE_ENGINES = {"local", "llm", "gemini"}


def _require_snowflake_engine_mode() -> str:
    raw = os.getenv("SNOWFLAKE_ENGINE")
    if raw is None:
        raise RuntimeError(
            "SNOWFLAKE_ENGINE 未配置：必须显式设置为 local/llm/gemini（例如：SNOWFLAKE_ENGINE=local）。"
        )
    mode = raw.strip().lower()
    if mode not in _ALLOWED_SNOWFLAKE_ENGINES:
        raise RuntimeError(f"SNOWFLAKE_ENGINE={raw!r} 非法：必须为 local/llm/gemini。")
    return mode


@app.on_event("startup")
async def _validate_snowflake_engine_config() -> None:
    _require_snowflake_engine_mode()


class IdeaPayload(BaseModel):
    idea: str


class LoglinePayload(BaseModel):
    logline: str


class ScenePayload(BaseModel):
    root: SnowflakeRoot
    characters: List[CharacterSheet] = Field(default_factory=list)


class Step4Result(BaseModel):
    root_id: str
    branch_id: str
    scenes: List[SceneNode]


class BranchPayload(BaseModel):
    branch_id: str = Field(..., min_length=1)


class BranchView(BaseModel):
    root_id: str
    branch_id: str


class CreateEntityPayload(BaseModel):
    name: str
    entity_type: str
    tags: list[str] = Field(default_factory=list)
    arc_status: str | None = None
    semantic_states: dict[str, Any] = Field(default_factory=dict)


class UpsertRelationPayload(BaseModel):
    from_entity_id: str = Field(..., min_length=1)
    to_entity_id: str = Field(..., min_length=1)
    relation_type: str = Field(..., min_length=1)
    tension: int = Field(..., ge=0, le=100)


class EntityView(BaseModel):
    entity_id: str
    name: str | None = None
    entity_type: str | None = None
    tags: list[str] = Field(default_factory=list)
    arc_status: str | None = None
    semantic_states: dict[str, Any] = Field(default_factory=dict)


class CharacterView(BaseModel):
    entity_id: str
    name: str | None = None


class EntityRelationView(BaseModel):
    from_entity_id: str
    to_entity_id: str
    relation_type: str
    tension: int


class SceneView(BaseModel):
    id: str
    branch_id: str
    status: str | None = None
    pov_character_id: str | None = None
    expected_outcome: str | None = None
    conflict_type: str | None = None
    actual_outcome: str
    logic_exception: bool | None = None
    logic_exception_reason: str | None = None
    is_dirty: bool


class RootGraphView(BaseModel):
    root_id: str
    branch_id: str
    logline: str | None = None
    theme: str | None = None
    ending: str | None = None
    characters: List[CharacterView] = Field(default_factory=list)
    scenes: List[SceneView] = Field(default_factory=list)
    relations: List[EntityRelationView] = Field(default_factory=list)


class StructureTreeActView(BaseModel):
    act_id: str
    act_index: int | None = None
    disaster: str | None = None
    scene_ids: List[str] = Field(default_factory=list)


class StructureTreeView(BaseModel):
    root_id: str
    branch_id: str
    acts: List[StructureTreeActView] = Field(default_factory=list)


class SceneReorderPayload(BaseModel):
    branch_id: str = Field(..., min_length=1)
    scene_ids: List[str] = Field(..., min_length=1)


class SceneReorderResult(BaseModel):
    ok: bool
    root_id: str
    branch_id: str
    scene_ids: List[str] = Field(default_factory=list)


class SceneContextView(BaseModel):
    root_id: str
    branch_id: str
    expected_outcome: str
    semantic_states: dict[str, Any]
    summary: str
    scene_entities: List[EntityView] = Field(default_factory=list)
    characters: List[CharacterView] = Field(default_factory=list)
    relations: List[EntityRelationView] = Field(default_factory=list)
    prev_scene_id: str | None = None
    next_scene_id: str | None = None


class SceneCompletePayload(BaseModel):
    actual_outcome: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)


class SceneCompletionOrchestratePayload(BaseModel):
    root_id: str = Field(..., min_length=1)
    branch_id: str = Field(..., min_length=1)
    outline_requirement: str = Field(..., min_length=1)
    world_state: dict[str, Any] = Field(default_factory=dict)
    user_intent: str = Field(..., min_length=1)
    mode: str = Field(..., min_length=1)
    force_reason: str | None = Field(default=None, min_length=1)
    content: str = Field(..., min_length=1)
    entity_ids: List[str] = Field(..., min_length=1)
    confirmed_proposals: List[StateProposal] = Field(...)
    actual_outcome: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)


class SceneCompletionResult(BaseModel):
    ok: bool
    scene_id: str
    root_id: str
    branch_id: str
    status: str
    actual_outcome: str
    summary: str
    logic_check: LogicCheckResult
    extracted_proposals: List[StateProposal]
    confirmed_count: int
    applied: int
    updated_entities: List[dict[str, Any]]


class SceneRenderResult(BaseModel):
    ok: bool
    scene_id: str
    branch_id: str
    content: str


class ForkFromCommitPayload(BaseModel):
    source_commit_id: str = Field(..., min_length=1)
    new_branch_id: str = Field(..., min_length=1)
    parent_branch_id: str | None = None


class ForkFromScenePayload(BaseModel):
    source_branch_id: str = Field(..., min_length=1)
    scene_origin_id: str = Field(..., min_length=1)
    new_branch_id: str = Field(..., min_length=1)
    commit_id: str | None = None


class ResetBranchPayload(BaseModel):
    commit_id: str = Field(..., min_length=1)


class CommitScenePayload(BaseModel):
    scene_origin_id: str = Field(..., min_length=1)
    content: dict[str, Any] = Field(default_factory=dict)
    message: str = Field(..., min_length=1)
    expected_head_version: int | None = None


class CommitResult(BaseModel):
    commit_id: str
    scene_version_ids: List[str] = Field(default_factory=list)


class CreateSceneOriginPayload(BaseModel):
    title: str = Field(..., min_length=1)
    parent_act_id: str = Field(..., min_length=1)
    content: dict[str, Any] = Field(default_factory=dict)


class CreateSceneOriginResult(BaseModel):
    commit_id: str
    scene_origin_id: str
    scene_version_id: str


class DeleteSceneOriginPayload(BaseModel):
    message: str = Field(..., min_length=1)


class GcPayload(BaseModel):
    retention_days: int = Field(..., ge=0)


class GcResult(BaseModel):
    deleted_commit_ids: List[str] = Field(default_factory=list)
    deleted_scene_version_ids: List[str] = Field(default_factory=list)


def get_llm_engine() -> LLMEngine | LocalStoryEngine | ToponeGateway:
    """默认依赖注入，可在测试中 override。"""
    engine_mode = _require_snowflake_engine_mode()
    if engine_mode == "local":
        return LocalStoryEngine()
    if engine_mode == "llm":
        return LLMEngine()
    if engine_mode == "gemini":
        return get_topone_gateway()
    raise RuntimeError("unreachable")


@lru_cache(maxsize=1)
def get_graph_storage() -> GraphStorage:
    """GraphStorage 单例，避免重复建立连接。"""
    repo_root = Path(__file__).resolve().parents[2]
    default_db_path = repo_root / "backend" / "data" / "snowflake.db"
    env_path = os.getenv("KUZU_DB_PATH")
    if env_path:
        candidate = Path(env_path)
        db_path = candidate if candidate.is_absolute() else repo_root / candidate
    else:
        db_path = default_db_path
    return GraphStorage(db_path=db_path)


def get_snowflake_manager(
    engine: LLMEngine | LocalStoryEngine | ToponeGateway = Depends(get_llm_engine),
    storage: GraphStorage = Depends(get_graph_storage),
) -> SnowflakeManager:
    """默认使用严格场景数量校验，可在测试 override。"""
    return SnowflakeManager(
        engine=engine,
        storage=storage,
        min_scenes=SCENE_MIN_COUNT,
        max_scenes=SCENE_MAX_COUNT,
    )


@lru_cache(maxsize=1)
def get_topone_client() -> ToponeClient:
    """TopOne Gemini 客户端单例，读取 .env 配置。"""
    return ToponeClient()


@lru_cache(maxsize=1)
def get_topone_gateway() -> ToponeGateway:
    """TopOne 统一网关单例：结构化输出校验入口。"""
    return ToponeGateway(get_topone_client())


@app.post("/api/v1/snowflake/step2", response_model=SnowflakeRoot)
async def generate_structure_endpoint(
    payload: LoglinePayload,
    engine: LLMEngine | LocalStoryEngine | ToponeGateway = Depends(get_llm_engine),
) -> SnowflakeRoot:
    return await engine.generate_root_structure(payload.logline)


@app.post("/api/v1/snowflake/step1", response_model=List[str])
async def generate_loglines_endpoint(
    payload: IdeaPayload, manager: SnowflakeManager = Depends(get_snowflake_manager)
) -> List[str]:
    return await manager.execute_step_1_logline(payload.idea)


@app.post("/api/v1/snowflake/step3", response_model=List[CharacterSheet])
async def generate_characters_endpoint(
    root: SnowflakeRoot, manager: SnowflakeManager = Depends(get_snowflake_manager)
) -> List[CharacterSheet]:
    return await manager.execute_step_3_characters(root)


@app.post("/api/v1/snowflake/step4", response_model=Step4Result)
async def generate_scene_endpoint(
    payload: ScenePayload, manager: SnowflakeManager = Depends(get_snowflake_manager)
) -> Step4Result:
    scenes = await manager.execute_step_4_scenes(payload.root, payload.characters)
    if not manager.last_persisted_root_id:
        raise HTTPException(status_code=500, detail="step4 did not persist root_id")
    return Step4Result(
        root_id=manager.last_persisted_root_id,
        branch_id=DEFAULT_BRANCH_ID,
        scenes=scenes,
    )


@app.post("/api/v1/roots/{root_id}/branches", response_model=BranchView)
async def create_branch_endpoint(
    root_id: str,
    payload: BranchPayload,
    storage: GraphStorage = Depends(get_graph_storage),
) -> BranchView:
    try:
        storage.create_branch(root_id=root_id, branch_id=payload.branch_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        detail = str(exc)
        status_code = 409 if "already exists" in detail else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc
    return BranchView(root_id=root_id, branch_id=payload.branch_id)


@app.get("/api/v1/roots/{root_id}/branches", response_model=List[str])
async def list_branches_endpoint(
    root_id: str, storage: GraphStorage = Depends(get_graph_storage)
) -> List[str]:
    try:
        return storage.list_branches(root_id=root_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/v1/roots/{root_id}/branches/{branch_id}/switch", response_model=BranchView)
async def switch_branch_endpoint(
    root_id: str,
    branch_id: str = Path(..., min_length=1),
    storage: GraphStorage = Depends(get_graph_storage),
) -> BranchView:
    try:
        storage.require_branch(root_id=root_id, branch_id=branch_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BranchView(root_id=root_id, branch_id=branch_id)


@app.post(
    "/api/v1/roots/{root_id}/branches/{branch_id}/merge",
    response_model=BranchView,
)
async def merge_branch_endpoint(
    root_id: str,
    branch_id: str = Path(..., min_length=1),
    storage: GraphStorage = Depends(get_graph_storage),
) -> BranchView:
    try:
        storage.merge_branch(root_id=root_id, branch_id=branch_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BranchView(root_id=root_id, branch_id=branch_id)


@app.post(
    "/api/v1/roots/{root_id}/branches/{branch_id}/revert",
    response_model=BranchView,
)
async def revert_branch_endpoint(
    root_id: str,
    branch_id: str = Path(..., min_length=1),
    storage: GraphStorage = Depends(get_graph_storage),
) -> BranchView:
    try:
        storage.revert_branch(root_id=root_id, branch_id=branch_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BranchView(root_id=root_id, branch_id=branch_id)


@app.post(
    "/api/v1/roots/{root_id}/branches/fork_from_commit",
    response_model=BranchView,
)
async def fork_from_commit_endpoint(
    root_id: str,
    payload: ForkFromCommitPayload,
    storage: GraphStorage = Depends(get_graph_storage),
) -> BranchView:
    try:
        storage.fork_from_commit(
            root_id=root_id,
            source_commit_id=payload.source_commit_id,
            new_branch_id=payload.new_branch_id,
            parent_branch_id=payload.parent_branch_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        detail = str(exc)
        status_code = 409 if "already exists" in detail else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc
    return BranchView(root_id=root_id, branch_id=payload.new_branch_id)


@app.post(
    "/api/v1/roots/{root_id}/branches/fork_from_scene",
    response_model=BranchView,
)
async def fork_from_scene_endpoint(
    root_id: str,
    payload: ForkFromScenePayload,
    storage: GraphStorage = Depends(get_graph_storage),
) -> BranchView:
    try:
        storage.fork_from_scene(
            root_id=root_id,
            source_branch_id=payload.source_branch_id,
            scene_origin_id=payload.scene_origin_id,
            new_branch_id=payload.new_branch_id,
            commit_id=payload.commit_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        detail = str(exc)
        status_code = 409 if "already exists" in detail else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc
    return BranchView(root_id=root_id, branch_id=payload.new_branch_id)


@app.post(
    "/api/v1/roots/{root_id}/branches/{branch_id}/reset",
    response_model=BranchView,
)
async def reset_branch_endpoint(
    root_id: str,
    branch_id: str = Path(..., min_length=1),
    payload: ResetBranchPayload = Body(...),
    storage: GraphStorage = Depends(get_graph_storage),
) -> BranchView:
    try:
        storage.reset_branch_head(
            root_id=root_id, branch_id=branch_id, commit_id=payload.commit_id
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BranchView(root_id=root_id, branch_id=branch_id)


@app.get(
    "/api/v1/roots/{root_id}/branches/{branch_id}/history",
    response_model=List[Commit],
)
async def branch_history_endpoint(
    root_id: str,
    branch_id: str = Path(..., min_length=1),
    limit: int = Query(50, ge=1),
    storage: GraphStorage = Depends(get_graph_storage),
) -> List[Commit]:
    try:
        return storage.get_branch_history(
            root_id=root_id, branch_id=branch_id, limit=limit
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post(
    "/api/v1/roots/{root_id}/branches/{branch_id}/commit",
    response_model=CommitResult,
)
async def commit_scene_endpoint(
    root_id: str,
    branch_id: str = Path(..., min_length=1),
    payload: CommitScenePayload = Body(...),
    storage: GraphStorage = Depends(get_graph_storage),
) -> CommitResult:
    try:
        return storage.commit_scene(
            root_id=root_id,
            branch_id=branch_id,
            scene_origin_id=payload.scene_origin_id,
            content=payload.content,
            message=payload.message,
            expected_head_version=payload.expected_head_version,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post(
    "/api/v1/roots/{root_id}/scene_origins",
    response_model=CreateSceneOriginResult,
)
async def create_scene_origin_endpoint(
    root_id: str,
    payload: CreateSceneOriginPayload,
    branch_id: str = Query(..., min_length=1),
    storage: GraphStorage = Depends(get_graph_storage),
) -> CreateSceneOriginResult:
    try:
        return storage.create_scene_origin(
            root_id=root_id,
            branch_id=branch_id,
            title=payload.title,
            parent_act_id=payload.parent_act_id,
            content=payload.content,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/commits/gc", response_model=GcResult)
async def gc_commits_endpoint(
    payload: GcPayload,
    storage: GraphStorage = Depends(get_graph_storage),
) -> GcResult:
    try:
        return storage.gc_orphan_commits(retention_days=payload.retention_days)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/roots/{root_id}", response_model=RootGraphView)
async def get_root_graph_endpoint(
    root_id: str,
    branch_id: str = Query(..., min_length=1),
    storage: GraphStorage = Depends(get_graph_storage),
) -> RootGraphView:
    try:
        snapshot = storage.get_root_snapshot(root_id=root_id, branch_id=branch_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return snapshot


@app.post("/api/v1/roots/{root_id}/entities", response_model=EntityView)
async def create_root_entity_endpoint(
    root_id: str,
    payload: CreateEntityPayload,
    branch_id: str = Query(..., min_length=1),
    storage: GraphStorage = Depends(get_graph_storage),
) -> EntityView:
    try:
        entity_id = storage.create_entity(
            root_id=root_id,
            branch_id=branch_id,
            name=payload.name,
            entity_type=payload.entity_type,
            tags=payload.tags,
            arc_status=payload.arc_status,
            semantic_states=payload.semantic_states,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return EntityView(
        entity_id=entity_id,
        name=payload.name,
        entity_type=payload.entity_type,
        tags=payload.tags,
        arc_status=payload.arc_status,
        semantic_states=payload.semantic_states,
    )


@app.get("/api/v1/roots/{root_id}/entities", response_model=List[EntityView])
async def list_root_entities_endpoint(
    root_id: str,
    branch_id: str = Query(..., min_length=1),
    storage: GraphStorage = Depends(get_graph_storage),
) -> List[EntityView]:
    return storage.list_entities(root_id=root_id, branch_id=branch_id)


@app.post("/api/v1/roots/{root_id}/relations", response_model=EntityRelationView)
async def upsert_relation_endpoint(
    root_id: str,
    payload: UpsertRelationPayload,
    branch_id: str = Query(..., min_length=1),
    storage: GraphStorage = Depends(get_graph_storage),
) -> EntityRelationView:
    try:
        storage.upsert_entity_relation(
            root_id=root_id,
            branch_id=branch_id,
            from_entity_id=payload.from_entity_id,
            to_entity_id=payload.to_entity_id,
            relation_type=payload.relation_type,
            tension=payload.tension,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return EntityRelationView(
        from_entity_id=payload.from_entity_id,
        to_entity_id=payload.to_entity_id,
        relation_type=payload.relation_type,
        tension=payload.tension,
    )


@app.get("/api/v1/scenes/{scene_id}/context", response_model=SceneContextView)
async def get_scene_context_endpoint(
    scene_id: str,
    branch_id: str = Query(..., min_length=1),
    storage: GraphStorage = Depends(get_graph_storage),
) -> SceneContextView:
    try:
        return storage.get_scene_context(scene_id=scene_id, branch_id=branch_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/scenes/{scene_id}/diff")
async def diff_scene_versions_endpoint(
    scene_id: str,
    from_commit_id: str = Query(..., min_length=1),
    to_commit_id: str = Query(..., min_length=1),
    storage: GraphStorage = Depends(get_graph_storage),
) -> dict[str, dict[str, Any]]:
    try:
        return storage.diff_scene_versions(
            scene_origin_id=scene_id,
            from_commit_id=from_commit_id,
            to_commit_id=to_commit_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post(
    "/api/v1/roots/{root_id}/scenes/{scene_id}/delete",
    response_model=CommitResult,
)
async def delete_scene_origin_endpoint(
    root_id: str,
    scene_id: str,
    payload: DeleteSceneOriginPayload,
    branch_id: str = Query(..., min_length=1),
    storage: GraphStorage = Depends(get_graph_storage),
) -> CommitResult:
    try:
        return storage.delete_scene_origin(
            root_id=root_id,
            branch_id=branch_id,
            scene_origin_id=scene_id,
            message=payload.message,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/scenes/{scene_id}/render", response_model=SceneRenderResult)
async def render_scene_endpoint(
    scene_id: str,
    payload: SceneRenderPayload,
    branch_id: str = Query(..., min_length=1),
    gateway: ToponeGateway = Depends(get_topone_gateway),
    storage: GraphStorage = Depends(get_graph_storage),
) -> SceneRenderResult:
    if _require_snowflake_engine_mode() != "gemini":
        raise HTTPException(
            status_code=400,
            detail="scene render 仅支持 gemini 模式，请设置 SNOWFLAKE_ENGINE=gemini。",
        )

    content = await gateway.render_scene(payload)
    try:
        storage.save_scene_render(
            scene_id=scene_id,
            branch_id=branch_id,
            content=content,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SceneRenderResult(
        ok=True,
        scene_id=scene_id,
        branch_id=branch_id,
        content=content,
    )


@app.post("/api/v1/scenes/{scene_id}/complete")
async def complete_scene_endpoint(
    scene_id: str,
    payload: SceneCompletePayload,
    branch_id: str = Query(..., min_length=1),
    storage: GraphStorage = Depends(get_graph_storage),
) -> dict[str, Any]:
    try:
        storage.complete_scene(
            scene_id=scene_id,
            branch_id=branch_id,
            actual_outcome=payload.actual_outcome,
            summary=payload.summary,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": True,
        "scene_id": scene_id,
        "branch_id": branch_id,
        "status": "committed",
        "actual_outcome": payload.actual_outcome,
        "summary": payload.summary,
    }


@app.post(
    "/api/v1/scenes/{scene_id}/complete/orchestrated",
    response_model=SceneCompletionResult,
    response_model_exclude_none=True,
)
async def complete_scene_orchestrated_endpoint(
    scene_id: str,
    payload: SceneCompletionOrchestratePayload,
    background_tasks: BackgroundTasks,
    gateway: ToponeGateway = Depends(get_topone_gateway),
    storage: GraphStorage = Depends(get_graph_storage),
) -> SceneCompletionResult:
    if _require_snowflake_engine_mode() != "gemini":
        raise HTTPException(
            status_code=400,
            detail="场景完成编排仅支持 gemini 模式，请设置 SNOWFLAKE_ENGINE=gemini。",
        )

    mode = payload.mode.strip().lower()
    if mode not in {"force_execute", "standard"}:
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {payload.mode!r}")

    is_logic_exception = False
    if mode == "force_execute":
        reason = payload.force_reason or payload.user_intent
        try:
            storage.mark_scene_logic_exception(
                root_id=payload.root_id,
                branch_id=payload.branch_id,
                scene_id=scene_id,
                reason=reason,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        is_logic_exception = False
    if mode != "force_execute":
        try:
            is_logic_exception = storage.is_scene_logic_exception(
                root_id=payload.root_id,
                branch_id=payload.branch_id,
                scene_id=scene_id,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    logic_payload = LogicCheckPayload(
        outline_requirement=payload.outline_requirement,
        world_state=payload.world_state,
        user_intent=payload.user_intent,
        mode=mode,
        root_id=payload.root_id,
        branch_id=payload.branch_id,
        scene_id=scene_id,
        force_reason=payload.force_reason,
    )
    logic_result = await gateway.logic_check(logic_payload)
    if mode != "force_execute" and not logic_result.ok and not is_logic_exception:
        raise HTTPException(
            status_code=400, detail=f"logic_check rejected: decision={logic_result.decision}"
        )
    if mode != "force_execute" and not is_logic_exception:
        background_tasks.add_task(
            _apply_impact_level,
            storage=storage,
            root_id=payload.root_id,
            branch_id=payload.branch_id,
            scene_id=scene_id,
            impact_level=logic_result.impact_level,
        )

    extract_payload = StateExtractPayload(
        content=payload.content,
        entity_ids=payload.entity_ids,
        root_id=payload.root_id,
        branch_id=payload.branch_id,
    )
    proposals = await gateway.state_extract(extract_payload)
    if not proposals:
        raise HTTPException(status_code=400, detail="state_extract returned empty proposals.")
    try:
        proposals = _enrich_state_proposals(
            storage=storage,
            root_id=payload.root_id,
            branch_id=payload.branch_id,
            proposals=proposals,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        updated_entities: list[dict[str, Any]] = []
        if payload.confirmed_proposals:
            extracted_entity_ids = {proposal.entity_id for proposal in proposals}
            for confirmed in payload.confirmed_proposals:
                if confirmed.entity_id not in extracted_entity_ids:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "confirmed_proposals entity_id not found in extracted proposals: "
                            f"{confirmed.entity_id}"
                        ),
                    )
            updated_entities = _apply_state_proposals(
                storage=storage,
                root_id=payload.root_id,
                branch_id=payload.branch_id,
                proposals=payload.confirmed_proposals,
            )
        storage.complete_scene(
            scene_id=scene_id,
            branch_id=payload.branch_id,
            actual_outcome=payload.actual_outcome,
            summary=payload.summary,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return SceneCompletionResult(
        ok=True,
        scene_id=scene_id,
        root_id=payload.root_id,
        branch_id=payload.branch_id,
        status="committed",
        actual_outcome=payload.actual_outcome,
        summary=payload.summary,
        logic_check=logic_result,
        extracted_proposals=proposals,
        confirmed_count=len(payload.confirmed_proposals),
        applied=len(updated_entities),
        updated_entities=updated_entities,
    )


@app.post("/api/v1/scenes/{scene_id}/dirty")
async def mark_scene_dirty_endpoint(
    scene_id: str,
    branch_id: str = Query(..., min_length=1),
    storage: GraphStorage = Depends(get_graph_storage),
) -> dict[str, Any]:
    try:
        storage.mark_scene_dirty(scene_id=scene_id, branch_id=branch_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, "scene_id": scene_id, "branch_id": branch_id}


@app.get("/api/v1/roots/{root_id}/dirty_scenes", response_model=List[str])
async def list_dirty_scenes_endpoint(
    root_id: str,
    branch_id: str = Query(..., min_length=1),
    storage: GraphStorage = Depends(get_graph_storage),
) -> List[str]:
    return storage.list_dirty_scenes(root_id=root_id, branch_id=branch_id)


@app.post("/api/v1/llm/topone/generate")
async def generate_topone_content(
    payload: ToponeGeneratePayload,
    client: ToponeClient = Depends(get_topone_client),
) -> Any:
    """调用 TopOne Gemini 原生接口，支持模型切换。"""
    return await client.generate_content(
        messages=[msg.model_dump() for msg in payload.messages],
        system_instruction=payload.system_instruction,
        generation_config=payload.generation_config,
        model=payload.model,
        timeout=payload.timeout,
    )


@app.post("/api/v1/logic/check", response_model=LogicCheckResult)
async def logic_check_endpoint(
    payload: LogicCheckPayload,
    gateway: ToponeGateway = Depends(get_topone_gateway),
    storage: GraphStorage = Depends(get_graph_storage),
) -> LogicCheckResult:
    mode = payload.mode.strip().lower()
    if mode not in {"force_execute", "standard"}:
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {payload.mode!r}")

    has_locator = any(
        value is not None for value in (payload.root_id, payload.branch_id, payload.scene_id)
    )
    if has_locator and not all(
        value is not None for value in (payload.root_id, payload.branch_id, payload.scene_id)
    ):
        raise HTTPException(
            status_code=400,
            detail="root_id/branch_id/scene_id must be all provided or all omitted.",
        )

    if _require_snowflake_engine_mode() != "gemini":
        raise HTTPException(
            status_code=400,
            detail="logic_check 仅支持 gemini 模式，请设置 SNOWFLAKE_ENGINE=gemini。",
        )

    if mode == "force_execute" and payload.root_id is not None:
        reason = payload.force_reason or payload.user_intent
        try:
            storage.mark_scene_logic_exception(
                root_id=payload.root_id,
                branch_id=payload.branch_id,
                scene_id=payload.scene_id,
                reason=reason,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    normalized_payload = payload.model_copy(update={"mode": mode})
    logic_result = await gateway.logic_check(normalized_payload)
    if payload.root_id is None or not logic_result.ok or mode == "force_execute":
        return logic_result
    try:
        if storage.is_scene_logic_exception(
            root_id=payload.root_id,
            branch_id=payload.branch_id,
            scene_id=payload.scene_id,
        ):
            return logic_result
        _apply_impact_level(
            storage=storage,
            root_id=payload.root_id,
            branch_id=payload.branch_id,
            scene_id=payload.scene_id,
            impact_level=logic_result.impact_level,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return logic_result


@app.post(
    "/api/v1/state/extract",
    response_model=List[StateProposal],
    response_model_exclude_none=True,
)
async def state_extract_endpoint(
    payload: StateExtractPayload,
    gateway: ToponeGateway = Depends(get_topone_gateway),
    storage: GraphStorage = Depends(get_graph_storage),
) -> List[StateProposal]:
    has_root = any(value is not None for value in (payload.root_id, payload.branch_id))
    if has_root and not all(value is not None for value in (payload.root_id, payload.branch_id)):
        raise HTTPException(
            status_code=400, detail="root_id/branch_id must be all provided or all omitted."
        )

    if _require_snowflake_engine_mode() != "gemini":
        raise HTTPException(
            status_code=400,
            detail="state_extract 仅支持 gemini 模式，请设置 SNOWFLAKE_ENGINE=gemini。",
        )

    proposals = await gateway.state_extract(payload)

    if payload.root_id is None:
        return proposals

    try:
        return _enrich_state_proposals(
            storage=storage,
            root_id=payload.root_id,
            branch_id=payload.branch_id,
            proposals=proposals,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _enrich_state_proposals(
    *,
    storage: GraphStorage,
    root_id: str,
    branch_id: str,
    proposals: List[StateProposal],
) -> List[StateProposal]:
    storage.require_root(root_id=root_id, branch_id=branch_id)
    enriched: list[StateProposal] = []
    for proposal in proposals:
        before = storage.get_entity_semantic_states(
            root_id=root_id,
            branch_id=branch_id,
            entity_id=proposal.entity_id,
        )
        after = before.copy()
        after.update(proposal.semantic_states_patch)
        enriched.append(
            proposal.model_copy(
                update={
                    "semantic_states_before": before,
                    "semantic_states_after": after,
                }
            )
        )
    return enriched


def _apply_state_proposals(
    *,
    storage: GraphStorage,
    root_id: str,
    branch_id: str,
    proposals: List[StateProposal],
) -> list[dict[str, Any]]:
    if not proposals:
        raise ValueError("proposals must not be empty.")
    storage.require_root(root_id=root_id, branch_id=branch_id)
    updated_entities: list[dict[str, Any]] = []
    for proposal in proposals:
        updated = storage.apply_semantic_states_patch(
            root_id=root_id,
            branch_id=branch_id,
            entity_id=proposal.entity_id,
            patch=proposal.semantic_states_patch,
        )
        updated_entities.append({"entity_id": proposal.entity_id, "semantic_states": updated})
    return updated_entities


def _apply_impact_level(
    *,
    storage: GraphStorage,
    root_id: str,
    branch_id: str,
    scene_id: str,
    impact_level: ImpactLevel,
) -> list[str]:
    if impact_level == ImpactLevel.NEGLIGIBLE:
        return []
    if impact_level == ImpactLevel.LOCAL:
        return storage.apply_local_scene_fix(
            root_id=root_id,
            branch_id=branch_id,
            scene_id=scene_id,
            limit=3,
        )
    if impact_level == ImpactLevel.CASCADING:
        return storage.mark_future_scenes_dirty(
            root_id=root_id,
            branch_id=branch_id,
            scene_id=scene_id,
        )
    raise ValueError(f"ImpactLevel {impact_level.value!r} is not supported in Phase1")


@app.post("/api/v1/state/commit")
async def state_commit_endpoint(
    *,
    root_id: str = Query(..., min_length=1),
    branch_id: str = Query(..., min_length=1),
    proposals: List[StateProposal] = Body(...),
    storage: GraphStorage = Depends(get_graph_storage),
) -> dict[str, Any]:
    try:
        updated_entities = _apply_state_proposals(
            storage=storage,
            root_id=root_id,
            branch_id=branch_id,
            proposals=proposals,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": True,
        "root_id": root_id,
        "branch_id": branch_id,
        "applied": len(updated_entities),
        "updated_entities": updated_entities,
    }
