"""LLM 结构化输出 Schema（Pydantic v2）。"""

from __future__ import annotations

from enum import Enum
from typing import Any, List

from pydantic import BaseModel, Field


class ToponeMessage(BaseModel):
    role: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class ToponeGeneratePayload(BaseModel):
    model: str | None = None
    system_instruction: str | None = None
    messages: List[ToponeMessage]
    generation_config: dict | None = None
    timeout: float | None = None


class ImpactLevel(str, Enum):
    NEGLIGIBLE = "negligible"
    LOCAL = "local"
    CASCADING = "cascading"


class LogicCheckPayload(BaseModel):
    outline_requirement: str = Field(..., min_length=1)
    world_state: dict[str, Any] = Field(default_factory=dict)
    user_intent: str = Field(..., min_length=1)
    mode: str = Field(..., min_length=1)
    root_id: str | None = Field(default=None, min_length=1)
    branch_id: str | None = Field(default=None, min_length=1)
    scene_id: str | None = Field(default=None, min_length=1)
    force_reason: str | None = Field(default=None, min_length=1)


class LogicCheckResult(BaseModel):
    ok: bool
    mode: str
    decision: str
    impact_level: ImpactLevel
    warnings: List[str] = Field(default_factory=list)


class StateExtractPayload(BaseModel):
    content: str = Field(..., min_length=1)
    entity_ids: List[str] = Field(..., min_length=1)
    root_id: str | None = Field(default=None, min_length=1)
    branch_id: str | None = Field(default=None, min_length=1)


class SceneRenderPayload(BaseModel):
    voice_dna: str = Field(..., min_length=1)
    conflict_type: str = Field(..., min_length=1)
    outline_requirement: str = Field(..., min_length=1)
    user_intent: str = Field(..., min_length=1)
    expected_outcome: str = Field(..., min_length=1)
    world_state: dict[str, Any] = Field(default_factory=dict)
    logic_exception: bool | None = None
    force_reason: str | None = Field(default=None, min_length=1)


class StateProposal(BaseModel):
    entity_id: str = Field(..., min_length=1)
    entity_name: str | None = Field(default=None, min_length=1)
    confidence: float
    semantic_states_patch: dict[str, Any]
    semantic_states_before: dict[str, Any] | None = None
    semantic_states_after: dict[str, Any] | None = None
    evidence: str | None = None
