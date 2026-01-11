"""核心领域模型定义，遵循雪花引擎 Phase 1 需求。"""

from __future__ import annotations

from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ValidationError, field_validator


class SnowflakeRoot(BaseModel):
    """雪花根节点：核心故事骨架。"""

    logline: str = Field(..., min_length=1, description="一句话核心故事")
    three_disasters: List[str] = Field(
        ..., min_length=3, max_length=3, description="三灾三难，长度必须为 3"
    )
    ending: str
    theme: str


class CharacterSheet(BaseModel):
    """人物小传。"""

    entity_id: UUID = Field(default_factory=uuid4)
    name: str
    ambition: str
    conflict: str
    epiphany: str
    voice_dna: str = Field(..., min_length=1, description="人物独特语气")

    @field_validator("voice_dna")
    @classmethod
    def ensure_voice_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("voice_dna must not be empty")
        return value


class SceneNode(BaseModel):
    """场景节点，面向 React Flow 渲染。"""

    id: UUID = Field(default_factory=uuid4)
    branch_id: str = Field(..., min_length=1)
    parent_act_id: Optional[UUID] = None
    pov_character_id: Optional[UUID] = None
    expected_outcome: str
    conflict_type: str
    actual_outcome: str
    logic_exception: bool = False
    is_dirty: bool


class CharacterValidationResult(BaseModel):
    """角色校验结果，用于流程中的冲突检查。"""

    valid: bool
    issues: List[str] = Field(default_factory=list)
