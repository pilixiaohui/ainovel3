import pytest
from uuid import uuid4
from pydantic import ValidationError

from app.constants import DEFAULT_BRANCH_ID
from app.models import CharacterSheet, SceneNode, SnowflakeRoot


def test_snowflake_root_validation():
    # 缺少 three_disasters/ending/theme 应抛出校验错误
    with pytest.raises(ValidationError):
        SnowflakeRoot(logline="仅有一句话")

    # three_disasters 长度不足应失败
    with pytest.raises(ValidationError):
        SnowflakeRoot(
            logline="完整信息",
            three_disasters=["A", "B"],
            ending="End",
            theme="Theme",
        )


def test_character_sheet_structure():
    char = CharacterSheet(
        entity_id=uuid4(),
        name="Neo",
        ambition="Find Truth",
        conflict="Agents",
        epiphany="I am the One",
        voice_dna="Stoic",
    )
    assert char.name == "Neo"
    assert char.entity_id

    # voice_dna 不可为空
    with pytest.raises(ValidationError):
        CharacterSheet(
            entity_id=uuid4(),
            name="Neo",
            ambition="Find Truth",
            conflict="Agents",
            epiphany="I am the One",
            voice_dna="",
        )


def test_scene_node_requires_expected_outcome():
    with pytest.raises(ValidationError):
        SceneNode(
            branch_id=DEFAULT_BRANCH_ID,
            conflict_type="internal",
            actual_outcome="",
            is_dirty=False,
        )


def test_scene_node_requires_branch_id():
    with pytest.raises(ValidationError):
        SceneNode(
            expected_outcome="Outcome",
            conflict_type="internal",
            actual_outcome="",
            is_dirty=False,
        )


def test_scene_node_requires_actual_outcome():
    with pytest.raises(ValidationError):
        SceneNode(
            branch_id=DEFAULT_BRANCH_ID,
            expected_outcome="Outcome",
            conflict_type="internal",
            is_dirty=False,
        )


def test_scene_node_requires_is_dirty():
    with pytest.raises(ValidationError):
        SceneNode(
            branch_id=DEFAULT_BRANCH_ID,
            expected_outcome="Outcome",
            conflict_type="internal",
            actual_outcome="",
        )
