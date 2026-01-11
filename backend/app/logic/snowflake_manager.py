"""雪花流程编排，负责将 LLM 输出组织成业务步骤。"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Protocol, Sequence

from app.models import CharacterSheet, CharacterValidationResult, SceneNode, SnowflakeRoot

if TYPE_CHECKING:  # 避免在未安装 kuzu 时导入报错
    from app.storage.graph import GraphStorage


class StoryEngine(Protocol):
    async def generate_logline_options(self, raw_idea: str) -> List[str]: ...

    async def generate_root_structure(self, idea: str) -> SnowflakeRoot: ...

    async def generate_characters(self, root: SnowflakeRoot) -> List[CharacterSheet]: ...

    async def validate_characters(
        self, root: SnowflakeRoot, characters: Sequence[CharacterSheet]
    ) -> CharacterValidationResult: ...

    async def generate_scene_list(
        self, root: SnowflakeRoot, characters: Sequence[CharacterSheet]
    ) -> List[SceneNode]: ...


class SnowflakeManager:
    """雪花写作法的业务 orchestrator。"""

    def __init__(
        self,
        engine: StoryEngine,
        min_scenes: int = 50,
        max_scenes: int = 100,
        storage: GraphStorage | None = None,
    ):
        self.engine = engine
        self.min_scenes = min_scenes
        self.max_scenes = max_scenes
        self.storage = storage
        self.last_persisted_root_id: str | None = None

    async def execute_step_1_logline(self, raw_idea: str) -> List[str]:
        options = await self.engine.generate_logline_options(raw_idea)
        if len(options) != 10:
            raise ValueError(f"Logline option count {len(options)} is not 10")
        return options

    async def execute_step_2_structure(self, selected_logline: str) -> SnowflakeRoot:
        return await self.engine.generate_root_structure(selected_logline)

    async def execute_step_3_characters(
        self, root: SnowflakeRoot
    ) -> List[CharacterSheet]:
        characters = await self.engine.generate_characters(root)

        validator = getattr(self.engine, "validate_characters", None)
        if callable(validator):
            validation = await validator(root, characters)
            if hasattr(validation, "valid") and not validation.valid:
                issues = getattr(validation, "issues", None) or []
                details = "; ".join(issues) if issues else "Character validation failed"
                raise ValueError(details)

        return characters

    async def execute_step_4_scenes(
        self, root: SnowflakeRoot, characters: Sequence[CharacterSheet]
    ) -> List[SceneNode]:
        scenes = await self.engine.generate_scene_list(root, characters)

        if not self.min_scenes <= len(scenes) <= self.max_scenes:
            raise ValueError(
                f"Scene count {len(scenes)} outside required range "
                f"{self.min_scenes}-{self.max_scenes}"
            )

        ids = [str(scene.id) for scene in scenes]
        if len(ids) != len(set(ids)):
            raise ValueError("Scene IDs must be unique")

        for scene in scenes:
            if not scene.expected_outcome or not scene.expected_outcome.strip():
                raise ValueError("Scene expected_outcome is required")
            if not scene.conflict_type or not scene.conflict_type.strip():
                raise ValueError("Scene conflict_type is required")

        if characters:
            pov_cycle = [c.entity_id for c in characters]
            for idx, scene in enumerate(scenes):
                if scene.pov_character_id is None:
                    scene.pov_character_id = pov_cycle[idx % len(pov_cycle)]
        elif self.storage:
            raise ValueError("characters is required for scene persistence")

        if self.storage:
            self.last_persisted_root_id = self.storage.save_snowflake(
                root=root, characters=characters, scenes=scenes
            )

        return scenes
