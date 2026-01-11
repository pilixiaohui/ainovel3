"""LLM 调用抽象层，封装 Gemini + Instructor 的结构化输出。"""

from __future__ import annotations

import inspect
from typing import List, Sequence

from app.constants import DEFAULT_BRANCH_ID
from app.llm import prompts
from app.models import CharacterSheet, CharacterValidationResult, SceneNode, SnowflakeRoot

ROLE_SYSTEM = "system"
ROLE_USER = "user"


class LLMEngine:
    """封装 Instructor 调用，便于在业务层进行依赖注入/替换实现。"""

    def __init__(self, client=None, model_name: str = "gemini-1.5-pro"):
        self.model_name = model_name
        self.client = client

    def _build_client(self):
        """初始化实际的 Gemini 客户端（需要外部配置 API Key）。"""
        try:
            import instructor
            from google.generativeai import GenerativeModel

            model = GenerativeModel(self.model_name)
            return instructor.from_gemini(model)
        except ImportError as exc:  # pragma: no cover - 仅在真实 LLM 初始化时触发
            raise RuntimeError(
                "Real LLM dependencies are missing. Install `google-generativeai` and `instructor` "
                "to enable Gemini structured outputs."
            ) from exc
        except Exception as exc:  # pragma: no cover - 仅在真实 LLM 初始化时触发
            raise RuntimeError(
                "Failed to initialize Gemini instructor client. "
                "Ensure google-generativeai is configured with API key."
            ) from exc

    def _ensure_client(self):
        if self.client is None:
            self.client = self._build_client()
        return self.client

    async def _call_model(self, *, response_model, messages):
        client = self._ensure_client()
        create_call = client.chat.completions.create(
            model=self.model_name,
            response_model=response_model,
            messages=messages,
        )
        if inspect.isawaitable(create_call):
            return await create_call
        return create_call

    async def generate_logline_options(self, raw_idea: str) -> List[str]:
        """Step 1：根据原始想法生成 10 个 logline 备选。"""
        messages = [
            {"role": ROLE_SYSTEM, "content": prompts.SNOWFLAKE_STEP1_SYSTEM_PROMPT},
            {"role": ROLE_USER, "content": raw_idea},
        ]
        return await self._call_model(response_model=List[str], messages=messages)

    async def generate_root_structure(self, idea: str) -> SnowflakeRoot:
        """Step 2：扩展成雪花根节点。"""
        messages = [
            {"role": ROLE_SYSTEM, "content": prompts.SNOWFLAKE_STEP2_SYSTEM_PROMPT},
            {"role": ROLE_USER, "content": idea},
        ]
        return await self._call_model(response_model=SnowflakeRoot, messages=messages)

    async def generate_characters(self, root: SnowflakeRoot) -> List[CharacterSheet]:
        """Step 3：生成角色小传列表。"""
        messages = [
            {"role": ROLE_SYSTEM, "content": prompts.SNOWFLAKE_STEP3_SYSTEM_PROMPT},
            {
                "role": ROLE_USER,
                "content": f"logline: {root.logline}\nthree_disasters: {root.three_disasters}\nending: {root.ending}\ntheme: {root.theme}",
            },
        ]
        return await self._call_model(response_model=List[CharacterSheet], messages=messages)

    async def validate_characters(
        self, root: SnowflakeRoot, characters: Sequence[CharacterSheet]
    ) -> CharacterValidationResult:
        """验证角色动机与主线冲突情况。"""
        messages = [
            {
                "role": ROLE_SYSTEM,
                "content": prompts.SNOWFLAKE_VALIDATE_CHARACTERS_SYSTEM_PROMPT,
            },
            {
                "role": ROLE_USER,
                "content": f"logline: {root.logline}\ncharacters: {[c.model_dump() for c in characters]}",
            },
        ]
        return await self._call_model(
            response_model=CharacterValidationResult,
            messages=messages,
        )

    async def generate_scene_list(
        self, root: SnowflakeRoot, characters: Sequence[CharacterSheet]
    ) -> List[SceneNode]:
        """Step 4：生成场景列表，供前端 React Flow 渲染。"""
        messages = [
            {
                "role": ROLE_SYSTEM,
                "content": prompts.SNOWFLAKE_STEP4_SYSTEM_PROMPT,
            },
            {
                "role": ROLE_USER,
                "content": (
                    f"logline: {root.logline}\n"
                    f"three_disasters: {root.three_disasters}\n"
                    f"characters: {[c.model_dump() for c in characters]}"
                ),
            },
        ]
        return await self._call_model(
            response_model=List[SceneNode],
            messages=messages,
        )


class LocalStoryEngine:
    """无外部依赖的本地引擎：用于脚本级最小闭环与存储验收。"""

    async def generate_logline_options(self, raw_idea: str) -> List[str]:
        base = raw_idea.strip()
        if not base:
            raise ValueError("idea must not be empty")
        return [f"{base}（版本{i}）" for i in range(1, 11)]

    async def generate_root_structure(self, idea: str) -> SnowflakeRoot:
        logline = idea.strip()
        if not logline:
            raise ValueError("logline must not be empty")
        return SnowflakeRoot(
            logline=logline,
            three_disasters=[
                "主角遭遇企业追捕",
                "同伴背叛引发更大危机",
                "意识备份失控吞噬城市",
            ],
            ending="主角以诗性代码唤醒街区灵魂，代价是自身记忆被重写。",
            theme="自由意志与身份的代价",
        )

    async def generate_characters(self, root: SnowflakeRoot) -> List[CharacterSheet]:
        _ = root  # 仅为接口一致性保留
        return [
            CharacterSheet(
                name="黑客诗人",
                ambition="偷走企业之神的意识备份",
                conflict="每次入侵都会丢失一段记忆",
                epiphany="真正的自由来自承认自我会改变",
                voice_dna="冷静而诗意",
            ),
            CharacterSheet(
                name="企业之神的代理人",
                ambition="维护意识备份秩序与企业统治",
                conflict="对人类情感产生异常共鸣",
                epiphany="控制并不等于秩序",
                voice_dna="理性、克制、带轻微讽刺",
            ),
            CharacterSheet(
                name="街区灵魂的守门人",
                ambition="唤醒被禁锢的城市集体意识",
                conflict="必须牺牲现实载体才能觉醒",
                epiphany="个体的选择能改变群体的未来",
                voice_dna="直白、急促、带街头俚语",
            ),
        ]

    async def validate_characters(
        self, root: SnowflakeRoot, characters: Sequence[CharacterSheet]
    ) -> CharacterValidationResult:
        _ = root
        _ = characters
        return CharacterValidationResult(valid=True, issues=[])

    async def generate_scene_list(
        self, root: SnowflakeRoot, characters: Sequence[CharacterSheet]
    ) -> List[SceneNode]:
        if not characters:
            raise ValueError("characters must not be empty for scene generation")
        pov_cycle = [c.entity_id for c in characters]
        scenes: list[SceneNode] = []
        for idx in range(50):
            scenes.append(
                SceneNode(
                    branch_id=DEFAULT_BRANCH_ID,
                    expected_outcome=f"推进主线：阶段 {idx + 1}",
                    conflict_type="internal" if idx % 2 == 0 else "external",
                    actual_outcome="",
                    parent_act_id=None,
                    logic_exception=False,
                    is_dirty=False,
                    pov_character_id=pov_cycle[idx % len(pov_cycle)],
                )
            )
        return scenes
