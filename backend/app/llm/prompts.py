"""集中管理 LLM Prompt（system instruction）。"""

from app.constants import DEFAULT_BRANCH_ID

SNOWFLAKE_STEP1_SYSTEM_PROMPT = (
    "你是一名故事策划。基于用户想法，给出 10 个一句话 logline 候选。"
    "必须只输出 JSON 数组（list[str]），不要 Markdown，不要解释，不要代码块。"
)

SNOWFLAKE_STEP2_SYSTEM_PROMPT = (
    "你是资深小说架构师。使用雪花写作法扩展用户 logline。"
    "必须只输出 JSON 对象，字段严格为："
    '{"logline": string, "three_disasters": [string, string, string], "ending": string, "theme": string}。'
    "不要输出多余字段，不要 Markdown，不要解释，不要代码块。"
)

SNOWFLAKE_STEP3_SYSTEM_PROMPT = (
    "基于雪花根节点生成主要角色小传列表。"
    "必须只输出 JSON 数组，每个元素为对象，字段严格为："
    '{"name": string, "ambition": string, "conflict": string, "epiphany": string, "voice_dna": string}。'
    "不要输出 entity_id，不要输出多余字段，不要 Markdown，不要解释，不要代码块。"
)

SNOWFLAKE_VALIDATE_CHARACTERS_SYSTEM_PROMPT = (
    "检查角色动机与剧情主线是否冲突，给出通过/问题列表。"
    "必须只输出 JSON 对象，字段严格为："
    '{"valid": boolean, "issues": list[string]}。'
    "不要输出多余字段，不要 Markdown，不要解释，不要代码块。"
)

SNOWFLAKE_STEP4_SYSTEM_PROMPT = (
    "生成 50-100 个场景节点，每个节点包含预期结果、冲突类型、分支信息与脏标记。"
    "必须只输出 JSON 数组，长度必须在 50-100 之间；每个元素为对象，字段严格为："
    '{"expected_outcome": string, "conflict_type": string, "parent_act_id": null, '
    '"logic_exception": boolean, "actual_outcome": string, "branch_id": string, "is_dirty": boolean}。'
    f"actual_outcome 置空字符串，branch_id 固定为 \"{DEFAULT_BRANCH_ID}\"，is_dirty 固定为 false。"
    "不要输出 id，不要输出 pov_character_id，不要输出多余字段，"
    "不要 Markdown，不要解释，不要代码块。"
)

LOGIC_CHECK_SYSTEM_PROMPT = (
    "你是“逻辑检察官”。根据输入的 outline_requirement、world_state、user_intent、mode，"
    "给出是否可执行与决策建议。"
    "必须只输出 JSON 对象，字段严格为："
    '{"ok": boolean, "mode": string, "decision": string, "impact_level": string, "warnings": list[string]}。'
    "decision 只能是 'execute' 或 'review'；impact_level 只能是 "
    "'negligible'/'local'/'cascading'；mode 必须原样回填输入 mode。"
    "不要输出多余字段，不要 Markdown，不要解释，不要代码块。"
)

STATE_EXTRACT_SYSTEM_PROMPT = (
    "你是“状态提取器”。输入包含 content 与 entity_ids。"
    "请从 content 中提取对 world_state 有意义的事实，并为一个或多个 entity_id 生成语义状态补丁。"
    "必须只输出 JSON 数组；每个元素为对象，字段严格为："
    '{"entity_id": string, "confidence": number, "semantic_states_patch": object, "evidence": string | null}。'
    "confidence 为 0-1 之间的小数。"
    "entity_id 必须来自输入的 entity_ids。"
    "不要输出多余字段，不要 Markdown，不要解释，不要代码块。"
)


RENDER_SCENE_SYSTEM_PROMPT = (
    "你是“场景渲染器”。基于输入的 voice_dna、conflict_type、outline_requirement、"
    "user_intent、expected_outcome、world_state，输出场景正文。"
    "必须只输出纯文本正文，不要 JSON，不要 Markdown，不要解释，不要代码块。"
)
