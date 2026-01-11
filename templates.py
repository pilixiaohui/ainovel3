"""
项目特定的初始化模板内容
"""


class ProjectTemplates:
    """项目初始化模板"""

    @staticmethod
    def global_context() -> str:
        """全局上下文模板"""
        return """
# Global Context

请在此维护本项目的全局目标、约束与黑板协议（Blackboard Contract）。

## 必要约束
- JSON 字段使用 `snake_case`
- 快速失败：不要添加兜底/容错逻辑来让流程继续
- 允许测试双/Mock：测试可使用真实依赖或测试双

## Blackboard Paths（以项目根目录为基准）
- `orchestrator/memory/`：长期记忆、全局上下文
- `orchestrator/memory/prompts/`：各代理提示词（固定文件，重置时保留）
- `orchestrator/memory/dev_plan.md`：全局开发计划快照（MAIN 维护，REVIEW 核实）
- New Task 目标：由用户在 New Task 时输入，追加写入 `orchestrator/memory/project_history.md`（MAIN 注入 history 后读取）
- `orchestrator/workspace/`：各子代理工单（由 MAIN 覆盖写入）
- `orchestrator/reports/`：各子代理输出报告（由编排器保存）
"""

    @staticmethod
    def project_history() -> str:
        """项目历史模板"""
        return """
# Project History (Append-only)

> 本文件由 MAIN 追加写入；每轮必须包含 `## Iteration {iteration}:`。
"""

    @staticmethod
    def dev_plan() -> str:
        """开发计划模板"""
        return """
# Dev Plan (Snapshot)

> 本文件由 MAIN 维护（可覆盖编辑）。它是"当前计划与进度"的快照；事实证据以 `orchestrator/reports/report_review.md` 与 `orchestrator/memory/project_history.md` 为准。

## 约束（强制）
- 任务总数必须保持在"几十条"以内（少而硬）
- 每个任务块必须包含：`status / acceptance / evidence`
- status 只允许：TODO / DOING / BLOCKED / DONE / VERIFIED
- 只有 **REVIEW 的证据** 才能把 DONE -> VERIFIED（evidence 必须引用 Iteration 与验证方式）

---

## Milestone M0: 引导与黑板契约

### M0-T1: 建立 dev_plan 状态机
- status: TODO
- acceptance:
- `orchestrator/memory/dev_plan.md` 存在并遵循固定字段
- 每轮更新在 `orchestrator/memory/project_history.md` 留痕（说明改了什么/为什么）
- evidence:

### M0-T2: 审阅代理输出可核实证据
- status: TODO
- acceptance:
- `orchestrator/reports/report_review.md` 包含可复现的命令与关键输出摘要
  - 进度核实：逐条对照 dev_plan 的任务给出 PASS/FAIL 与证据
- evidence:
"""

    @staticmethod
    def finish_review_config() -> str:
        """最终审阅配置模板（JSON）"""
        return """{
  "task_goal_anchor": "## Task Goal (New Task)",
  "task_goal_anchor_mode": "prefix_latest",
  "docs": [
    "doc/<design_doc>.md",
    "doc/<api_doc>.md"
  ],
  "code_root": "."
}
"""

    @staticmethod
    def verification_policy() -> str:
        """验证策略配置模板（JSON）"""
        return """{
  "version": 1,
  "report_rules": {
    "apply_to": ["REVIEW", "TEST", "FINISH_REVIEW"],
    "require_verdict": true,
    "verdict_prefix": "结论：",
    "verdict_allowed": ["PASS", "FAIL", "BLOCKED"],
    "blocker_prefix": "阻塞：",
    "blocker_clear_value": "无"
  }
}
"""

    @staticmethod
    def task_file(agent: str, iteration: int = 0) -> str:
        """任务文件模板"""
        return f"""# Current Task (Iteration {iteration})
assigned_agent: {agent}

本文件由 MAIN 覆盖写入；{agent} 子代理仅以此为唯一任务来源。

## Acceptance Criteria
- TDD：先写/补齐测试，红-绿-重构；若任务类型不适用，说明原因
"""

    @staticmethod
    def report_file(agent: str) -> str:
        """报告文件模板"""
        return f"""# Report: {agent}

(no report yet)
"""
