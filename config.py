"""
项目特定配置
包含目录结构、验证规则、初始化模板等项目特定配置
"""

from pathlib import Path
from typing import Literal

ProjectAgent = Literal["TEST", "DEV", "REVIEW"]


class ProjectConfig:
    """项目配置类"""

    def __init__(self, project_root: Path):
        self.project_root = project_root

        # orchestrator 数据目录
        self.orchestrator_dir = project_root / "orchestrator"

        # 目录结构配置（现在都在 orchestrator/ 下）
        self.memory_dir = self.orchestrator_dir / "memory"
        self.workspace_dir = self.orchestrator_dir / "workspace"
        self.reports_dir = self.orchestrator_dir / "reports"
        self.prompts_dir = self.memory_dir / "prompts"

        # 核心黑板文件
        self.project_history_file = self.memory_dir / "project_history.md"
        self.global_context_file = self.memory_dir / "global_context.md"
        self.dev_plan_file = self.memory_dir / "dev_plan.md"
        self.finish_review_config_file = self.memory_dir / "finish_review_config.json"
        self.verification_policy_file = self.memory_dir / "verification_policy.json"
        self.dev_plan_staged_file = self.workspace_dir / "main/dev_plan_next.md"

        # 工单文件
        self.test_task_file = self.workspace_dir / "test/current_task.md"
        self.dev_task_file = self.workspace_dir / "dev/current_task.md"
        self.review_task_file = self.workspace_dir / "review/current_task.md"

        # 报告文件
        self.report_test_file = self.reports_dir / "report_test.md"
        self.report_dev_file = self.reports_dir / "report_dev.md"
        self.report_review_file = self.reports_dir / "report_review.md"
        self.report_finish_review_file = self.reports_dir / "report_finish_review.md"
        self.report_main_decision_file = self.reports_dir / "report_main_decision.json"

        self.report_iteration_summary_file = self.reports_dir / "report_iteration_summary.json"
        self.report_iteration_summary_history_file = self.reports_dir / "report_iteration_summary_history.jsonl"
        # 备份目录
        self.reports_backup_dir = self.reports_dir / "backups"
        self.workspace_backup_dir = self.workspace_dir / "backups"
        self.memory_backup_dir = self.memory_dir / "backups"

        # Dev Plan 验证规则
        self.dev_plan_allowed_statuses = {"TODO", "DOING", "BLOCKED", "DONE", "VERIFIED"}
        self.dev_plan_max_tasks = 60
        self.dev_plan_max_line_length = 2000
        self.dev_plan_banned_substrings = (
            "Note to=functions.",
            "*** Begin Patch",
            "*** End Patch",
            "functions.apply_patch",
            "apply_patch(",
            "apply_patch(auto_approved=",
            "file update",
            "diff --git",
            "structuredContent",
            '"isError"',
            "tool serena.",
            "tokens used",
        )

        # 代理列表
        self.agents: list[ProjectAgent] = ["TEST", "DEV", "REVIEW"]
        self.editable_md_skip_dirs = {
            ".git",
            ".codex",
            ".serena",
            ".pytest_cache",
            "__pycache__",
            "node_modules",
        }

    def get_task_file(self, agent: str) -> Path:
        """获取指定代理的工单文件"""
        if agent == "TEST":
            return self.test_task_file
        elif agent == "DEV":
            return self.dev_task_file
        elif agent == "REVIEW":
            return self.review_task_file
        else:
            raise ValueError(f"Unknown agent: {agent}")

    def get_report_file(self, agent: str) -> Path:
        """获取指定代理的报告文件"""
        if agent == "TEST":
            return self.report_test_file
        elif agent == "DEV":
            return self.report_dev_file
        elif agent == "REVIEW":
            return self.report_review_file
        else:
            raise ValueError(f"Unknown agent: {agent}")

    def get_prompt_file(self, agent: str) -> Path:
        """获取指定代理的提示词文件"""
        agent_lower = agent.lower()
        return self.prompts_dir / f"subagent_prompt_{agent_lower}.md"

    def list_editable_md_files(self) -> list[str]:
        """列出所有可编辑的 markdown 文件"""
        files: list[str] = []
        if not self.project_root.exists():
            return files
        for path in self.project_root.rglob("*.md"):
            if not path.is_file():
                continue
            rel = path.relative_to(self.project_root)
            if any(part in self.editable_md_skip_dirs for part in rel.parts):
                continue
            files.append(rel.as_posix())
        return sorted(set(files))

    def resolve_editable_md_path(self, relative_path: str) -> Path:
        """解析并验证可编辑的 markdown 文件路径"""
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ValueError("path is required")
        if "\x00" in relative_path:
            raise ValueError("path contains null byte")
        raw = Path(relative_path)
        if raw.is_absolute():
            raise ValueError("path must be project-relative")

        resolved = (self.project_root / raw).resolve()
        if resolved.suffix.lower() != ".md":
            raise ValueError("only .md files are editable")
        try:
            rel = resolved.relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError("path is outside project root") from exc
        if any(part in self.editable_md_skip_dirs for part in rel.parts):
            raise ValueError("path is inside a blocked directory")

        return resolved
