# Project 模块 - AI小说生成系统

本目录包含 AI 小说生成系统（ainovel）的所有项目特定代码，与 orchestrator 工作流框架完全分离。

## 目录结构

```
project/
├── __init__.py       # 模块导出
├── config.py         # 项目配置类（orchestrator 配置）
├── templates.py      # 初始化模板（orchestrator 模板）
├── README.md         # 本文档
│
├── backend/          # AI 小说后端服务
│   ├── app/          # FastAPI 应用
│   │   ├── main.py   # 主应用入口
│   │   ├── models.py # 数据模型
│   │   ├── config.py # 后端配置
│   │   ├── llm/      # LLM 相关（Gemini API）
│   │   ├── storage/  # 图数据库存储（Kuzu）
│   │   ├── logic/    # 雪花算法管理器
│   │   └── services/ # 服务层
│   ├── data/         # 数据库文件
│   ├── tests/        # 单元测试
│   └── README.md     # 后端文档
│
├── scripts/          # 测试和检查脚本
│   ├── cyberpunk_integration_test.py  # 端到端测试
│   ├── graph_health_check.py          # 图数据库健康检查
│   ├── m5_ontology_check.py           # 本体检查
│   ├── m6_negotiation_check.py        # 协商功能检查
│   └── performance_stress_test.py     # 性能/压测脚本
│
└── data_backup/      # 旧数据备份
```

## 项目概述

### AI 小说生成系统（Snowflake Engine）

这是一个基于图数据库和 LLM 的小说生成系统：

- **后端技术栈**：FastAPI + Kuzu 图数据库 + Google Gemini API
- **核心功能**：
  - 小说结构管理（Root、Branch、Scene）
  - 逻辑一致性检查
  - 状态管理和追踪
  - 协商式内容生成
  - 语义状态提取

### 与 Orchestrator 的关系

- **orchestrator.py**：通用多智能体工作流框架（可复用）
- **project/**：AI 小说项目的具体实现（项目特定）

orchestrator 使用 project/config.py 和 project/templates.py 进行配置，而 project/backend/ 是被开发的应用本身。

## 模块说明

### config.py - ProjectConfig (Orchestrator 配置)

项目配置类，包含所有项目特定的配置：

- **目录结构配置**：定义 memory/、workspace/、reports/ 等目录路径
- **文件路径配置**：定义黑板文件、工单文件、报告文件路径
- **验证规则配置**：dev_plan 的验证规则、约束条件
- **代理配置**：项目使用的代理列表（TEST、DEV、REVIEW）

**主要方法**：
- `get_task_file(agent)` - 获取指定代理的工单文件路径
- `get_report_file(agent)` - 获取指定代理的报告文件路径
- `get_prompt_file(agent)` - 获取指定代理的提示词文件路径
- `list_editable_md_files()` - 列出所有可编辑的 markdown 文件
- `resolve_editable_md_path(path)` - 解析并验证可编辑的 markdown 文件路径

### templates.py - ProjectTemplates

项目初始化模板类，包含各种文件的初始化模板内容：

**静态方法**：
- `global_context()` - 全局上下文模板
- `project_history()` - 项目历史模板
- `dev_plan()` - 开发计划模板
- `task_file(agent, iteration)` - 任务文件模板
- `report_file(agent)` - 报告文件模板

## 使用方式

### 启动后端服务

```bash
# 进入后端目录
cd project/backend

# 安装依赖
pip install -e .

# 配置环境变量（复制 .env.example）
cp .env.example .env

# 必填：设置 SNOWFLAKE_ENGINE=local|llm|gemini
# 可选：SCENE_MIN_COUNT/SCENE_MAX_COUNT 控制场景数量范围（默认 50-100）
# 例如：
# export SNOWFLAKE_ENGINE=local

# 启动服务
uvicorn app.main:app --reload --port 8000
```

### 运行测试脚本

```bash
# 图数据库健康检查
python project/scripts/graph_health_check.py --db project/backend/data/snowflake.db

# 端到端集成测试
python project/scripts/cyberpunk_integration_test.py

# 本体检查
python project/scripts/m5_ontology_check.py

# 性能/压测（需 SNOWFLAKE_ENGINE=gemini）
python project/scripts/performance_stress_test.py --p0-threshold-seconds 30 --p1-threshold-seconds 180

# 协商功能检查
python project/scripts/m6_negotiation_check.py
```

### 使用 Orchestrator 开发

```bash
# 启动 orchestrator UI
python orchestrator.py --ui

# 或命令行模式
python orchestrator.py --max-iterations 10 --task "实现新功能"
```

## 在 Orchestrator 中的使用

在 orchestrator.py 中的使用示例：

```python
from project import ProjectConfig, ProjectTemplates

# 初始化配置
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG = ProjectConfig(PROJECT_ROOT)

# 使用配置
MEMORY_DIR = CONFIG.memory_dir
DEV_PLAN_FILE = CONFIG.dev_plan_file

# 使用模板
_write_text_if_missing(
    CONFIG.global_context_file,
    ProjectTemplates.global_context()
)
```

## 自定义项目配置

如果要为新项目创建不同的配置：

1. 复制 `project/` 目录到新项目
2. 修改 `config.py` 中的配置参数：
   - 目录结构
   - 代理列表
   - 验证规则
3. 修改 `templates.py` 中的模板内容
4. orchestrator.py 无需修改，自动使用新配置

## 设计原则

- **关注点分离**：工作流框架代码与项目配置分离
- **可复用性**：orchestrator.py 可作为通用框架复用到其他项目
- **可扩展性**：新项目只需修改 project/ 目录内容
- **向后兼容**：保持全局变量兼容性，最小化迁移成本
