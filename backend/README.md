# Backend Notes

- 启动必须显式配置 `SNOWFLAKE_ENGINE`（local/llm/gemini），无默认值。
- 环境变量：`SCENE_MIN_COUNT` 与 `SCENE_MAX_COUNT` 控制生成场景数量范围，默认 50-100。
- TopOne 模型默认值：`TOPONE_DEFAULT_MODEL=gemini-3-pro-preview-11-2025`，`TOPONE_SECONDARY_MODEL=gemini-3-flash-preview`，可在 `.env` 覆盖。
- 数据库路径：`KUZU_DB_PATH` 的相对路径以仓库根目录为基准，默认 `backend/data/snowflake.db`，与 `.env` 示例和健康检查保持一致。
- 协商 WebSocket `/ws/negotiation` 已移除，当前不可用。
