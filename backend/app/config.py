"""基础配置与环境变量加载器，支持 .env 文件与系统环境并存."""
from __future__ import annotations

import os
from pathlib import Path

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def _load_env_file(path: Path = _ENV_PATH) -> None:
    """读取 .env 文件到 os.environ，不覆盖已存在的环境变量."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        cleaned = value.strip().strip('"').strip("'")
        os.environ[key] = cleaned


_load_env_file()

TOPONE_API_KEY: str | None = os.getenv("TOPONE_API_KEY")
TOPONE_BASE_URL: str = os.getenv("TOPONE_BASE_URL", "https://api.toponeapi.top")
TOPONE_DEFAULT_MODEL: str = os.getenv(
    "TOPONE_DEFAULT_MODEL", "gemini-3-pro-preview-11-2025"
)
TOPONE_SECONDARY_MODEL: str = os.getenv(
    "TOPONE_SECONDARY_MODEL", "gemini-3-flash-preview"
)
TOPONE_TIMEOUT_SECONDS: float = float(os.getenv("TOPONE_TIMEOUT_SECONDS", "30"))


def _get_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


SCENE_MIN_COUNT: int = _get_positive_int("SCENE_MIN_COUNT", 50)
SCENE_MAX_COUNT: int = _get_positive_int("SCENE_MAX_COUNT", 100)
if SCENE_MIN_COUNT > SCENE_MAX_COUNT:
    raise ValueError("SCENE_MIN_COUNT must be <= SCENE_MAX_COUNT")
