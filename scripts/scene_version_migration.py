#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
DEFAULT_DB_PATH = BACKEND_DIR / "data" / "snowflake.db"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import config  # noqa: E402,F401
from app.storage.graph import GraphStorage  # noqa: E402


def resolve_db_path(cli_path: str | None) -> Path:
    if cli_path:
        candidate = Path(cli_path)
        return candidate if candidate.is_absolute() else REPO_ROOT / candidate
    env_path = os.getenv("KUZU_DB_PATH")
    if env_path:
        candidate = Path(env_path)
        return candidate if candidate.is_absolute() else REPO_ROOT / candidate
    return DEFAULT_DB_PATH


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate legacy Scene data to SceneOrigin/SceneVersion with commits."
        )
    )
    parser.add_argument(
        "--db",
        dest="db_path",
        help=(
            "Path to Kùzu database (defaults to KUZU_DB_PATH or backend/data/snowflake.db)"
        ),
    )
    parser.add_argument("--root-id", dest="root_id", help="Limit to a root_id")
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback migrated SceneOrigin/SceneVersion/Commit/BranchHead data",
    )
    args = parser.parse_args()

    db_path = resolve_db_path(args.db_path)
    storage = GraphStorage(db_path=db_path)
    try:
        if args.rollback:
            result = storage.rollback_scene_version_migration(root_id=args.root_id)
            print(f"Rollback completed for {db_path}: {result}")
        else:
            result = storage.migrate_scene_versions(root_id=args.root_id)
            print(f"Migration completed for {db_path}: {result}")
    finally:
        storage.close()


if __name__ == "__main__":
    main()
