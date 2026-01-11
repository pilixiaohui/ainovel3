#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple


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
        env_candidate = Path(env_path)
        return env_candidate if env_candidate.is_absolute() else REPO_ROOT / env_candidate
    return DEFAULT_DB_PATH


def _collect_rows(storage: GraphStorage, query: str) -> List[Tuple]:
    return [tuple(row) for row in storage.conn.execute(query)]


def check_missing_fields(storage: GraphStorage) -> List[str]:
    issues: List[str] = []
    queries = {
        "Scene missing branch_id/status": (
            "MATCH (s:Scene) WHERE s.branch_id IS NULL OR s.status IS NULL "
            "RETURN s.id, s.branch_id, s.status;"
        ),
        "Entity missing branch_id": (
            "MATCH (e:Entity) WHERE e.branch_id IS NULL RETURN e.id;"
        ),
    }
    for title, query in queries.items():
        rows = _collect_rows(storage, query)
        if rows:
            issues.append(f"{title}: {rows}")
    return issues


def check_scene_entities(storage: GraphStorage) -> List[str]:
    rows = _collect_rows(
        storage,
        (
            "MATCH (s:Scene) "
            "OPTIONAL MATCH (s)-[r:SceneEntity]->(:Entity) "
            "WITH s, COUNT(r) AS rc "
            "WHERE rc = 0 "
            "RETURN s.id, s.branch_id;"
        ),
    )
    return [f"Scene without SceneEntity links: {rows}"] if rows else []


def check_scene_next(storage: GraphStorage) -> List[str]:
    issues: List[str] = []
    prev_rows = _collect_rows(
        storage,
        (
            "MATCH (p:Scene)-[r:SceneNext]->(s:Scene) "
            "WITH s, COUNT(r) AS cnt "
            "WHERE cnt > 1 "
            "RETURN s.id, s.branch_id, cnt;"
        ),
    )
    if prev_rows:
        issues.append(f"Scenes with multiple previous nodes: {prev_rows}")

    next_rows = _collect_rows(
        storage,
        (
            "MATCH (s:Scene)-[r:SceneNext]->(n:Scene) "
            "WITH s, COUNT(r) AS cnt "
            "WHERE cnt > 1 "
            "RETURN s.id, s.branch_id, cnt;"
        ),
    )
    if next_rows:
        issues.append(f"Scenes with multiple next nodes: {next_rows}")

    branch_mismatch = _collect_rows(
        storage,
        (
            "MATCH (a:Scene)-[r:SceneNext]->(b:Scene) "
            "WHERE r.branch_id <> a.branch_id OR r.branch_id <> b.branch_id "
            "RETURN a.id, b.id, r.branch_id, a.branch_id, b.branch_id;"
        ),
    )
    if branch_mismatch:
        issues.append(f"SceneNext branch_id mismatch: {branch_mismatch}")

    return issues


def run_checks(storage: GraphStorage) -> List[str]:
    findings: List[str] = []
    for checker in (check_missing_fields, check_scene_entities, check_scene_next):
        findings.extend(checker(storage))
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Kùzu graph health checks.")
    parser.add_argument(
        "--db",
        dest="db_path",
        help=(
            "Path to Kùzu database file (defaults to KUZU_DB_PATH relative to repo root or "
            "backend/data/snowflake.db)"
        ),
    )
    args = parser.parse_args()

    db_path = resolve_db_path(args.db_path)
    storage: GraphStorage | None = None
    findings: List[str] = []
    try:
        storage = GraphStorage(db_path=db_path)
        findings = run_checks(storage)
    except Exception as exc:
        print(f"Health check aborted for {db_path}: {exc}")
        raise SystemExit(1)
    finally:
        if storage:
            storage.close()

    if findings:
        print(f"Health check failed for {db_path}:")
        for item in findings:
            print(f"- {item}")
        raise SystemExit(1)

    print(f"Health check passed for {db_path}")


if __name__ == "__main__":
    main()
