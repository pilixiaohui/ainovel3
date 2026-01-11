#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
DEFAULT_DB_PATH = BACKEND_DIR / "data" / "snowflake.db"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.models import CharacterSheet, SceneNode, SnowflakeRoot  # noqa: E402
from app.storage.graph import DEFAULT_BRANCH_ID, GraphStorage  # noqa: E402


def resolve_db_path(cli_path: str | None) -> Path:
    if cli_path is None:
        return DEFAULT_DB_PATH
    candidate = Path(cli_path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Milestone M5 ontology check (RootContainsAct/ActContainsScene + relation tension)."
    )
    parser.add_argument("--db", dest="db_path", required=True, help="Kùzu database path")
    args = parser.parse_args()

    db_path = resolve_db_path(args.db_path)
    storage = GraphStorage(db_path=db_path)
    try:
        # 1) Schema: Act table exists
        _ = list(storage.conn.execute("CALL table_info('Act') RETURN *;"))

        # 2) Write minimal data then assert Root->Act->Scene exists
        root = SnowflakeRoot(
            logline="M5 ontology check",
            three_disasters=["Act1 disaster", "Act2 disaster", "Act3 disaster"],
            ending="The end",
            theme="Theme",
        )
        character = CharacterSheet(
            name="Tester",
            ambition="A",
            conflict="C",
            epiphany="E",
            voice_dna="VD",
        )
        scenes = [
            SceneNode(
                branch_id=DEFAULT_BRANCH_ID,
                expected_outcome="S1",
                conflict_type="external",
                actual_outcome="A1",
                pov_character_id=character.entity_id,
                is_dirty=False,
            ),
            SceneNode(
                branch_id=DEFAULT_BRANCH_ID,
                expected_outcome="S2",
                conflict_type="internal",
                actual_outcome="A2",
                pov_character_id=character.entity_id,
                is_dirty=False,
            ),
            SceneNode(
                branch_id=DEFAULT_BRANCH_ID,
                expected_outcome="S3",
                conflict_type="mixed",
                actual_outcome="A3",
                pov_character_id=character.entity_id,
                is_dirty=False,
            ),
        ]
        root_id = storage.save_snowflake(
            root=root, characters=[character], scenes=scenes
        )

        path_count_rows = list(
            storage.conn.execute(
                (
                    "MATCH (r:Root)-[:RootContainsAct]->(:Act)-[:ActContainsScene]->(:Scene) "
                    f"WHERE r.id = '{storage._esc(root_id)}' "
                    f"AND r.branch_id = '{DEFAULT_BRANCH_ID}' "
                    "RETURN COUNT(*)"
                )
            )
        )
        assert path_count_rows, "Root->Act->Scene count query returned no rows"
        assert path_count_rows[0][0] > 0, "Root->Act->Scene path missing"

        # 3) EntityRelation.tension can be written and queried
        tension_value = 77
        storage.upsert_entity_relation(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            from_entity_id=str(character.entity_id),
            to_entity_id=str(character.entity_id),
            relation_type="knows",
            tension=tension_value,
        )
        tension_rows = list(
            storage.conn.execute(
                (
                    "MATCH (a:Entity)-[r:EntityRelation]->(b:Entity) "
                    f"WHERE a.id = '{storage._esc(str(character.entity_id))}' "
                    f"AND b.id = '{storage._esc(str(character.entity_id))}' "
                    f"AND r.branch_id = '{DEFAULT_BRANCH_ID}' "
                    "AND r.relation_type = 'knows' "
                    "RETURN r.tension LIMIT 1;"
                )
            )
        )
        assert tension_rows, "EntityRelation.tension query returned no rows"
        assert (
            tension_rows[0][0] == tension_value
        ), f"EntityRelation.tension mismatch: expected={tension_value} got={tension_rows[0][0]}"
    finally:
        storage.close()


if __name__ == "__main__":
    main()
