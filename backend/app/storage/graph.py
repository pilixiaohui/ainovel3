"""Kùzu Graph 存储封装，用于持久化雪花结构。"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence
from uuid import UUID, uuid4

import kuzu

from app.constants import DEFAULT_BRANCH_ID
from app.models import CharacterSheet, SceneNode, SnowflakeRoot


DEFAULT_SCENE_STATUS = "draft"
DEFAULT_RELATION_TENSION = 0
_BRANCH_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9._-]{1,64}$")
_SCENE_VERSION_FIELDS = (
    "status",
    "pov_character_id",
    "expected_outcome",
    "conflict_type",
    "actual_outcome",
    "summary",
    "rendered_content",
    "logic_exception",
    "logic_exception_reason",
    "dirty",
)


class GraphStorage:
    """封装 Kùzu 的基本写入与查询。"""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = kuzu.Database(str(self.db_path))
        self.conn = kuzu.Connection(self.db)
        self._ensure_schema()

    def close(self) -> None:
        self.conn.close()
        self.db.close()

    def _table_columns(self, table: str) -> set[str]:
        result = self.conn.execute(f"CALL table_info('{table}') RETURN *;")
        return {row[1] for row in result}

    def _ensure_columns(self, table: str, columns: dict[str, str]) -> None:
        existing = self._table_columns(table)
        for name, type_ in columns.items():
            if name in existing:
                continue
            self.conn.execute(f"ALTER TABLE {table} ADD {name} {type_};")

    def _ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE NODE TABLE IF NOT EXISTS Root(
                id STRING,
                branch_id STRING,
                logline STRING,
                theme STRING,
                ending STRING,
                PRIMARY KEY (id)
            );
            """
        )
        self.conn.execute(
            """
            CREATE NODE TABLE IF NOT EXISTS Act(
                id STRING,
                origin_id STRING,
                branch_id STRING,
                root_id STRING,
                act_index INT,
                disaster STRING,
                PRIMARY KEY (id)
            );
            """
        )
        self.conn.execute(
            """
            CREATE NODE TABLE IF NOT EXISTS Entity(
                id STRING,
                origin_id STRING,
                branch_id STRING,
                root_id STRING,
                name STRING,
                entity_type STRING,
                tags_json STRING,
                arc_status STRING,
                semantic_states_json STRING,
                ambition STRING,
                conflict STRING,
                epiphany STRING,
                voice_dna STRING,
                PRIMARY KEY (id)
            );
            """
        )
        self.conn.execute(
            """
            CREATE NODE TABLE IF NOT EXISTS Scene(
                id STRING,
                origin_id STRING,
                branch_id STRING,
                status STRING,
                pov_character_id STRING,
                expected_outcome STRING,
                conflict_type STRING,
                actual_outcome STRING,
                summary STRING,
                rendered_content STRING,
                logic_exception BOOLEAN,
                logic_exception_reason STRING,
                parent_act_id STRING,
                root_id STRING,
                dirty BOOLEAN,
                PRIMARY KEY (id)
            );
            """
        )
        self.conn.execute(
            """
            CREATE REL TABLE IF NOT EXISTS SceneEntity(
                FROM Scene TO Entity,
                branch_id STRING
            );
            """
        )
        self.conn.execute(
            """
            CREATE REL TABLE IF NOT EXISTS SceneNext(
                FROM Scene TO Scene,
                branch_id STRING
            );
            """
        )
        self.conn.execute(
            """
            CREATE REL TABLE IF NOT EXISTS RootContainsAct(
                FROM Root TO Act,
                branch_id STRING
            );
            """
        )
        self.conn.execute(
            """
            CREATE REL TABLE IF NOT EXISTS ActContainsScene(
                FROM Act TO Scene,
                branch_id STRING
            );
            """
        )
        self.conn.execute(
            """
            CREATE REL TABLE IF NOT EXISTS EntityRelation(
                FROM Entity TO Entity,
                branch_id STRING,
                relation_type STRING,
                tension INT
            );
            """
        )
        self.conn.execute(
            """
            CREATE NODE TABLE IF NOT EXISTS Branch(
                id STRING,
                root_id STRING,
                branch_id STRING,
                branch_root_id STRING,
                PRIMARY KEY (id)
            );
            """
        )
        self.conn.execute(
            """
            CREATE NODE TABLE IF NOT EXISTS Commit(
                id STRING,
                parent_id STRING,
                root_id STRING,
                created_at STRING,
                message STRING,
                PRIMARY KEY (id)
            );
            """
        )
        self.conn.execute(
            """
            CREATE NODE TABLE IF NOT EXISTS SceneOrigin(
                id STRING,
                root_id STRING,
                title STRING,
                created_at STRING,
                initial_commit_id STRING,
                sequence_index INT,
                parent_act_id STRING,
                PRIMARY KEY (id)
            );
            """
        )
        self.conn.execute(
            """
            CREATE NODE TABLE IF NOT EXISTS SceneVersion(
                id STRING,
                scene_origin_id STRING,
                commit_id STRING,
                created_at STRING,
                pov_character_id STRING,
                status STRING,
                expected_outcome STRING,
                conflict_type STRING,
                actual_outcome STRING,
                summary STRING,
                rendered_content STRING,
                logic_exception BOOLEAN,
                logic_exception_reason STRING,
                dirty BOOLEAN,
                PRIMARY KEY (id)
            );
            """
        )
        self.conn.execute(
            """
            CREATE NODE TABLE IF NOT EXISTS BranchHead(
                id STRING,
                root_id STRING,
                branch_id STRING,
                head_commit_id STRING,
                fork_point_commit_id STRING,
                version INT,
                PRIMARY KEY (id)
            );
            """
        )
        self.conn.execute(
            """
            CREATE REL TABLE IF NOT EXISTS CommitParent(
                FROM Commit TO Commit
            );
            """
        )
        self.conn.execute(
            """
            CREATE REL TABLE IF NOT EXISTS CommitContainsSceneVersion(
                FROM Commit TO SceneVersion
            );
            """
        )
        self.conn.execute(
            """
            CREATE REL TABLE IF NOT EXISTS BranchPointsTo(
                FROM BranchHead TO Commit
            );
            """
        )
        self.conn.execute(
            """
            CREATE REL TABLE IF NOT EXISTS OriginHasVersion(
                FROM SceneOrigin TO SceneVersion
            );
            """
        )
        self.conn.execute(
            """
            CREATE REL TABLE IF NOT EXISTS BranchContainsCommit(
                FROM Branch TO Commit
            );
            """
        )
        self.conn.execute(
            """
            CREATE REL TABLE IF NOT EXISTS RootContainsSceneOrigin(
                FROM Root TO SceneOrigin
            );
            """
        )

        self._ensure_columns("Root", {"branch_id": "STRING"})
        self._ensure_columns(
            "Act",
            {
                "origin_id": "STRING",
                "branch_id": "STRING",
                "root_id": "STRING",
                "act_index": "INT",
                "disaster": "STRING",
            },
        )
        self._ensure_columns(
            "Entity",
            {
                "origin_id": "STRING",
                "branch_id": "STRING",
                "root_id": "STRING",
                "name": "STRING",
                "entity_type": "STRING",
                "tags_json": "STRING",
                "arc_status": "STRING",
                "semantic_states_json": "STRING",
                "ambition": "STRING",
                "conflict": "STRING",
                "epiphany": "STRING",
                "voice_dna": "STRING",
            },
        )
        self._ensure_columns(
            "Scene",
            {
                "origin_id": "STRING",
                "branch_id": "STRING",
                "status": "STRING",
                "pov_character_id": "STRING",
                "expected_outcome": "STRING",
                "conflict_type": "STRING",
                "actual_outcome": "STRING",
                "summary": "STRING",
                "rendered_content": "STRING",
                "logic_exception": "BOOLEAN",
                "logic_exception_reason": "STRING",
                "parent_act_id": "STRING",
                "root_id": "STRING",
                "dirty": "BOOLEAN",
            },
        )
        self._ensure_columns("SceneEntity", {"branch_id": "STRING"})
        self._ensure_columns("SceneNext", {"branch_id": "STRING"})
        self._ensure_columns("RootContainsAct", {"branch_id": "STRING"})
        self._ensure_columns("ActContainsScene", {"branch_id": "STRING"})
        self._ensure_columns(
            "EntityRelation",
            {"branch_id": "STRING", "relation_type": "STRING", "tension": "INT"},
        )
        self._ensure_columns(
            "Branch",
            {
                "root_id": "STRING",
                "branch_id": "STRING",
                "branch_root_id": "STRING",
                "parent_branch_id": "STRING",
                "fork_scene_origin_id": "STRING",
                "fork_commit_id": "STRING",
            },
        )


    def run_backfill_migrations(self) -> None:
        self._backfill_branch_metadata()
        self._backfill_origin_ids()
        self._backfill_required_fields()
        self._backfill_scene_entity_links()
        self._backfill_entity_relation_tension()

    def migrate_scene_versions(self, *, root_id: str | None = None) -> dict[str, int]:
        self._backfill_origin_ids()
        scenes = self._collect_legacy_scenes(root_id=root_id)
        if not scenes:
            return {"migrated_roots": 0, "skipped_roots": 0}

        scene_by_id = {scene["id"]: scene for scene in scenes}
        edges_by_root_branch = self._collect_scene_next_edges(
            root_id=root_id, scene_by_id=scene_by_id
        )
        scenes_by_root: dict[str, list[dict[str, Any]]] = {}
        scenes_by_root_branch: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for scene in scenes:
            scenes_by_root.setdefault(scene["root_id"], []).append(scene)
            scenes_by_root_branch.setdefault(
                (scene["root_id"], scene["branch_id"]), []
            ).append(scene)

        migrated_roots = 0
        skipped_roots = 0
        for current_root_id, root_scenes in scenes_by_root.items():
            branch_ids = {scene["branch_id"] for scene in root_scenes}
            if DEFAULT_BRANCH_ID not in branch_ids:
                raise ValueError(
                    "DEFAULT_BRANCH_MISSING: "
                    f"root_id={current_root_id} branch_id={DEFAULT_BRANCH_ID}"
                )
            order_by_branch: dict[str, list[str]] = {}
            for branch_id in branch_ids:
                branch_scenes = scenes_by_root_branch[(current_root_id, branch_id)]
                branch_scene_ids = [scene["id"] for scene in branch_scenes]
                branch_edges = edges_by_root_branch.get(
                    (current_root_id, branch_id), []
                )
                order_by_branch[branch_id] = self._build_scene_chain(
                    root_id=current_root_id,
                    branch_id=branch_id,
                    scene_ids=branch_scene_ids,
                    edges=branch_edges,
                )

            default_order = order_by_branch[DEFAULT_BRANCH_ID]
            default_origin_order: list[str] = []
            origin_ids: set[str] = set()
            default_scene_by_origin: dict[str, dict[str, Any]] = {}
            for scene_id in default_order:
                scene = scene_by_id[scene_id]
                origin_id = scene["origin_id"]
                if origin_id in origin_ids:
                    raise ValueError(
                        "DUPLICATE_SCENE_ORIGIN: "
                        f"root_id={current_root_id} origin_id={origin_id}"
                    )
                origin_ids.add(origin_id)
                default_origin_order.append(origin_id)
                default_scene_by_origin[origin_id] = scene

            for branch_id, scene_order in order_by_branch.items():
                if branch_id == DEFAULT_BRANCH_ID:
                    continue
                branch_origin_order: list[str] = []
                for scene_id in scene_order:
                    scene = scene_by_id[scene_id]
                    origin_id = scene["origin_id"]
                    if origin_id not in origin_ids:
                        raise ValueError(
                            "SCENE_ORIGIN_MISMATCH: "
                            f"root_id={current_root_id} branch_id={branch_id} "
                            f"origin_id={origin_id}"
                        )
                    branch_origin_order.append(origin_id)
                if branch_origin_order != default_origin_order:
                    raise ValueError(
                        "SCENE_ORDER_MISMATCH: "
                        f"root_id={current_root_id} branch_id={branch_id}"
                    )

            origin_rows = self.conn.execute(
                (
                    "MATCH (o:SceneOrigin) "
                    f"WHERE o.root_id = '{self._esc(current_root_id)}' "
                    "RETURN o.id;"
                )
            )
            existing_origin_ids = {row[0] for row in origin_rows}
            if existing_origin_ids:
                expected_scene_count = len(root_scenes)
                self._assert_scene_version_migration_consistent(
                    root_id=current_root_id,
                    origin_ids=origin_ids,
                    expected_scene_count=expected_scene_count,
                    branch_ids=branch_ids,
                )
                skipped_roots += 1
                continue
            head_rows = self.conn.execute(
                (
                    "MATCH (h:BranchHead) "
                    f"WHERE h.root_id = '{self._esc(current_root_id)}' "
                    "RETURN h.id LIMIT 1;"
                )
            )
            if next(iter(head_rows), None):
                raise ValueError(
                    "MIGRATION_INCONSISTENT: BranchHead exists without SceneOrigin "
                    f"root_id={current_root_id}"
                )
            commit_rows = self.conn.execute(
                (
                    "MATCH (c:Commit) "
                    f"WHERE c.root_id = '{self._esc(current_root_id)}' "
                    "RETURN c.id LIMIT 1;"
                )
            )
            if next(iter(commit_rows), None):
                raise ValueError(
                    "MIGRATION_INCONSISTENT: Commit exists without SceneOrigin "
                    f"root_id={current_root_id}"
                )

            initial_commit_id = str(uuid4())
            created_at = self._now_iso()
            self.conn.execute(
                (
                    "CREATE (:Commit {"
                    f"id: '{self._esc(initial_commit_id)}', "
                    "parent_id: NULL, "
                    f"root_id: '{self._esc(current_root_id)}', "
                    f"created_at: '{self._esc(created_at)}', "
                    "message: 'migration initial'"
                    "});"
                )
            )

            for branch_id in branch_ids:
                branch_key = self._branch_key(
                    root_id=current_root_id, branch_id=branch_id
                )
                self.conn.execute(
                    (
                        "MERGE (b:Branch {"
                        f"id: '{self._esc(branch_key)}'"
                        "}) SET "
                        f"b.root_id = '{self._esc(current_root_id)}', "
                        f"b.branch_id = '{self._esc(branch_id)}', "
                        f"b.branch_root_id = '{self._esc(current_root_id)}', "
                        f"b.parent_branch_id = {self._sql_str(None)}, "
                        f"b.fork_scene_origin_id = {self._sql_str(None)}, "
                        f"b.fork_commit_id = {self._sql_str(None)};"
                    )
                )

            default_head_id = self._branch_head_key(
                root_id=current_root_id, branch_id=DEFAULT_BRANCH_ID
            )
            self.conn.execute(
                (
                    "CREATE (:BranchHead {"
                    f"id: '{self._esc(default_head_id)}', "
                    f"root_id: '{self._esc(current_root_id)}', "
                    f"branch_id: '{self._esc(DEFAULT_BRANCH_ID)}', "
                    f"head_commit_id: '{self._esc(initial_commit_id)}', "
                    "fork_point_commit_id: NULL, "
                    "version: 1"
                    "});"
                )
            )
            default_branch_key = self._branch_key(
                root_id=current_root_id, branch_id=DEFAULT_BRANCH_ID
            )
            self.conn.execute(
                (
                    "MATCH (b:Branch), (c:Commit) "
                    f"WHERE b.id = '{self._esc(default_branch_key)}' "
                    f"AND c.id = '{self._esc(initial_commit_id)}' "
                    "CREATE (b)-[:BranchContainsCommit]->(c);"
                )
            )
            self.conn.execute(
                (
                    "MATCH (h:BranchHead), (c:Commit) "
                    f"WHERE h.id = '{self._esc(default_head_id)}' "
                    f"AND c.id = '{self._esc(initial_commit_id)}' "
                    "CREATE (h)-[:BranchPointsTo]->(c);"
                )
            )

            def create_scene_version(
                *, scene: dict[str, Any], scene_origin_id: str, commit_id: str
            ) -> None:
                version_id = str(uuid4())
                self.conn.execute(
                    (
                        "CREATE (:SceneVersion {"
                        f"id: '{self._esc(version_id)}', "
                        f"scene_origin_id: '{self._esc(scene_origin_id)}', "
                        f"commit_id: '{self._esc(commit_id)}', "
                        f"created_at: '{self._esc(self._now_iso())}', "
                        f"pov_character_id: {self._sql_str(scene['pov_character_id'])}, "
                        f"status: {self._sql_str(scene['status'])}, "
                        f"expected_outcome: {self._sql_str(scene['expected_outcome'])}, "
                        f"conflict_type: {self._sql_str(scene['conflict_type'])}, "
                        f"actual_outcome: {self._sql_str(scene['actual_outcome'])}, "
                        f"summary: {self._sql_str(scene['summary'])}, "
                        f"rendered_content: {self._sql_str(scene['rendered_content'])}, "
                        f"logic_exception: {self._sql_bool(scene['logic_exception'])}, "
                        f"logic_exception_reason: {self._sql_str(scene['logic_exception_reason'])}, "
                        f"dirty: {self._sql_bool(scene['dirty'])}"
                        "});"
                    )
                )
                self.conn.execute(
                    (
                        "MATCH (o:SceneOrigin), (v:SceneVersion) "
                        f"WHERE o.id = '{self._esc(scene_origin_id)}' "
                        f"AND v.id = '{self._esc(version_id)}' "
                        "CREATE (o)-[:OriginHasVersion]->(v);"
                    )
                )
                self.conn.execute(
                    (
                        "MATCH (c:Commit), (v:SceneVersion) "
                        f"WHERE c.id = '{self._esc(commit_id)}' "
                        f"AND v.id = '{self._esc(version_id)}' "
                        "CREATE (c)-[:CommitContainsSceneVersion]->(v);"
                    )
                )

            for sequence_index, origin_id in enumerate(default_origin_order, start=1):
                scene = default_scene_by_origin[origin_id]
                self.conn.execute(
                    (
                        "CREATE (:SceneOrigin {"
                        f"id: '{self._esc(origin_id)}', "
                        f"root_id: '{self._esc(current_root_id)}', "
                        "title: NULL, "
                        f"created_at: '{self._esc(self._now_iso())}', "
                        f"initial_commit_id: '{self._esc(initial_commit_id)}', "
                        f"sequence_index: {sequence_index}, "
                        f"parent_act_id: {self._sql_str(scene['parent_act_id'])}"
                        "});"
                    )
                )
                self.conn.execute(
                    (
                        "MATCH (r:Root), (o:SceneOrigin) "
                        f"WHERE r.id = '{self._esc(current_root_id)}' "
                        f"AND o.id = '{self._esc(origin_id)}' "
                        "CREATE (r)-[:RootContainsSceneOrigin]->(o);"
                    )
                )

            default_scenes = scenes_by_root_branch[
                (current_root_id, DEFAULT_BRANCH_ID)
            ]
            for scene in default_scenes:
                create_scene_version(
                    scene=scene,
                    scene_origin_id=scene["origin_id"],
                    commit_id=initial_commit_id,
                )

            for branch_id in branch_ids:
                if branch_id == DEFAULT_BRANCH_ID:
                    continue
                branch_commit_id = str(uuid4())
                self.conn.execute(
                    (
                        "CREATE (:Commit {"
                        f"id: '{self._esc(branch_commit_id)}', "
                        "parent_id: NULL, "
                        f"root_id: '{self._esc(current_root_id)}', "
                        f"created_at: '{self._esc(self._now_iso())}', "
                        f"message: 'migration branch {self._esc(branch_id)}'"
                        "});"
                    )
                )
                head_id = self._branch_head_key(
                    root_id=current_root_id, branch_id=branch_id
                )
                self.conn.execute(
                    (
                        "CREATE (:BranchHead {"
                        f"id: '{self._esc(head_id)}', "
                        f"root_id: '{self._esc(current_root_id)}', "
                        f"branch_id: '{self._esc(branch_id)}', "
                        f"head_commit_id: '{self._esc(branch_commit_id)}', "
                        "fork_point_commit_id: NULL, "
                        "version: 1"
                        "});"
                    )
                )
                branch_key = self._branch_key(
                    root_id=current_root_id, branch_id=branch_id
                )
                self.conn.execute(
                    (
                        "MATCH (b:Branch), (c:Commit) "
                        f"WHERE b.id = '{self._esc(branch_key)}' "
                        f"AND c.id = '{self._esc(branch_commit_id)}' "
                        "CREATE (b)-[:BranchContainsCommit]->(c);"
                    )
                )
                self.conn.execute(
                    (
                        "MATCH (h:BranchHead), (c:Commit) "
                        f"WHERE h.id = '{self._esc(head_id)}' "
                        f"AND c.id = '{self._esc(branch_commit_id)}' "
                        "CREATE (h)-[:BranchPointsTo]->(c);"
                    )
                )
                branch_scenes = scenes_by_root_branch[(current_root_id, branch_id)]
                for scene in branch_scenes:
                    create_scene_version(
                        scene=scene,
                        scene_origin_id=scene["origin_id"],
                        commit_id=branch_commit_id,
                    )

            migrated_roots += 1

        return {"migrated_roots": migrated_roots, "skipped_roots": skipped_roots}

    def rollback_scene_version_migration(
        self, *, root_id: str | None = None
    ) -> dict[str, int]:
        roots: list[str] = []
        if root_id is not None:
            roots = [root_id]
        else:
            root_rows = self.conn.execute(
                "MATCH (o:SceneOrigin) RETURN DISTINCT o.root_id;"
            )
            roots = [row[0] for row in root_rows if row and row[0]]
        if not roots:
            return {"rolled_back_roots": 0}

        for current_root_id in roots:
            legacy_rows = self.conn.execute(
                (
                    "MATCH (s:Scene) "
                    f"WHERE s.root_id = '{self._esc(current_root_id)}' "
                    "RETURN COUNT(*)"
                )
            )
            legacy_count_rows = [row[0] for row in legacy_rows]
            legacy_count = legacy_count_rows[0] if legacy_count_rows else 0
            if legacy_count == 0:
                raise ValueError(
                    "ROLLBACK_NOT_ALLOWED: legacy scenes missing "
                    f"root_id={current_root_id}"
                )
            origin_rows = self.conn.execute(
                (
                    "MATCH (o:SceneOrigin) "
                    f"WHERE o.root_id = '{self._esc(current_root_id)}' "
                    "RETURN o.id LIMIT 1;"
                )
            )
            if not next(iter(origin_rows), None):
                raise ValueError(
                    "ROLLBACK_NOT_ALLOWED: no migrated SceneOrigin "
                    f"root_id={current_root_id}"
                )
            commit_rows = self.conn.execute(
                (
                    "MATCH (c:Commit) "
                    f"WHERE c.root_id = '{self._esc(current_root_id)}' "
                    "RETURN c.id, c.message;"
                )
            )
            for row in commit_rows:
                message = row[1] or ""
                if not str(message).startswith("migration "):
                    raise ValueError(
                        "ROLLBACK_NOT_ALLOWED: non-migration commit found "
                        f"root_id={current_root_id} commit_id={row[0]}"
                    )

            self.conn.execute(
                (
                    "MATCH (o:SceneOrigin)-[:OriginHasVersion]->(v:SceneVersion) "
                    f"WHERE o.root_id = '{self._esc(current_root_id)}' "
                    "DETACH DELETE v;"
                )
            )
            self.conn.execute(
                (
                    "MATCH (o:SceneOrigin) "
                    f"WHERE o.root_id = '{self._esc(current_root_id)}' "
                    "DETACH DELETE o;"
                )
            )
            self.conn.execute(
                (
                    "MATCH (h:BranchHead) "
                    f"WHERE h.root_id = '{self._esc(current_root_id)}' "
                    "DETACH DELETE h;"
                )
            )
            self.conn.execute(
                (
                    "MATCH (b:Branch) "
                    f"WHERE b.root_id = '{self._esc(current_root_id)}' "
                    "DETACH DELETE b;"
                )
            )
            self.conn.execute(
                (
                    "MATCH (c:Commit) "
                    f"WHERE c.root_id = '{self._esc(current_root_id)}' "
                    "DETACH DELETE c;"
                )
            )

        return {"rolled_back_roots": len(roots)}

    def _collect_legacy_scenes(
        self, *, root_id: str | None = None
    ) -> list[dict[str, Any]]:
        where_clause = ""
        if root_id is not None:
            where_clause = f"WHERE s.root_id = '{self._esc(root_id)}' "
        rows = self.conn.execute(
            (
                "MATCH (s:Scene) "
                f"{where_clause}"
                "RETURN s.id, s.origin_id, s.root_id, s.branch_id, "
                "s.status, s.pov_character_id, s.expected_outcome, "
                "s.conflict_type, s.actual_outcome, s.summary, "
                "s.rendered_content, s.logic_exception, s.logic_exception_reason, "
                "s.parent_act_id, s.dirty;"
            )
        )
        scenes: list[dict[str, Any]] = []
        for row in rows:
            scene = {
                "id": row[0],
                "origin_id": row[1],
                "root_id": row[2],
                "branch_id": row[3],
                "status": row[4],
                "pov_character_id": row[5],
                "expected_outcome": row[6],
                "conflict_type": row[7],
                "actual_outcome": row[8],
                "summary": row[9],
                "rendered_content": row[10],
                "logic_exception": row[11],
                "logic_exception_reason": row[12],
                "parent_act_id": row[13],
                "dirty": row[14],
            }
            missing: list[str] = []
            if not scene["id"]:
                missing.append("id")
            if not scene["origin_id"]:
                missing.append("origin_id")
            if not scene["root_id"]:
                missing.append("root_id")
            if not scene["branch_id"]:
                missing.append("branch_id")
            if scene["status"] is None:
                missing.append("status")
            if scene["dirty"] is None:
                missing.append("dirty")
            if scene["pov_character_id"] in (None, ""):
                missing.append("pov_character_id")
            if scene["expected_outcome"] is None:
                missing.append("expected_outcome")
            if scene["conflict_type"] is None:
                missing.append("conflict_type")
            if missing:
                raise ValueError(
                    "SCENE_REQUIRED_FIELDS_MISSING: "
                    f"scene_id={scene['id']} missing={','.join(missing)}"
                )
            scenes.append(scene)
        return scenes

    def _collect_scene_next_edges(
        self, *, root_id: str | None, scene_by_id: dict[str, dict[str, Any]]
    ) -> dict[tuple[str, str], list[tuple[str, str]]]:
        where_clause = ""
        if root_id is not None:
            where_clause = f"WHERE a.root_id = '{self._esc(root_id)}' "
        rows = self.conn.execute(
            (
                "MATCH (a:Scene)-[r:SceneNext]->(b:Scene) "
                f"{where_clause}"
                "RETURN a.id, a.root_id, a.branch_id, "
                "b.id, b.root_id, b.branch_id, r.branch_id;"
            )
        )
        edges_by_root_branch: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for row in rows:
            from_id = row[0]
            from_root_id = row[1]
            from_branch_id = row[2]
            to_id = row[3]
            to_root_id = row[4]
            to_branch_id = row[5]
            rel_branch_id = row[6]
            if from_id not in scene_by_id or to_id not in scene_by_id:
                raise ValueError(
                    "SCENENEXT_INVALID_REFERENCE: "
                    f"from_id={from_id} to_id={to_id}"
                )
            if from_root_id != to_root_id:
                raise ValueError(
                    "SCENENEXT_ROOT_MISMATCH: "
                    f"from_id={from_id} to_id={to_id}"
                )
            if rel_branch_id != from_branch_id or rel_branch_id != to_branch_id:
                raise ValueError(
                    "SCENENEXT_BRANCH_MISMATCH: "
                    f"from_id={from_id} to_id={to_id}"
                )
            edges_by_root_branch.setdefault(
                (from_root_id, from_branch_id), []
            ).append((from_id, to_id))
        return edges_by_root_branch

    def _build_scene_chain(
        self,
        *,
        root_id: str,
        branch_id: str,
        scene_ids: list[str],
        edges: list[tuple[str, str]],
    ) -> list[str]:
        if not scene_ids:
            return []
        next_map: dict[str, str] = {}
        prev_map: dict[str, str] = {}
        for from_id, to_id in edges:
            if from_id not in scene_ids or to_id not in scene_ids:
                raise ValueError(
                    "SCENENEXT_INVALID_REFERENCE: "
                    f"root_id={root_id} branch_id={branch_id} "
                    f"from_id={from_id} to_id={to_id}"
                )
            if from_id in next_map:
                raise ValueError(
                    "SCENENEXT_MULTIPLE_NEXT: "
                    f"root_id={root_id} branch_id={branch_id} scene_id={from_id}"
                )
            if to_id in prev_map:
                raise ValueError(
                    "SCENENEXT_MULTIPLE_PREV: "
                    f"root_id={root_id} branch_id={branch_id} scene_id={to_id}"
                )
            next_map[from_id] = to_id
            prev_map[to_id] = from_id

        start_nodes = [scene_id for scene_id in scene_ids if scene_id not in prev_map]
        if len(scene_ids) > 1 and len(start_nodes) != 1:
            raise ValueError(
                "SCENENEXT_MULTIPLE_STARTS: "
                f"root_id={root_id} branch_id={branch_id} "
                f"count={len(start_nodes)}"
            )
        if len(scene_ids) == 1 and len(start_nodes) != 1:
            raise ValueError(
                "SCENENEXT_CYCLE: "
                f"root_id={root_id} branch_id={branch_id}"
            )
        start_id = start_nodes[0]
        order: list[str] = []
        visited: set[str] = set()
        current = start_id
        while current:
            if current in visited:
                raise ValueError(
                    "SCENENEXT_CYCLE: "
                    f"root_id={root_id} branch_id={branch_id} scene_id={current}"
                )
            visited.add(current)
            order.append(current)
            current = next_map.get(current)
        if len(order) != len(scene_ids):
            raise ValueError(
                "SCENENEXT_INCOMPLETE: "
                f"root_id={root_id} branch_id={branch_id} "
                f"expected={len(scene_ids)} actual={len(order)}"
            )
        return order

    def _assert_scene_version_migration_consistent(
        self,
        *,
        root_id: str,
        origin_ids: set[str],
        expected_scene_count: int,
        branch_ids: set[str],
    ) -> None:
        origin_rows = self.conn.execute(
            (
                "MATCH (o:SceneOrigin) "
                f"WHERE o.root_id = '{self._esc(root_id)}' "
                "RETURN o.id;"
            )
        )
        existing_origin_ids = {row[0] for row in origin_rows}
        missing = origin_ids - existing_origin_ids
        extra = existing_origin_ids - origin_ids
        if missing or extra:
            raise ValueError(
                "MIGRATION_INCONSISTENT: SceneOrigin mismatch "
                f"root_id={root_id} missing={sorted(missing)} extra={sorted(extra)}"
            )
        version_rows = self.conn.execute(
            (
                "MATCH (o:SceneOrigin)-[:OriginHasVersion]->(v:SceneVersion) "
                f"WHERE o.root_id = '{self._esc(root_id)}' "
                "RETURN COUNT(*)"
            )
        )
        version_count_rows = [row[0] for row in version_rows]
        version_count = version_count_rows[0] if version_count_rows else 0
        if version_count != expected_scene_count:
            raise ValueError(
                "MIGRATION_INCONSISTENT: SceneVersion count mismatch "
                f"root_id={root_id} expected={expected_scene_count} "
                f"actual={version_count}"
            )
        commit_rows = self.conn.execute(
            (
                "MATCH (c:Commit) "
                f"WHERE c.root_id = '{self._esc(root_id)}' "
                "RETURN COUNT(*)"
            )
        )
        commit_count_rows = [row[0] for row in commit_rows]
        commit_count = commit_count_rows[0] if commit_count_rows else 0
        if commit_count < len(branch_ids):
            raise ValueError(
                "MIGRATION_INCONSISTENT: Commit count mismatch "
                f"root_id={root_id} expected>={len(branch_ids)} actual={commit_count}"
            )
        for branch_id in branch_ids:
            head_rows = self.conn.execute(
                (
                    "MATCH (h:BranchHead) "
                    f"WHERE h.root_id = '{self._esc(root_id)}' "
                    f"AND h.branch_id = '{self._esc(branch_id)}' "
                    "RETURN h.id LIMIT 1;"
                )
            )
            if not next(iter(head_rows), None):
                raise ValueError(
                    "MIGRATION_INCONSISTENT: BranchHead missing "
                    f"root_id={root_id} branch_id={branch_id}"
                )

    def _backfill_entity_relation_tension(self) -> None:
        self.conn.execute(
            (
                "MATCH (:Entity)-[r:EntityRelation]->(:Entity) "
                f"WHERE r.tension IS NULL SET r.tension = {DEFAULT_RELATION_TENSION};"
            )
        )

    def _backfill_branch_metadata(self) -> None:
        rows = self.conn.execute("MATCH (r:Root) RETURN r.id, r.branch_id;")
        for row in rows:
            root_id = row[0]
            branch_id = row[1]
            if not root_id or not branch_id:
                raise ValueError(
                    "Root missing id/branch_id for branch metadata backfill."
                )
            if branch_id != DEFAULT_BRANCH_ID:
                continue
            branch_key = self._branch_key(root_id=root_id, branch_id=branch_id)
            self.conn.execute(
                (
                    "MERGE (b:Branch {"
                    f"id: '{self._esc(branch_key)}'"
                    "}) SET "
                    f"b.root_id = '{self._esc(root_id)}', "
                    f"b.branch_id = '{self._esc(branch_id)}', "
                    f"b.branch_root_id = '{self._esc(root_id)}';"
                )
            )

    def _backfill_origin_ids(self) -> None:
        for table in ("Act", "Entity", "Scene"):
            self.conn.execute(
                f"MATCH (n:{table}) WHERE n.origin_id IS NULL SET n.origin_id = n.id;"
            )

    def _backfill_required_fields(self) -> None:
        rows = self.conn.execute(
            (
                "MATCH (s:Scene) "
                "WHERE s.branch_id IS NULL OR s.status IS NULL OR s.dirty IS NULL "
                "RETURN s.id, s.branch_id, s.status, s.dirty LIMIT 5;"
            )
        )
        missing: list[str] = []
        for row in rows:
            missing_fields: list[str] = []
            if row[1] is None:
                missing_fields.append("branch_id")
            if row[2] is None:
                missing_fields.append("status")
            if row[3] is None:
                missing_fields.append("dirty")
            if missing_fields:
                missing.append(
                    f"scene_id={row[0]} missing={','.join(missing_fields)}"
                )
        if missing:
            raise ValueError(
                "Scene required fields missing (no backfill): " + "; ".join(missing)
            )

    def _backfill_scene_entity_links(self) -> None:
        rows = self.conn.execute(
            "MATCH (s:Scene) RETURN DISTINCT s.root_id, s.branch_id;"
        )
        targets = [(row[0], row[1]) for row in rows if row and row[0] and row[1]]
        for root_id, branch_id in targets:
            anchor_id = f"{root_id}::{branch_id}::anchor"
            anchor_rows = self.conn.execute(
                (
                    "MATCH (e:Entity) "
                    f"WHERE e.id = '{self._esc(anchor_id)}' "
                    "RETURN e.id LIMIT 1;"
                )
            )
            if not next(iter(anchor_rows), None):
                raise ValueError(
                    "SceneEntity anchor missing: "
                    f"root_id={root_id} branch_id={branch_id} anchor_id={anchor_id}"
                )
            scene_count_result = self.conn.execute(
                (
                    "MATCH (s:Scene) "
                    f"WHERE s.root_id = '{self._esc(root_id)}' "
                    f"AND s.branch_id = '{self._esc(branch_id)}' "
                    "RETURN COUNT(*)"
                )
            )
            scene_counts = [row[0] for row in scene_count_result]
            scene_count = scene_counts[0] if scene_counts else 0
            anchor_count_result = self.conn.execute(
                (
                    "MATCH (s:Scene)-[:SceneEntity]->(e:Entity) "
                    f"WHERE s.root_id = '{self._esc(root_id)}' "
                    f"AND s.branch_id = '{self._esc(branch_id)}' "
                    f"AND e.id = '{self._esc(anchor_id)}' "
                    "RETURN COUNT(*)"
                )
            )
            anchor_counts = [row[0] for row in anchor_count_result]
            anchor_count = anchor_counts[0] if anchor_counts else 0
            if anchor_count != scene_count:
                raise ValueError(
                    "SceneEntity anchor relation missing: "
                    f"root_id={root_id} branch_id={branch_id} "
                    f"expected={scene_count} actual={anchor_count}"
                )

    def save_snowflake(
        self,
        root: SnowflakeRoot,
        characters: Sequence[CharacterSheet],
        scenes: Sequence[SceneNode],
    ) -> str:
        root_id = str(uuid4())
        three_disasters = getattr(root, "three_disasters", None)
        if not three_disasters or len(three_disasters) != 3:
            raise ValueError("SnowflakeRoot.three_disasters must be a list of 3 items")

        created_at = self._now_iso()
        self.conn.execute(
            (
                "CREATE (:Root {"
                f"id: '{root_id}', "
                f"branch_id: '{DEFAULT_BRANCH_ID}', "
                f"logline: '{self._esc(root.logline)}', "
                f"theme: '{self._esc(root.theme)}', "
                f"ending: '{self._esc(root.ending)}'"
                "});"
            )
        )
        branch_key = self._branch_key(root_id=root_id, branch_id=DEFAULT_BRANCH_ID)
        self.conn.execute(
            (
                "CREATE (:Branch {"
                f"id: '{self._esc(branch_key)}', "
                f"root_id: '{self._esc(root_id)}', "
                f"branch_id: '{self._esc(DEFAULT_BRANCH_ID)}', "
                f"branch_root_id: '{self._esc(root_id)}'"
                "});"
            )
        )

        commit_id = str(uuid4())
        self.conn.execute(
            (
                "CREATE (:Commit {"
                f"id: '{self._esc(commit_id)}', "
                "parent_id: NULL, "
                f"root_id: '{self._esc(root_id)}', "
                f"created_at: '{self._esc(created_at)}', "
                "message: 'initial'"
                "});"
            )
        )
        head_id = self._branch_head_key(root_id=root_id, branch_id=DEFAULT_BRANCH_ID)
        self.conn.execute(
            (
                "CREATE (:BranchHead {"
                f"id: '{self._esc(head_id)}', "
                f"root_id: '{self._esc(root_id)}', "
                f"branch_id: '{self._esc(DEFAULT_BRANCH_ID)}', "
                f"head_commit_id: '{self._esc(commit_id)}', "
                "fork_point_commit_id: NULL, "
                "version: 1"
                "});"
            )
        )
        self.conn.execute(
            (
                "MATCH (b:Branch), (c:Commit) "
                f"WHERE b.id = '{self._esc(branch_key)}' "
                f"AND c.id = '{self._esc(commit_id)}' "
                "CREATE (b)-[:BranchContainsCommit]->(c);"
            )
        )
        self.conn.execute(
            (
                "MATCH (h:BranchHead), (c:Commit) "
                f"WHERE h.id = '{self._esc(head_id)}' "
                f"AND c.id = '{self._esc(commit_id)}' "
                "CREATE (h)-[:BranchPointsTo]->(c);"
            )
        )

        acts: list[dict[str, Any]] = []
        for idx, disaster in enumerate(three_disasters, start=1):
            act_id = str(uuid4())
            acts.append({"id": act_id, "act_index": idx, "disaster": disaster})
            self.conn.execute(
                (
                    "CREATE (:Act {"
                    f"id: '{self._esc(act_id)}', "
                    f"origin_id: '{self._esc(act_id)}', "
                    f"branch_id: '{DEFAULT_BRANCH_ID}', "
                    f"root_id: '{self._esc(root_id)}', "
                    f"act_index: {idx}, "
                    f"disaster: '{self._esc(disaster)}'"
                    "});"
                )
            )
            self.conn.execute(
                (
                    "MATCH (r:Root), (a:Act) "
                    f"WHERE r.id = '{self._esc(root_id)}' "
                    f"AND r.branch_id = '{DEFAULT_BRANCH_ID}' "
                    f"AND a.id = '{self._esc(act_id)}' "
                    f"MERGE (r)-[:RootContainsAct {{branch_id: '{DEFAULT_BRANCH_ID}'}}]->(a);"
                )
            )

        for idx, c in enumerate(characters):
            tags_json = "[]"
            semantic_states_json = "{}"
            relation_type = "knows"
            self.conn.execute(
                (
                    "CREATE (:Entity {"
                    f"id: '{c.entity_id}', "
                    f"origin_id: '{c.entity_id}', "
                    f"branch_id: '{DEFAULT_BRANCH_ID}', "
                    f"root_id: '{root_id}', "
                    f"name: '{self._esc(c.name)}', "
                    "entity_type: 'character', "
                    f"tags_json: '{tags_json}', "
                    "arc_status: NULL, "
                    f"semantic_states_json: '{semantic_states_json}', "
                    f"ambition: '{self._esc(c.ambition)}', "
                    f"conflict: '{self._esc(c.conflict)}', "
                    f"epiphany: '{self._esc(c.epiphany)}', "
                    f"voice_dna: '{self._esc(c.voice_dna)}'"
                    "});"
                )
            )
            if idx > 0:
                prev = characters[idx - 1].entity_id
                self.conn.execute(
                    (
                        "MATCH (a:Entity), (b:Entity) "
                        f"WHERE a.id = '{prev}' AND b.id = '{c.entity_id}' "
                        f"CREATE (a)-[:EntityRelation {{branch_id: '{DEFAULT_BRANCH_ID}', relation_type: '{relation_type}', tension: {DEFAULT_RELATION_TENSION}}}]->(b);"
                    )
                )

        if len(characters) == 1:
            only = characters[0].entity_id
            self.conn.execute(
                (
                    "MATCH (a:Entity) "
                    f"WHERE a.id = '{only}' "
                    f"MERGE (a)-[:EntityRelation {{branch_id: '{DEFAULT_BRANCH_ID}', relation_type: 'knows', tension: {DEFAULT_RELATION_TENSION}}}]->(a);"
                )
            )

        total_scenes = len(scenes)
        for scene_index, s in enumerate(scenes):
            if s.branch_id != DEFAULT_BRANCH_ID:
                raise ValueError(
                    f"Scene branch_id must be {DEFAULT_BRANCH_ID!r} for persistence"
                )
            if s.pov_character_id in (None, ""):
                raise ValueError("Scene pov_character_id is required for persistence")
            act_idx = (scene_index * 3) // max(total_scenes, 1)
            assigned_act = acts[act_idx]
            s.parent_act_id = UUID(assigned_act["id"])
            origin_id = str(s.id)
            origin_created_at = self._now_iso()
            self.conn.execute(
                (
                    "CREATE (:SceneOrigin {"
                    f"id: '{self._esc(origin_id)}', "
                    f"root_id: '{self._esc(root_id)}', "
                    "title: NULL, "
                    f"created_at: '{self._esc(origin_created_at)}', "
                    f"initial_commit_id: '{self._esc(commit_id)}', "
                    f"sequence_index: {scene_index + 1}, "
                    f"parent_act_id: {self._sql_str(str(s.parent_act_id))}"
                    "});"
                )
            )
            self.conn.execute(
                (
                    "MATCH (r:Root), (o:SceneOrigin) "
                    f"WHERE r.id = '{self._esc(root_id)}' "
                    f"AND o.id = '{self._esc(origin_id)}' "
                    "CREATE (r)-[:RootContainsSceneOrigin]->(o);"
                )
            )
            version_id = str(uuid4())
            version_created_at = self._now_iso()
            self.conn.execute(
                (
                    "CREATE (:SceneVersion {"
                    f"id: '{self._esc(version_id)}', "
                    f"scene_origin_id: '{self._esc(origin_id)}', "
                    f"commit_id: '{self._esc(commit_id)}', "
                    f"created_at: '{self._esc(version_created_at)}', "
                    f"pov_character_id: '{self._esc(str(s.pov_character_id))}', "
                    f"status: '{DEFAULT_SCENE_STATUS}', "
                    f"expected_outcome: '{self._esc(s.expected_outcome)}', "
                    f"conflict_type: '{self._esc(s.conflict_type)}', "
                    f"actual_outcome: '{self._esc(s.actual_outcome)}', "
                    "summary: NULL, "
                    "rendered_content: NULL, "
                    f"logic_exception: {self._sql_bool(bool(s.logic_exception))}, "
                    "logic_exception_reason: NULL, "
                    f"dirty: {self._sql_bool(bool(s.is_dirty))}"
                    "});"
                )
            )
            self.conn.execute(
                (
                    "MATCH (o:SceneOrigin), (v:SceneVersion) "
                    f"WHERE o.id = '{self._esc(origin_id)}' "
                    f"AND v.id = '{self._esc(version_id)}' "
                    "CREATE (o)-[:OriginHasVersion]->(v);"
                )
            )
            self.conn.execute(
                (
                    "MATCH (c:Commit), (v:SceneVersion) "
                    f"WHERE c.id = '{self._esc(commit_id)}' "
                    f"AND v.id = '{self._esc(version_id)}' "
                    "CREATE (c)-[:CommitContainsSceneVersion]->(v);"
                )
            )

        return root_id

    @staticmethod
    def _esc(value: str) -> str:
        return value.replace("'", "''")

    @staticmethod
    def _branch_key(*, root_id: str, branch_id: str) -> str:
        return f"{root_id}::{branch_id}"

    @staticmethod
    def _branch_head_key(*, root_id: str, branch_id: str) -> str:
        return f"{root_id}::{branch_id}::head"

    @staticmethod
    def _now_iso() -> str:
        return datetime.utcnow().isoformat()

    def _validate_branch_name(self, branch_id: str) -> None:
        if not branch_id.strip():
            raise ValueError("branch_id must not be blank")
        if not _BRANCH_NAME_PATTERN.fullmatch(branch_id):
            raise ValueError("INVALID_BRANCH_NAME")

    def _sql_str(self, value: str | None) -> str:
        if value is None:
            return "NULL"
        return f"'{self._esc(str(value))}'"

    @staticmethod
    def _sql_bool(value: bool | None) -> str:
        if value is None:
            return "NULL"
        return "true" if bool(value) else "false"

    def _resolve_root_node_id(self, *, root_id: str, branch_id: str) -> str:
        root_rows = self.conn.execute(
            (
                "MATCH (r:Root) "
                f"WHERE r.id = '{self._esc(root_id)}' "
                "RETURN r.id LIMIT 1;"
            )
        )
        if not next(iter(root_rows), None):
            raise KeyError(f"Root not found: root_id={root_id} branch_id={branch_id}")
        return root_id

    def branch_exists(self, *, root_id: str, branch_id: str) -> bool:
        result = self.conn.execute(
            (
                "MATCH (b:Branch) "
                f"WHERE b.root_id = '{self._esc(root_id)}' "
                f"AND b.branch_id = '{self._esc(branch_id)}' "
                "RETURN b.id LIMIT 1;"
            )
        )
        return next(iter(result), None) is not None

    def require_branch(self, *, root_id: str, branch_id: str) -> None:
        self._validate_branch_name(branch_id)
        if not self.branch_exists(root_id=root_id, branch_id=branch_id):
            raise KeyError(
                f"BRANCH_NOT_FOUND: Branch not found root_id={root_id} branch_id={branch_id}"
            )

    def create_branch(self, *, root_id: str, branch_id: str) -> None:
        self._validate_branch_name(branch_id)
        self.require_root(root_id=root_id, branch_id=DEFAULT_BRANCH_ID)
        if self.branch_exists(root_id=root_id, branch_id=branch_id):
            raise ValueError(
                f"BRANCH_ALREADY_EXISTS: Branch already exists root_id={root_id} branch_id={branch_id}"
            )
        head = self.get_branch_head(root_id=root_id, branch_id=DEFAULT_BRANCH_ID)
        self.fork_from_commit(
            root_id=root_id,
            source_commit_id=head["head_commit_id"],
            new_branch_id=branch_id,
            parent_branch_id=DEFAULT_BRANCH_ID,
        )

    def get_branch_head(self, *, root_id: str, branch_id: str) -> dict[str, Any]:
        result = self.conn.execute(
            (
                "MATCH (h:BranchHead) "
                f"WHERE h.root_id = '{self._esc(root_id)}' "
                f"AND h.branch_id = '{self._esc(branch_id)}' "
                "RETURN h.id, h.head_commit_id, h.fork_point_commit_id, h.version LIMIT 1;"
            )
        )
        row = next(iter(result), None)
        if not row:
            raise KeyError(
                f"Branch head not found: root_id={root_id} branch_id={branch_id}"
            )
        return {
            "id": row[0],
            "head_commit_id": row[1],
            "fork_point_commit_id": row[2],
            "version": row[3],
        }

    def _get_commit_summary(self, *, root_id: str, commit_id: str) -> dict[str, Any]:
        result = self.conn.execute(
            (
                "MATCH (c:Commit) "
                f"WHERE c.id = '{self._esc(commit_id)}' "
                "RETURN c.id, c.parent_id, c.root_id, c.created_at, c.message LIMIT 1;"
            )
        )
        row = next(iter(result), None)
        if not row or row[2] != root_id:
            raise KeyError(
                f"COMMIT_NOT_FOUND: Commit not found root_id={root_id} commit_id={commit_id}"
            )
        return {
            "id": row[0],
            "parent_id": row[1],
            "root_id": row[2],
            "created_at": row[3],
            "message": row[4],
        }

    def reset_branch_head(
        self, *, root_id: str, branch_id: str, commit_id: str
    ) -> None:
        self.require_branch(root_id=root_id, branch_id=branch_id)
        head = self.get_branch_head(root_id=root_id, branch_id=branch_id)
        self._get_commit_summary(root_id=root_id, commit_id=commit_id)
        if not self._is_ancestor(
            ancestor_commit_id=commit_id, descendant_commit_id=head["head_commit_id"]
        ):
            raise ValueError("INVALID_RESET")
        fork_point_id = head["fork_point_commit_id"]
        if fork_point_id and not self._is_ancestor(
            ancestor_commit_id=fork_point_id, descendant_commit_id=commit_id
        ):
            raise ValueError("INVALID_RESET")
        self._update_branch_head(
            root_id=root_id,
            branch_id=branch_id,
            new_commit_id=commit_id,
            expected_version=head["version"],
        )

    def get_branch_history(
        self, *, root_id: str, branch_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        self.require_branch(root_id=root_id, branch_id=branch_id)
        head = self.get_branch_head(root_id=root_id, branch_id=branch_id)
        commits: list[dict[str, Any]] = []
        current = head["head_commit_id"]
        while current and len(commits) < limit:
            commit = self._get_commit_summary(root_id=root_id, commit_id=current)
            commits.append(commit)
            current = commit["parent_id"]
        return commits

    def get_commit_detail(self, *, root_id: str, commit_id: str) -> dict[str, Any]:
        commit = self._get_commit_summary(root_id=root_id, commit_id=commit_id)
        rows = self.conn.execute(
            (
                "MATCH (c:Commit)-[:CommitContainsSceneVersion]->(sv:SceneVersion) "
                f"WHERE c.id = '{self._esc(commit_id)}' "
                "RETURN sv.id, sv.scene_origin_id, sv.commit_id, sv.created_at, "
                "sv.pov_character_id, sv.status, sv.expected_outcome, "
                "sv.conflict_type, sv.actual_outcome, sv.summary, "
                "sv.rendered_content, sv.logic_exception, "
                "sv.logic_exception_reason, sv.dirty "
                "ORDER BY sv.created_at DESC;"
            )
        )
        scene_versions = [self._build_scene_version(row) for row in rows]
        return {"commit": commit, "scene_versions": scene_versions}

    def list_scene_versions(self, *, scene_origin_id: str) -> list[dict[str, Any]]:
        self._get_scene_origin(scene_origin_id=scene_origin_id)
        result = self.conn.execute(
            (
                "MATCH (o:SceneOrigin)-[:OriginHasVersion]->(sv:SceneVersion) "
                f"WHERE o.id = '{self._esc(scene_origin_id)}' "
                "RETURN sv.id, sv.scene_origin_id, sv.commit_id, sv.created_at, "
                "sv.pov_character_id, sv.status, sv.expected_outcome, "
                "sv.conflict_type, sv.actual_outcome, sv.summary, "
                "sv.rendered_content, sv.logic_exception, "
                "sv.logic_exception_reason, sv.dirty "
                "ORDER BY sv.created_at DESC;"
            )
        )
        return [self._build_scene_version(row) for row in result]

    def gc_orphan_commits(self, *, retention_days: int) -> dict[str, list[str]]:
        if retention_days < 0:
            raise ValueError("retention_days must be >= 0")
        cutoff = datetime.utcnow() - timedelta(days=retention_days)
        head_rows = self.conn.execute(
            "MATCH (h:BranchHead) RETURN DISTINCT h.head_commit_id;"
        )
        reachable: set[str] = set()
        for row in head_rows:
            head_commit_id = row[0]
            if not head_commit_id:
                continue
            current = head_commit_id
            while current:
                if current in reachable:
                    break
                reachable.add(current)
                current = self._get_commit_parent_id(current)

        commit_rows = self.conn.execute("MATCH (c:Commit) RETURN c.id, c.created_at;")
        orphan_commit_ids: list[str] = []
        for row in commit_rows:
            commit_id = row[0]
            created_at = row[1]
            if commit_id in reachable:
                continue
            if created_at is None:
                raise ValueError(f"Commit created_at missing: commit_id={commit_id}")
            commit_time = datetime.fromisoformat(created_at)
            if commit_time < cutoff:
                orphan_commit_ids.append(commit_id)

        for commit_id in orphan_commit_ids:
            self.conn.execute(
                (
                    "MATCH (c:Commit) "
                    f"WHERE c.id = '{self._esc(commit_id)}' "
                    "DETACH DELETE c;"
                )
            )

        orphan_scene_versions: list[str] = []
        scene_rows = self.conn.execute(
            (
                "MATCH (sv:SceneVersion) "
                "WHERE NOT EXISTS { MATCH (:Commit)-[:CommitContainsSceneVersion]->(sv) } "
                "RETURN sv.id;"
            )
        )
        for row in scene_rows:
            scene_version_id = row[0]
            orphan_scene_versions.append(scene_version_id)
            self.conn.execute(
                (
                    "MATCH (sv:SceneVersion) "
                    f"WHERE sv.id = '{self._esc(scene_version_id)}' "
                    "DETACH DELETE sv;"
                )
            )
        return {
            "deleted_commit_ids": orphan_commit_ids,
            "deleted_scene_version_ids": orphan_scene_versions,
        }

    def delete_branch(self, *, root_id: str, branch_id: str) -> None:
        self._remove_branch(root_id=root_id, branch_id=branch_id)

    def fork_from_commit(
        self,
        *,
        root_id: str,
        source_commit_id: str,
        new_branch_id: str,
        parent_branch_id: str | None = None,
        fork_scene_origin_id: str | None = None,
    ) -> None:
        self._validate_branch_name(new_branch_id)
        self.require_root(root_id=root_id, branch_id=DEFAULT_BRANCH_ID)
        if self.branch_exists(root_id=root_id, branch_id=new_branch_id):
            raise ValueError(
                f"BRANCH_ALREADY_EXISTS: Branch already exists root_id={root_id} branch_id={new_branch_id}"
            )
        commit = self._require_commit(root_id=root_id, commit_id=source_commit_id)

        branch_key = self._branch_key(root_id=root_id, branch_id=new_branch_id)
        self.conn.execute(
            (
                "CREATE (:Branch {"
                f"id: '{self._esc(branch_key)}', "
                f"root_id: '{self._esc(root_id)}', "
                f"branch_id: '{self._esc(new_branch_id)}', "
                f"branch_root_id: '{self._esc(root_id)}', "
                f"parent_branch_id: {self._sql_str(parent_branch_id)}, "
                f"fork_scene_origin_id: {self._sql_str(fork_scene_origin_id)}, "
                f"fork_commit_id: '{self._esc(commit['id'])}'"
                "});"
            )
        )
        head_id = self._branch_head_key(root_id=root_id, branch_id=new_branch_id)
        self.conn.execute(
            (
                "CREATE (:BranchHead {"
                f"id: '{self._esc(head_id)}', "
                f"root_id: '{self._esc(root_id)}', "
                f"branch_id: '{self._esc(new_branch_id)}', "
                f"head_commit_id: '{self._esc(commit['id'])}', "
                f"fork_point_commit_id: '{self._esc(commit['id'])}', "
                "version: 1"
                "});"
            )
        )
        self.conn.execute(
            (
                "MATCH (b:Branch), (c:Commit) "
                f"WHERE b.id = '{self._esc(branch_key)}' "
                f"AND c.id = '{self._esc(commit['id'])}' "
                "CREATE (b)-[:BranchContainsCommit]->(c);"
            )
        )
        self.conn.execute(
            (
                "MATCH (h:BranchHead), (c:Commit) "
                f"WHERE h.id = '{self._esc(head_id)}' "
                f"AND c.id = '{self._esc(commit['id'])}' "
                "CREATE (h)-[:BranchPointsTo]->(c);"
            )
        )

    def fork_from_scene(
        self,
        *,
        root_id: str,
        source_branch_id: str,
        scene_origin_id: str,
        new_branch_id: str,
        commit_id: str | None = None,
    ) -> None:
        self._validate_branch_name(new_branch_id)
        self.require_branch(root_id=root_id, branch_id=source_branch_id)
        if commit_id is None:
            head = self.get_branch_head(root_id=root_id, branch_id=source_branch_id)
            source_commit_id = head["head_commit_id"]
            require_direct_version = False
        else:
            source_commit_id = commit_id
            require_direct_version = True
        self._require_commit(root_id=root_id, commit_id=source_commit_id)
        origin = self._get_scene_origin(scene_origin_id=scene_origin_id)
        if origin["root_id"] != root_id:
            raise KeyError("SCENE_NOT_FOUND")
        if require_direct_version:
            if not self._find_scene_version_at_commit(
                scene_origin_id=scene_origin_id, commit_id=source_commit_id
            ):
                raise KeyError("SCENE_NOT_FOUND")
        else:
            try:
                self.get_scene_at_commit(
                    scene_origin_id=scene_origin_id, commit_id=source_commit_id
                )
            except KeyError as exc:
                if "Scene not found in commit chain" in str(exc):
                    raise KeyError("SCENE_NOT_FOUND") from None
                raise
        self.fork_from_commit(
            root_id=root_id,
            source_commit_id=source_commit_id,
            new_branch_id=new_branch_id,
            parent_branch_id=source_branch_id,
            fork_scene_origin_id=scene_origin_id,
        )

    def _require_commit(self, *, root_id: str, commit_id: str) -> dict[str, Any]:
        result = self.conn.execute(
            (
                "MATCH (c:Commit) "
                f"WHERE c.id = '{self._esc(commit_id)}' "
                "RETURN c.id, c.parent_id, c.root_id LIMIT 1;"
            )
        )
        row = next(iter(result), None)
        if not row or row[2] != root_id:
            raise KeyError(
                f"COMMIT_NOT_FOUND: Commit not found root_id={root_id} commit_id={commit_id}"
            )
        return {"id": row[0], "parent_id": row[1], "root_id": row[2]}

    def _get_commit_parent_id(self, commit_id: str) -> str | None:
        result = self.conn.execute(
            (
                "MATCH (c:Commit) "
                f"WHERE c.id = '{self._esc(commit_id)}' "
                "RETURN c.parent_id LIMIT 1;"
            )
        )
        row = next(iter(result), None)
        if not row:
            raise KeyError(f"COMMIT_NOT_FOUND: Commit not found commit_id={commit_id}")
        return row[0]

    def _is_ancestor(self, *, ancestor_commit_id: str, descendant_commit_id: str) -> bool:
        current = descendant_commit_id
        while current:
            if current == ancestor_commit_id:
                return True
            current = self._get_commit_parent_id(current)
        return False

    def _update_branch_head(
        self,
        *,
        root_id: str,
        branch_id: str,
        new_commit_id: str,
        expected_version: int,
    ) -> int:
        head_id = self._branch_head_key(root_id=root_id, branch_id=branch_id)
        result = self.conn.execute(
            (
                "MATCH (h:BranchHead) "
                f"WHERE h.id = '{self._esc(head_id)}' "
                f"AND h.version = {expected_version} "
                f"SET h.head_commit_id = '{self._esc(new_commit_id)}', "
                "h.version = h.version + 1 "
                "RETURN h.version LIMIT 1;"
            )
        )
        row = next(iter(result), None)
        if not row:
            raise ValueError("CONCURRENT_MODIFICATION")
        self.conn.execute(
            (
                "MATCH (h:BranchHead)-[rel:BranchPointsTo]->(:Commit) "
                f"WHERE h.id = '{self._esc(head_id)}' "
                "DELETE rel;"
            )
        )
        self.conn.execute(
            (
                "MATCH (h:BranchHead), (c:Commit) "
                f"WHERE h.id = '{self._esc(head_id)}' "
                f"AND c.id = '{self._esc(new_commit_id)}' "
                "CREATE (h)-[:BranchPointsTo]->(c);"
            )
        )
        return row[0]

    def _get_scene_origin(self, *, scene_origin_id: str) -> dict[str, Any]:
        result = self.conn.execute(
            (
                "MATCH (o:SceneOrigin) "
                f"WHERE o.id = '{self._esc(scene_origin_id)}' "
                "RETURN o.root_id, o.sequence_index, o.parent_act_id LIMIT 1;"
            )
        )
        row = next(iter(result), None)
        if not row:
            raise KeyError(
                f"SCENE_NOT_FOUND: Scene origin not found scene_origin_id={scene_origin_id}"
            )
        return {
            "id": scene_origin_id,
            "root_id": row[0],
            "sequence_index": row[1],
            "parent_act_id": row[2],
        }

    def _get_scene_origin_by_sequence(
        self, *, root_id: str, sequence_index: int
    ) -> dict[str, Any] | None:
        result = self.conn.execute(
            (
                "MATCH (o:SceneOrigin) "
                f"WHERE o.root_id = '{self._esc(root_id)}' "
                f"AND o.sequence_index = {sequence_index} "
                "RETURN o.id, o.root_id, o.sequence_index, o.parent_act_id LIMIT 1;"
            )
        )
        row = next(iter(result), None)
        if not row:
            return None
        return {
            "id": row[0],
            "root_id": row[1],
            "sequence_index": row[2],
            "parent_act_id": row[3],
        }

    def _list_scene_origins(self, *, root_id: str) -> list[dict[str, Any]]:
        result = self.conn.execute(
            (
                "MATCH (o:SceneOrigin) "
                f"WHERE o.root_id = '{self._esc(root_id)}' "
                "RETURN o.id, o.sequence_index ORDER BY o.sequence_index;"
            )
        )
        return [{"id": row[0], "sequence_index": row[1]} for row in result]

    def _build_scene_version(self, row: tuple) -> dict[str, Any]:
        return {
            "id": row[0],
            "scene_origin_id": row[1],
            "commit_id": row[2],
            "created_at": row[3],
            "pov_character_id": row[4],
            "status": row[5],
            "expected_outcome": row[6],
            "conflict_type": row[7],
            "actual_outcome": row[8],
            "summary": row[9],
            "rendered_content": row[10],
            "logic_exception": row[11],
            "logic_exception_reason": row[12],
            "dirty": row[13],
        }

    def _find_scene_version_at_commit(
        self, *, scene_origin_id: str, commit_id: str
    ) -> dict[str, Any] | None:
        result = self.conn.execute(
            (
                "MATCH (c:Commit)-[:CommitContainsSceneVersion]->(sv:SceneVersion) "
                f"WHERE c.id = '{self._esc(commit_id)}' "
                f"AND sv.scene_origin_id = '{self._esc(scene_origin_id)}' "
                "RETURN sv.id, sv.scene_origin_id, sv.commit_id, sv.created_at, "
                "sv.pov_character_id, sv.status, sv.expected_outcome, "
                "sv.conflict_type, sv.actual_outcome, sv.summary, "
                "sv.rendered_content, sv.logic_exception, "
                "sv.logic_exception_reason, sv.dirty LIMIT 1;"
            )
        )
        row = next(iter(result), None)
        if not row:
            return None
        return self._build_scene_version(row)

    def get_scene_at_commit(
        self, *, scene_origin_id: str, commit_id: str
    ) -> dict[str, Any]:
        commit_rows = self.conn.execute(
            "MATCH (c:Commit) RETURN c.id, c.parent_id;"
        )
        parent_map = {row[0]: row[1] for row in commit_rows}
        if commit_id not in parent_map:
            raise KeyError(f"COMMIT_NOT_FOUND: Commit not found commit_id={commit_id}")

        version_rows = self.conn.execute(
            (
                "MATCH (sv:SceneVersion) "
                f"WHERE sv.scene_origin_id = '{self._esc(scene_origin_id)}' "
                "RETURN sv.id, sv.scene_origin_id, sv.commit_id, sv.created_at, "
                "sv.pov_character_id, sv.status, sv.expected_outcome, "
                "sv.conflict_type, sv.actual_outcome, sv.summary, "
                "sv.rendered_content, sv.logic_exception, "
                "sv.logic_exception_reason, sv.dirty;"
            )
        )
        versions = {row[2]: self._build_scene_version(row) for row in version_rows}
        current_commit = commit_id
        while current_commit:
            version = versions.get(current_commit)
            if version:
                return version
            parent_id = parent_map.get(current_commit)
            if parent_id is None:
                break
            if parent_id not in parent_map:
                raise KeyError(
                    f"COMMIT_NOT_FOUND: Commit not found commit_id={parent_id}"
                )
            current_commit = parent_id
        raise KeyError(
            f"Scene not found in commit chain: scene_origin_id={scene_origin_id} commit_id={commit_id}"
        )

    def diff_scene_versions(
        self, *, scene_origin_id: str, from_commit_id: str, to_commit_id: str
    ) -> dict[str, dict[str, Any]]:
        from_version = self.get_scene_at_commit(
            scene_origin_id=scene_origin_id, commit_id=from_commit_id
        )
        to_version = self.get_scene_at_commit(
            scene_origin_id=scene_origin_id, commit_id=to_commit_id
        )
        diff: dict[str, dict[str, Any]] = {}
        for field in _SCENE_VERSION_FIELDS:
            if from_version[field] != to_version[field]:
                diff[field] = {
                    "from": from_version[field],
                    "to": to_version[field],
                }
        return diff

    def commit_scene(
        self,
        *,
        root_id: str,
        branch_id: str,
        scene_origin_id: str,
        content: dict[str, Any],
        message: str,
        expected_head_version: int | None = None,
    ) -> dict[str, Any]:
        return self.commit_scenes(
            root_id=root_id,
            branch_id=branch_id,
            changes=[{"scene_origin_id": scene_origin_id, "content": content}],
            message=message,
            expected_head_version=expected_head_version,
        )

    def commit_scenes(
        self,
        *,
        root_id: str,
        branch_id: str,
        changes: Sequence[dict[str, Any]],
        message: str,
        expected_head_version: int | None = None,
    ) -> dict[str, Any]:
        if not changes:
            raise ValueError("changes must not be empty")
        if len(message) > 500:
            raise ValueError("commit message too long")
        self.require_branch(root_id=root_id, branch_id=branch_id)
        head = self.get_branch_head(root_id=root_id, branch_id=branch_id)
        if expected_head_version is not None and expected_head_version != head["version"]:
            raise ValueError("CONCURRENT_MODIFICATION")
        expected_version = head["version"]
        commit_parent_id = head["head_commit_id"]
        self._require_commit(root_id=root_id, commit_id=commit_parent_id)

        seen_scene_ids: set[str] = set()
        pending: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        for change in changes:
            scene_origin_id = change.get("scene_origin_id")
            if not scene_origin_id:
                raise ValueError("scene_origin_id must not be empty")
            if scene_origin_id in seen_scene_ids:
                raise ValueError("scene_origin_id must be unique")
            seen_scene_ids.add(scene_origin_id)
            origin = self._get_scene_origin(scene_origin_id=scene_origin_id)
            if origin["root_id"] != root_id:
                raise KeyError(
                    "Scene origin root mismatch: "
                    f"root_id={root_id} scene_origin_id={scene_origin_id}"
                )
            current = self.get_scene_at_commit(
                scene_origin_id=scene_origin_id, commit_id=commit_parent_id
            )
            merged = {field: current[field] for field in _SCENE_VERSION_FIELDS}
            content = change.get("content", {})
            for key, value in content.items():
                if key not in _SCENE_VERSION_FIELDS:
                    raise ValueError(f"Unsupported scene field: {key}")
                merged[key] = value
            if all(merged[field] == current[field] for field in _SCENE_VERSION_FIELDS):
                continue
            pending.append((scene_origin_id, merged, current))

        if not pending:
            raise ValueError("NO_CHANGES")

        commit_id = str(uuid4())
        created_at = self._now_iso()
        self.conn.execute(
            (
                "CREATE (:Commit {"
                f"id: '{self._esc(commit_id)}', "
                f"parent_id: '{self._esc(commit_parent_id)}', "
                f"root_id: '{self._esc(root_id)}', "
                f"created_at: '{self._esc(created_at)}', "
                f"message: '{self._esc(message)}'"
                "});"
            )
        )
        self.conn.execute(
            (
                "MATCH (p:Commit), (c:Commit) "
                f"WHERE p.id = '{self._esc(commit_parent_id)}' "
                f"AND c.id = '{self._esc(commit_id)}' "
                "CREATE (c)-[:CommitParent]->(p);"
            )
        )

        version_ids: list[str] = []
        for scene_origin_id, merged, _current in pending:
            version_id = str(uuid4())
            version_ids.append(version_id)
            self.conn.execute(
                (
                    "CREATE (:SceneVersion {"
                    f"id: '{self._esc(version_id)}', "
                    f"scene_origin_id: '{self._esc(scene_origin_id)}', "
                    f"commit_id: '{self._esc(commit_id)}', "
                    f"created_at: '{self._esc(self._now_iso())}', "
                    f"pov_character_id: {self._sql_str(merged['pov_character_id'])}, "
                    f"status: {self._sql_str(merged['status'])}, "
                    f"expected_outcome: {self._sql_str(merged['expected_outcome'])}, "
                    f"conflict_type: {self._sql_str(merged['conflict_type'])}, "
                    f"actual_outcome: {self._sql_str(merged['actual_outcome'])}, "
                    f"summary: {self._sql_str(merged['summary'])}, "
                    f"rendered_content: {self._sql_str(merged['rendered_content'])}, "
                    f"logic_exception: {self._sql_bool(merged['logic_exception'])}, "
                    f"logic_exception_reason: {self._sql_str(merged['logic_exception_reason'])}, "
                    f"dirty: {self._sql_bool(merged['dirty'])}"
                    "});"
                )
            )
            self.conn.execute(
                (
                    "MATCH (o:SceneOrigin), (v:SceneVersion) "
                    f"WHERE o.id = '{self._esc(scene_origin_id)}' "
                    f"AND v.id = '{self._esc(version_id)}' "
                    "CREATE (o)-[:OriginHasVersion]->(v);"
                )
            )
            self.conn.execute(
                (
                    "MATCH (c:Commit), (v:SceneVersion) "
                    f"WHERE c.id = '{self._esc(commit_id)}' "
                    f"AND v.id = '{self._esc(version_id)}' "
                    "CREATE (c)-[:CommitContainsSceneVersion]->(v);"
                )
            )

        self._update_branch_head(
            root_id=root_id,
            branch_id=branch_id,
            new_commit_id=commit_id,
            expected_version=expected_version,
        )
        return {"commit_id": commit_id, "scene_version_ids": version_ids}

    def create_scene_origin(
        self,
        *,
        root_id: str,
        branch_id: str,
        title: str,
        parent_act_id: str,
        content: dict[str, Any],
    ) -> dict[str, Any]:
        if not title.strip():
            raise ValueError("title must not be blank")
        if not parent_act_id:
            raise ValueError("parent_act_id must not be blank")
        self.require_root(root_id=root_id, branch_id=branch_id)
        act_rows = self.conn.execute(
            (
                "MATCH (a:Act) "
                f"WHERE a.id = '{self._esc(parent_act_id)}' "
                "RETURN a.root_id, a.branch_id LIMIT 1;"
            )
        )
        act_row = next(iter(act_rows), None)
        if (
            not act_row
            or act_row[0] != root_id
            or act_row[1] not in (branch_id, DEFAULT_BRANCH_ID)
        ):
            raise ValueError("INVALID_SCENE_PARENT_ACT")
        head = self.get_branch_head(root_id=root_id, branch_id=branch_id)
        max_rows = self.conn.execute(
            (
                "MATCH (o:SceneOrigin) "
                f"WHERE o.root_id = '{self._esc(root_id)}' "
                "RETURN MAX(o.sequence_index);"
            )
        )
        max_row = next(iter(max_rows), None)
        max_index = max_row[0] if max_row and max_row[0] is not None else 0
        next_index = int(max_index) + 1

        commit_id = str(uuid4())
        self.conn.execute(
            (
                "CREATE (:Commit {"
                f"id: '{self._esc(commit_id)}', "
                f"parent_id: '{self._esc(head['head_commit_id'])}', "
                f"root_id: '{self._esc(root_id)}', "
                f"created_at: '{self._esc(self._now_iso())}', "
                "message: 'create scene'"
                "});"
            )
        )
        self.conn.execute(
            (
                "MATCH (p:Commit), (c:Commit) "
                f"WHERE p.id = '{self._esc(head['head_commit_id'])}' "
                f"AND c.id = '{self._esc(commit_id)}' "
                "CREATE (c)-[:CommitParent]->(p);"
            )
        )

        origin_id = str(uuid4())
        self.conn.execute(
            (
                "CREATE (:SceneOrigin {"
                f"id: '{self._esc(origin_id)}', "
                f"root_id: '{self._esc(root_id)}', "
                f"title: '{self._esc(title)}', "
                f"created_at: '{self._esc(self._now_iso())}', "
                f"initial_commit_id: '{self._esc(commit_id)}', "
                f"sequence_index: {next_index}, "
                f"parent_act_id: '{self._esc(parent_act_id)}'"
                "});"
            )
        )
        self.conn.execute(
            (
                "MATCH (r:Root), (o:SceneOrigin) "
                f"WHERE r.id = '{self._esc(root_id)}' "
                f"AND o.id = '{self._esc(origin_id)}' "
                "CREATE (r)-[:RootContainsSceneOrigin]->(o);"
            )
        )

        merged = {
            "pov_character_id": content.get("pov_character_id"),
            "status": content.get("status", DEFAULT_SCENE_STATUS),
            "expected_outcome": content.get("expected_outcome"),
            "conflict_type": content.get("conflict_type"),
            "actual_outcome": content.get("actual_outcome", ""),
            "summary": content.get("summary"),
            "rendered_content": content.get("rendered_content"),
            "logic_exception": content.get("logic_exception", False),
            "logic_exception_reason": content.get("logic_exception_reason"),
            "dirty": content.get("dirty", False),
        }
        if not merged["expected_outcome"] or not merged["conflict_type"]:
            raise ValueError("expected_outcome and conflict_type are required")
        if merged["pov_character_id"] in (None, ""):
            raise ValueError("pov_character_id must not be blank")

        version_id = str(uuid4())
        self.conn.execute(
            (
                "CREATE (:SceneVersion {"
                f"id: '{self._esc(version_id)}', "
                f"scene_origin_id: '{self._esc(origin_id)}', "
                f"commit_id: '{self._esc(commit_id)}', "
                f"created_at: '{self._esc(self._now_iso())}', "
                f"pov_character_id: '{self._esc(str(merged['pov_character_id']))}', "
                f"status: {self._sql_str(merged['status'])}, "
                f"expected_outcome: {self._sql_str(merged['expected_outcome'])}, "
                f"conflict_type: {self._sql_str(merged['conflict_type'])}, "
                f"actual_outcome: {self._sql_str(merged['actual_outcome'])}, "
                f"summary: {self._sql_str(merged['summary'])}, "
                f"rendered_content: {self._sql_str(merged['rendered_content'])}, "
                f"logic_exception: {self._sql_bool(merged['logic_exception'])}, "
                f"logic_exception_reason: {self._sql_str(merged['logic_exception_reason'])}, "
                f"dirty: {self._sql_bool(merged['dirty'])}"
                "});"
            )
        )
        self.conn.execute(
            (
                "MATCH (o:SceneOrigin), (v:SceneVersion) "
                f"WHERE o.id = '{self._esc(origin_id)}' "
                f"AND v.id = '{self._esc(version_id)}' "
                "CREATE (o)-[:OriginHasVersion]->(v);"
            )
        )
        self.conn.execute(
            (
                "MATCH (c:Commit), (v:SceneVersion) "
                f"WHERE c.id = '{self._esc(commit_id)}' "
                f"AND v.id = '{self._esc(version_id)}' "
                "CREATE (c)-[:CommitContainsSceneVersion]->(v);"
            )
        )

        self._update_branch_head(
            root_id=root_id,
            branch_id=branch_id,
            new_commit_id=commit_id,
            expected_version=head["version"],
        )
        return {
            "commit_id": commit_id,
            "scene_origin_id": origin_id,
            "scene_version_id": version_id,
        }

    def delete_scene_origin(
        self,
        *,
        root_id: str,
        branch_id: str,
        scene_origin_id: str,
        message: str,
    ) -> dict[str, Any]:
        if not message.strip():
            raise ValueError("message must not be blank")
        return self.commit_scene(
            root_id=root_id,
            branch_id=branch_id,
            scene_origin_id=scene_origin_id,
            content={"status": "archived"},
            message=message,
        )

    def list_branches(self, *, root_id: str) -> list[str]:
        self.require_root(root_id=root_id, branch_id=DEFAULT_BRANCH_ID)
        result = self.conn.execute(
            (
                "MATCH (b:Branch) "
                f"WHERE b.root_id = '{self._esc(root_id)}' "
                "RETURN b.branch_id ORDER BY b.branch_id;"
            )
        )
        return [row[0] for row in result]

    def _remove_branch(self, *, root_id: str, branch_id: str) -> None:
        if branch_id == DEFAULT_BRANCH_ID:
            raise ValueError("branch_id must not be default branch")
        self.require_branch(root_id=root_id, branch_id=branch_id)
        self.conn.execute(
            (
                "MATCH (h:BranchHead) "
                f"WHERE h.root_id = '{self._esc(root_id)}' "
                f"AND h.branch_id = '{self._esc(branch_id)}' "
                "DETACH DELETE h;"
            )
        )
        self.conn.execute(
            (
                "MATCH (b:Branch) "
                f"WHERE b.root_id = '{self._esc(root_id)}' "
                f"AND b.branch_id = '{self._esc(branch_id)}' "
                "DETACH DELETE b;"
            )
        )

    def _purge_branch_data(
        self, *, root_id: str, branch_id: str, branch_root_id: str | None = None
    ) -> None:
        if branch_id == DEFAULT_BRANCH_ID:
            raise ValueError("branch_id must not be default branch")
        self._remove_branch(root_id=root_id, branch_id=branch_id)

    def _apply_branch_snapshot_to_main(self, *, root_id: str, branch_id: str) -> None:
        if branch_id == DEFAULT_BRANCH_ID:
            raise ValueError("branch_id must not be default branch")
        self.require_branch(root_id=root_id, branch_id=branch_id)
        main_head = self.get_branch_head(root_id=root_id, branch_id=DEFAULT_BRANCH_ID)
        branch_head = self.get_branch_head(root_id=root_id, branch_id=branch_id)
        if not self._is_ancestor(
            ancestor_commit_id=main_head["head_commit_id"],
            descendant_commit_id=branch_head["head_commit_id"],
        ):
            raise ValueError("INVALID_MERGE")
        self._update_branch_head(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            new_commit_id=branch_head["head_commit_id"],
            expected_version=main_head["version"],
        )
        self._remove_branch(root_id=root_id, branch_id=branch_id)

    def merge_branch(self, *, root_id: str, branch_id: str) -> None:
        self._apply_branch_snapshot_to_main(root_id=root_id, branch_id=branch_id)

    def revert_branch(self, *, root_id: str, branch_id: str) -> None:
        self._purge_branch_data(root_id=root_id, branch_id=branch_id)

    def require_root(self, *, root_id: str, branch_id: str) -> None:
        root_rows = self.conn.execute(
            (
                "MATCH (r:Root) "
                f"WHERE r.id = '{self._esc(root_id)}' "
                "RETURN r.id LIMIT 1;"
            )
        )
        if not next(iter(root_rows), None):
            raise KeyError(f"Root not found: root_id={root_id} branch_id={branch_id}")

    def create_entity(
        self,
        *,
        root_id: str,
        branch_id: str,
        name: str,
        entity_type: str,
        tags: Iterable[str],
        arc_status: str | None,
        semantic_states: dict[str, Any],
    ) -> str:
        self.require_root(root_id=root_id, branch_id=branch_id)
        entity_branch_id = DEFAULT_BRANCH_ID
        entity_id = str(uuid4())
        arc_status_value = (
            "NULL" if arc_status is None else f"'{self._esc(arc_status)}'"
        )
        self.conn.execute(
            (
                "CREATE (:Entity {"
                f"id: '{entity_id}', "
                f"origin_id: '{entity_id}', "
                f"branch_id: '{self._esc(entity_branch_id)}', "
                f"root_id: '{self._esc(root_id)}', "
                f"name: '{self._esc(name)}', "
                f"entity_type: '{self._esc(entity_type)}', "
                f"tags_json: '{self._esc(json.dumps(list(tags), ensure_ascii=False))}', "
                f"arc_status: {arc_status_value}, "
                f"semantic_states_json: '{self._esc(json.dumps(semantic_states, ensure_ascii=False))}'"
                "});"
            )
        )
        return entity_id

    def get_entity_semantic_states(
        self, *, root_id: str, branch_id: str, entity_id: str
    ) -> dict[str, Any]:
        entity_branch_id = DEFAULT_BRANCH_ID
        result = self.conn.execute(
            (
                "MATCH (e:Entity) "
                f"WHERE e.id = '{self._esc(entity_id)}' "
                f"AND e.root_id = '{self._esc(root_id)}' "
                f"AND e.branch_id = '{self._esc(entity_branch_id)}' "
                "RETURN e.semantic_states_json LIMIT 1;"
            )
        )
        row = next(iter(result), None)
        if not row:
            raise KeyError(
                f"Entity not found: root_id={root_id} branch_id={branch_id} entity_id={entity_id}"
            )
        raw = row[0] or "{}"
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(
                "Entity.semantic_states_json must be a JSON object (dict in Python)."
            )
        return data

    def get_entity_name(
        self, *, root_id: str, branch_id: str, entity_id: str
    ) -> str:
        entity_branch_id = DEFAULT_BRANCH_ID
        result = self.conn.execute(
            (
                "MATCH (e:Entity) "
                f"WHERE e.id = '{self._esc(entity_id)}' "
                f"AND e.root_id = '{self._esc(root_id)}' "
                f"AND e.branch_id = '{self._esc(entity_branch_id)}' "
                "RETURN e.name LIMIT 1;"
            )
        )
        row = next(iter(result), None)
        if not row:
            raise KeyError(
                f"Entity not found: root_id={root_id} branch_id={branch_id} entity_id={entity_id}"
            )
        name = row[0]
        if name is None or not str(name).strip():
            raise ValueError(
                "Entity name missing: "
                f"root_id={root_id} branch_id={branch_id} entity_id={entity_id}"
            )
        return str(name)

    def set_entity_semantic_states(
        self,
        *,
        root_id: str,
        branch_id: str,
        entity_id: str,
        semantic_states: dict[str, Any],
    ) -> None:
        entity_branch_id = DEFAULT_BRANCH_ID
        payload = self._esc(json.dumps(semantic_states, ensure_ascii=False))
        result = self.conn.execute(
            (
                "MATCH (e:Entity) "
                f"WHERE e.id = '{self._esc(entity_id)}' "
                f"AND e.root_id = '{self._esc(root_id)}' "
                f"AND e.branch_id = '{self._esc(entity_branch_id)}' "
                f"SET e.semantic_states_json = '{payload}' "
                "RETURN e.id LIMIT 1;"
            )
        )
        if not next(iter(result), None):
            raise KeyError(
                f"Entity not found: root_id={root_id} branch_id={branch_id} entity_id={entity_id}"
            )

    def apply_semantic_states_patch(
        self,
        *,
        root_id: str,
        branch_id: str,
        entity_id: str,
        patch: dict[str, Any],
    ) -> dict[str, Any]:
        if not patch:
            raise ValueError("semantic_states_patch must not be empty")
        current = self.get_entity_semantic_states(
            root_id=root_id,
            branch_id=branch_id,
            entity_id=entity_id,
        )
        current.update(patch)
        self.set_entity_semantic_states(
            root_id=root_id,
            branch_id=branch_id,
            entity_id=entity_id,
            semantic_states=current,
        )
        return current

    def list_entities(self, *, root_id: str, branch_id: str) -> list[dict[str, Any]]:
        entity_branch_id = DEFAULT_BRANCH_ID
        result = self.conn.execute(
            (
                "MATCH (e:Entity) "
                f"WHERE e.root_id = '{self._esc(root_id)}' "
                f"AND e.branch_id = '{self._esc(entity_branch_id)}' "
                "RETURN e.id, e.name, e.entity_type, e.tags_json, e.arc_status, e.semantic_states_json;"
            )
        )
        entities: list[dict[str, Any]] = []
        for row in result:
            tags_raw = row[3] or "[]"
            semantic_states_raw = row[5] or "{}"
            entities.append(
                {
                    "entity_id": row[0],
                    "name": row[1],
                    "entity_type": row[2],
                    "tags": json.loads(tags_raw),
                    "arc_status": row[4],
                    "semantic_states": json.loads(semantic_states_raw),
                }
            )
        return entities

    def get_root_snapshot(
        self, *, root_id: str, branch_id: str
    ) -> dict[str, Any]:
        self.require_root(root_id=root_id, branch_id=branch_id)
        root_rows = self.conn.execute(
            (
                "MATCH (r:Root) "
                f"WHERE r.id = '{self._esc(root_id)}' "
                "RETURN r.id, r.logline, r.theme, r.ending LIMIT 1;"
            )
        )
        root_row = next(iter(root_rows), None)
        if not root_row:
            raise KeyError(f"Root not found: root_id={root_id} branch_id={branch_id}")

        head = self.get_branch_head(root_id=root_id, branch_id=branch_id)
        scenes: list[dict[str, Any]] = []
        for origin in self._list_scene_origins(root_id=root_id):
            version = self.get_scene_at_commit(
                scene_origin_id=origin["id"], commit_id=head["head_commit_id"]
            )
            scenes.append(
                {
                    "id": origin["id"],
                    "branch_id": branch_id,
                    "status": version["status"],
                    "pov_character_id": version["pov_character_id"],
                    "expected_outcome": version["expected_outcome"],
                    "conflict_type": version["conflict_type"],
                    "actual_outcome": version["actual_outcome"],
                    "logic_exception": version["logic_exception"],
                    "logic_exception_reason": version["logic_exception_reason"],
                    "is_dirty": version["dirty"],
                }
            )

        entity_branch_id = DEFAULT_BRANCH_ID
        characters_rows = self.conn.execute(
            (
                "MATCH (e:Entity) "
                f"WHERE e.root_id = '{self._esc(root_id)}' "
                f"AND e.branch_id = '{self._esc(entity_branch_id)}' "
                "AND e.entity_type = 'character' "
                "RETURN e.id, e.name;"
            )
        )
        characters = [{"entity_id": row[0], "name": row[1]} for row in characters_rows]

        relations_rows = self.conn.execute(
            (
                "MATCH (a:Entity)-[r:EntityRelation]->(b:Entity) "
                f"WHERE a.root_id = '{self._esc(root_id)}' "
                f"AND a.branch_id = '{self._esc(entity_branch_id)}' "
                "RETURN a.id, b.id, r.relation_type, r.tension;"
            )
        )
        relations = [
            {
                "from_entity_id": row[0],
                "to_entity_id": row[1],
                "relation_type": row[2],
                "tension": row[3],
            }
            for row in relations_rows
        ]

        return {
            "root_id": root_id,
            "branch_id": branch_id,
            "logline": root_row[1],
            "theme": root_row[2],
            "ending": root_row[3],
            "characters": characters,
            "scenes": scenes,
            "relations": relations,
        }

    def upsert_entity_relation(
        self,
        *,
        root_id: str,
        branch_id: str,
        from_entity_id: str,
        to_entity_id: str,
        relation_type: str,
        tension: int,
    ) -> None:
        if not 0 <= tension <= 100:
            raise ValueError("tension must be in range 0-100")
        entity_branch_id = DEFAULT_BRANCH_ID
        result = self.conn.execute(
            (
                "MATCH (a:Entity), (b:Entity) "
                f"WHERE a.id = '{self._esc(from_entity_id)}' "
                f"AND b.id = '{self._esc(to_entity_id)}' "
                f"AND a.root_id = '{self._esc(root_id)}' "
                f"AND b.root_id = '{self._esc(root_id)}' "
                f"AND a.branch_id = '{self._esc(entity_branch_id)}' "
                f"AND b.branch_id = '{self._esc(entity_branch_id)}' "
                f"MERGE (a)-[r:EntityRelation {{branch_id: '{self._esc(entity_branch_id)}', relation_type: '{self._esc(relation_type)}'}}]->(b) "
                f"SET r.tension = {tension} "
                "RETURN r.tension LIMIT 1;"
            )
        )
        if not next(iter(result), None):
            raise KeyError(
                "Entity relation upsert failed: "
                f"root_id={root_id} branch_id={branch_id} "
                f"from_entity_id={from_entity_id} to_entity_id={to_entity_id}"
            )

    def reorder_scenes(
        self, *, root_id: str, branch_id: str, scene_ids: Sequence[str]
    ) -> None:
        raise ValueError("REORDER_SCENES_DISABLED")

    def list_structure_tree(self, *, root_id: str, branch_id: str) -> dict[str, Any]:
        self.require_root(root_id=root_id, branch_id=branch_id)
        acts_rows = self.conn.execute(
            (
                "MATCH (a:Act) "
                f"WHERE a.root_id = '{self._esc(root_id)}' "
                f"AND a.branch_id = '{DEFAULT_BRANCH_ID}' "
                "RETURN a.id, a.act_index, a.disaster "
                "ORDER BY a.act_index;"
            )
        )
        acts: list[dict[str, Any]] = []
        for row in acts_rows:
            act_id = row[0]
            scenes_rows = self.conn.execute(
                (
                    "MATCH (o:SceneOrigin) "
                    f"WHERE o.root_id = '{self._esc(root_id)}' "
                    f"AND o.parent_act_id = '{self._esc(act_id)}' "
                    "RETURN o.id ORDER BY o.sequence_index;"
                )
            )
            acts.append(
                {
                    "act_id": act_id,
                    "act_index": row[1],
                    "disaster": row[2],
                    "scene_ids": [scene_row[0] for scene_row in scenes_rows],
                }
            )
        return {"root_id": root_id, "branch_id": branch_id, "acts": acts}

    def get_scene_context(self, *, scene_id: str, branch_id: str) -> dict[str, Any]:
        origin = self._get_scene_origin(scene_origin_id=scene_id)
        root_id = origin["root_id"]
        head = self.get_branch_head(root_id=root_id, branch_id=branch_id)
        current_version = self.get_scene_at_commit(
            scene_origin_id=scene_id, commit_id=head["head_commit_id"]
        )
        expected_outcome = current_version["expected_outcome"]
        if not expected_outcome or not str(expected_outcome).strip():
            raise ValueError(
                f"Scene expected_outcome missing: scene_id={scene_id} branch_id={branch_id}"
            )

        prev_scene_id = None
        summary = ""
        if origin["sequence_index"] and origin["sequence_index"] > 1:
            prev_origin = self._get_scene_origin_by_sequence(
                root_id=root_id, sequence_index=origin["sequence_index"] - 1
            )
            if prev_origin:
                prev_scene_id = prev_origin["id"]
                prev_version = self.get_scene_at_commit(
                    scene_origin_id=prev_scene_id, commit_id=head["head_commit_id"]
                )
                summary = prev_version["summary"]
                if summary is None or not str(summary).strip():
                    raise ValueError(
                        "Scene summary missing for previous scene: "
                        f"scene_id={scene_id} branch_id={branch_id} prev_scene_id={prev_scene_id}"
                    )

        next_origin = self._get_scene_origin_by_sequence(
            root_id=root_id, sequence_index=origin["sequence_index"] + 1
        )

        pov_character_id = current_version["pov_character_id"]
        if pov_character_id in (None, ""):
            raise ValueError(
                "Scene pov_character_id missing: "
                f"scene_id={scene_id} branch_id={branch_id}"
            )
        entity_rows = self.conn.execute(
            (
                "MATCH (e:Entity) "
                f"WHERE e.id = '{self._esc(str(pov_character_id))}' "
                f"AND e.root_id = '{self._esc(root_id)}' "
                f"AND e.branch_id = '{DEFAULT_BRANCH_ID}' "
                "RETURN e.id, e.name, e.entity_type, e.semantic_states_json LIMIT 1;"
            )
        )
        row = next(iter(entity_rows), None)
        if not row:
            raise KeyError(
                f"Entity not found: root_id={root_id} entity_id={pov_character_id}"
            )
        raw_states = row[3] or "{}"
        states = json.loads(raw_states)
        if not isinstance(states, dict):
            raise ValueError(
                "Scene entity semantic_states must be a JSON object: "
                f"scene_id={scene_id} branch_id={branch_id} entity_id={row[0]}"
            )
        scene_entities = [
            {
                "entity_id": row[0],
                "name": row[1],
                "entity_type": row[2],
                "semantic_states": states,
            }
        ]
        semantic_states = {str(row[0]): states}

        root_graph = self.get_root_snapshot(root_id=root_id, branch_id=branch_id)
        return {
            "root_id": root_id,
            "branch_id": branch_id,
            "expected_outcome": expected_outcome,
            "semantic_states": semantic_states,
            "summary": summary,
            "scene_entities": scene_entities,
            "characters": root_graph.get("characters", []),
            "relations": root_graph.get("relations", []),
            "prev_scene_id": prev_scene_id,
            "next_scene_id": next_origin["id"] if next_origin else None,
        }

    def get_scene_snapshot(self, *, scene_id: str, branch_id: str) -> dict[str, Any]:
        origin = self._get_scene_origin(scene_origin_id=scene_id)
        head = self.get_branch_head(root_id=origin["root_id"], branch_id=branch_id)
        version = self.get_scene_at_commit(
            scene_origin_id=scene_id, commit_id=head["head_commit_id"]
        )
        return {
            "scene_id": scene_id,
            "expected_outcome": version["expected_outcome"],
            "summary": version["summary"],
            "rendered_content": version["rendered_content"],
            "is_dirty": version["dirty"],
        }

    def is_scene_logic_exception(
        self, *, root_id: str, branch_id: str, scene_id: str
    ) -> bool:
        origin = self._get_scene_origin(scene_origin_id=scene_id)
        if origin["root_id"] != root_id:
            raise KeyError(
                f"Scene not found: root_id={root_id} branch_id={branch_id} scene_id={scene_id}"
            )
        head = self.get_branch_head(root_id=root_id, branch_id=branch_id)
        version = self.get_scene_at_commit(
            scene_origin_id=scene_id, commit_id=head["head_commit_id"]
        )
        return bool(version["logic_exception"])

    def mark_scene_logic_exception(
        self,
        *,
        root_id: str,
        branch_id: str,
        scene_id: str,
        reason: str,
    ) -> None:
        if not reason.strip():
            raise ValueError("reason must not be blank")
        origin = self._get_scene_origin(scene_origin_id=scene_id)
        if origin["root_id"] != root_id:
            raise KeyError(
                f"Scene not found: root_id={root_id} branch_id={branch_id} scene_id={scene_id}"
            )
        self.commit_scene(
            root_id=root_id,
            branch_id=branch_id,
            scene_origin_id=scene_id,
            content={
                "logic_exception": True,
                "logic_exception_reason": reason,
            },
            message="mark logic exception",
        )

    def complete_scene(
        self,
        *,
        scene_id: str,
        branch_id: str,
        actual_outcome: str,
        summary: str,
    ) -> None:
        if not actual_outcome.strip():
            raise ValueError("actual_outcome must not be blank")
        if not summary.strip():
            raise ValueError("summary must not be blank")
        origin = self._get_scene_origin(scene_origin_id=scene_id)
        self.commit_scene(
            root_id=origin["root_id"],
            branch_id=branch_id,
            scene_origin_id=scene_id,
            content={
                "actual_outcome": actual_outcome,
                "summary": summary,
                "status": "committed",
            },
            message="complete scene",
        )

    def save_scene_render(self, *, scene_id: str, branch_id: str, content: str) -> None:
        if not content.strip():
            raise ValueError("content must not be blank")
        origin = self._get_scene_origin(scene_origin_id=scene_id)
        self.commit_scene(
            root_id=origin["root_id"],
            branch_id=branch_id,
            scene_origin_id=scene_id,
            content={"rendered_content": content},
            message="render scene",
        )

    def apply_local_scene_fix(
        self,
        *,
        root_id: str,
        branch_id: str,
        scene_id: str,
        limit: int = 3,
    ) -> list[str]:
        if not 1 <= limit <= 3:
            raise ValueError("limit must be between 1 and 3")
        origin = self._get_scene_origin(scene_origin_id=scene_id)
        if origin["root_id"] != root_id:
            raise KeyError(
                "Scene not found: "
                f"root_id={root_id} branch_id={branch_id} scene_id={scene_id}"
            )
        head = self.get_branch_head(root_id=root_id, branch_id=branch_id)
        current_version = self.get_scene_at_commit(
            scene_origin_id=scene_id, commit_id=head["head_commit_id"]
        )
        base_outcome = current_version["expected_outcome"]
        if base_outcome is None or not str(base_outcome).strip():
            raise ValueError(
                "Scene expected_outcome missing for local fix: "
                f"scene_id={scene_id} branch_id={branch_id}"
            )

        target_ids: list[str] = []
        changes: list[dict[str, Any]] = []
        for offset in range(1, limit + 1):
            target_origin = self._get_scene_origin_by_sequence(
                root_id=root_id, sequence_index=origin["sequence_index"] + offset
            )
            if not target_origin:
                break
            target_id = target_origin["id"]
            target_version = self.get_scene_at_commit(
                scene_origin_id=target_id, commit_id=head["head_commit_id"]
            )
            target_outcome = target_version["expected_outcome"]
            if target_outcome is None or not str(target_outcome).strip():
                raise ValueError(
                    "Scene expected_outcome missing for local fix: "
                    f"scene_id={target_id} branch_id={branch_id}"
                )
            updated_expected_outcome = (
                f"{target_outcome} [local_fix:{base_outcome}]"
            )
            updated_summary = f"local_fix from {scene_id}: {base_outcome}"
            changes.append(
                {
                    "scene_origin_id": target_id,
                    "content": {
                        "expected_outcome": updated_expected_outcome,
                        "summary": updated_summary,
                        "dirty": True,
                    },
                }
            )
            target_ids.append(target_id)

        if not target_ids:
            return []
        self.commit_scenes(
            root_id=root_id,
            branch_id=branch_id,
            changes=changes,
            message="local scene fix",
        )
        return target_ids

    def mark_next_scenes_dirty(
        self,
        *,
        root_id: str,
        branch_id: str,
        scene_id: str,
        limit: int = 3,
    ) -> list[str]:
        if not 1 <= limit <= 3:
            raise ValueError("limit must be between 1 and 3")
        origin = self._get_scene_origin(scene_origin_id=scene_id)
        if origin["root_id"] != root_id:
            raise KeyError(
                "Scene not found: "
                f"root_id={root_id} branch_id={branch_id} scene_id={scene_id}"
            )

        target_ids: list[str] = []
        changes: list[dict[str, Any]] = []
        for offset in range(1, limit + 1):
            target_origin = self._get_scene_origin_by_sequence(
                root_id=root_id, sequence_index=origin["sequence_index"] + offset
            )
            if not target_origin:
                break
            target_id = target_origin["id"]
            target_ids.append(target_id)
            changes.append({"scene_origin_id": target_id, "content": {"dirty": True}})

        if not target_ids:
            return []
        self.commit_scenes(
            root_id=root_id,
            branch_id=branch_id,
            changes=changes,
            message="mark next scenes dirty",
        )
        return target_ids

    def mark_future_scenes_dirty(
        self,
        *,
        root_id: str,
        branch_id: str,
        scene_id: str,
    ) -> list[str]:
        origin = self._get_scene_origin(scene_origin_id=scene_id)
        if origin["root_id"] != root_id:
            raise KeyError(
                "Scene not found: "
                f"root_id={root_id} branch_id={branch_id} scene_id={scene_id}"
            )

        target_ids: list[str] = []
        changes: list[dict[str, Any]] = []
        offset = 1
        while True:
            target_origin = self._get_scene_origin_by_sequence(
                root_id=root_id, sequence_index=origin["sequence_index"] + offset
            )
            if not target_origin:
                break
            target_id = target_origin["id"]
            target_ids.append(target_id)
            changes.append({"scene_origin_id": target_id, "content": {"dirty": True}})
            offset += 1

        if not target_ids:
            return []
        self.commit_scenes(
            root_id=root_id,
            branch_id=branch_id,
            changes=changes,
            message="mark future scenes dirty",
        )
        return target_ids

    def mark_scene_dirty(self, *, scene_id: str, branch_id: str) -> None:
        origin = self._get_scene_origin(scene_origin_id=scene_id)
        self.commit_scene(
            root_id=origin["root_id"],
            branch_id=branch_id,
            scene_origin_id=scene_id,
            content={"dirty": True},
            message="mark scene dirty",
        )

    def list_dirty_scenes(self, *, root_id: str, branch_id: str) -> list[str]:
        head = self.get_branch_head(root_id=root_id, branch_id=branch_id)
        dirty_ids: list[str] = []
        for origin in self._list_scene_origins(root_id=root_id):
            version = self.get_scene_at_commit(
                scene_origin_id=origin["id"], commit_id=head["head_commit_id"]
            )
            if version["dirty"]:
                dirty_ids.append(origin["id"])
        return dirty_ids

    def count_scenes(self, root_id: str) -> int:
        result = self.conn.execute(
            (
                "MATCH (o:SceneOrigin) "
                f"WHERE o.root_id='{self._esc(root_id)}' "
                "RETURN COUNT(*)"
            )
        )
        rows = [row[0] for row in result]
        return rows[0] if rows else 0

    def count_scene_versions(self, root_id: str) -> int:
        result = self.conn.execute(
            (
                "MATCH (o:SceneOrigin)-[:OriginHasVersion]->(v:SceneVersion) "
                f"WHERE o.root_id='{self._esc(root_id)}' "
                "RETURN COUNT(*)"
            )
        )
        rows = [row[0] for row in result]
        return rows[0] if rows else 0

    def fetch_scene_ids(self, root_id: str) -> list[str]:
        result = self.conn.execute(
            (
                "MATCH (o:SceneOrigin) "
                f"WHERE o.root_id='{self._esc(root_id)}' "
                "RETURN o.id ORDER BY o.sequence_index;"
            )
        )
        return [row[0] for row in result]
