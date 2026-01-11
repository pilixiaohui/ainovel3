"""Kùzu Graph 存储封装，用于持久化雪花结构。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence
from uuid import UUID, uuid4

import kuzu

from app.constants import DEFAULT_BRANCH_ID
from app.models import CharacterSheet, SceneNode, SnowflakeRoot


DEFAULT_SCENE_STATUS = "draft"
DEFAULT_RELATION_TENSION = 0


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
            },
        )

        self._backfill_branch_metadata()
        self._backfill_origin_ids()
        self._backfill_required_fields()
        self._backfill_scene_entity_links()
        self._backfill_entity_relation_tension()

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
            parent = f"'{s.parent_act_id}'"
            self.conn.execute(
                (
                    "CREATE (:Scene {"
                    f"id: '{s.id}', "
                    f"origin_id: '{s.id}', "
                    f"branch_id: '{DEFAULT_BRANCH_ID}', "
                    f"status: '{DEFAULT_SCENE_STATUS}', "
                    f"pov_character_id: '{s.pov_character_id}', "
                    f"expected_outcome: '{self._esc(s.expected_outcome)}', "
                    f"conflict_type: '{self._esc(s.conflict_type)}', "
                    f"actual_outcome: '{self._esc(s.actual_outcome)}', "
                    f"logic_exception: {str(bool(s.logic_exception)).lower()}, "
                    "logic_exception_reason: NULL, "
                    f"parent_act_id: {parent}, "
                    f"root_id: '{root_id}', "
                    f"dirty: {str(bool(s.is_dirty)).lower()}"
                    "});"
                )
            )
            self.conn.execute(
                (
                    "MATCH (a:Act), (s:Scene) "
                    f"WHERE a.id = '{self._esc(assigned_act['id'])}' "
                    f"AND a.branch_id = '{DEFAULT_BRANCH_ID}' "
                    f"AND s.id = '{s.id}' "
                    f"MERGE (a)-[:ActContainsScene {{branch_id: '{DEFAULT_BRANCH_ID}'}}]->(s);"
                )
            )
            self.conn.execute(
                (
                    "MATCH (scene:Scene), (entity:Entity) "
                    f"WHERE scene.id = '{s.id}' "
                    f"AND entity.id = '{s.pov_character_id}' "
                    f"MERGE (scene)-[:SceneEntity {{branch_id: '{DEFAULT_BRANCH_ID}'}}]->(entity);"
                )
            )

        for current, next_ in zip(scenes, scenes[1:]):
            self.conn.execute(
                (
                    "MATCH (a:Scene), (b:Scene) "
                    f"WHERE a.id = '{current.id}' AND b.id = '{next_.id}' "
                    f"MERGE (a)-[:SceneNext {{branch_id: '{DEFAULT_BRANCH_ID}'}}]->(b);"
                )
            )

        return root_id

    @staticmethod
    def _esc(value: str) -> str:
        return value.replace("'", "''")

    @staticmethod
    def _branch_key(*, root_id: str, branch_id: str) -> str:
        return f"{root_id}::{branch_id}"

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
        if branch_id == DEFAULT_BRANCH_ID:
            return root_id
        result = self.conn.execute(
            (
                "MATCH (b:Branch) "
                f"WHERE b.root_id = '{self._esc(root_id)}' "
                f"AND b.branch_id = '{self._esc(branch_id)}' "
                "RETURN b.branch_root_id LIMIT 1;"
            )
        )
        row = next(iter(result), None)
        if not row:
            raise KeyError(f"Branch not found: root_id={root_id} branch_id={branch_id}")
        branch_root_id = row[0]
        if not branch_root_id:
            raise ValueError(
                f"Branch root id missing: root_id={root_id} branch_id={branch_id}"
            )
        return branch_root_id

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
        if not branch_id.strip():
            raise ValueError("branch_id must not be blank")
        if not self.branch_exists(root_id=root_id, branch_id=branch_id):
            raise KeyError(
                f"Branch not found: root_id={root_id} branch_id={branch_id}"
            )

    def create_branch(self, *, root_id: str, branch_id: str) -> None:
        if not branch_id.strip():
            raise ValueError("branch_id must not be blank")
        self.require_root(root_id=root_id, branch_id=DEFAULT_BRANCH_ID)
        if self.branch_exists(root_id=root_id, branch_id=branch_id):
            raise ValueError(
                f"Branch already exists: root_id={root_id} branch_id={branch_id}"
            )

        root_rows = self.conn.execute(
            (
                "MATCH (r:Root) "
                f"WHERE r.id = '{self._esc(root_id)}' "
                f"AND r.branch_id = '{DEFAULT_BRANCH_ID}' "
                "RETURN r.logline, r.theme, r.ending;"
            )
        )
        root_row = next(iter(root_rows), None)
        if not root_row:
            raise KeyError(
                f"Root not found: root_id={root_id} branch_id={DEFAULT_BRANCH_ID}"
            )

        branch_root_id = str(uuid4())
        self.conn.execute(
            (
                "CREATE (:Root {"
                f"id: '{self._esc(branch_root_id)}', "
                f"branch_id: '{self._esc(branch_id)}', "
                f"logline: {self._sql_str(root_row[0])}, "
                f"theme: {self._sql_str(root_row[1])}, "
                f"ending: {self._sql_str(root_row[2])}"
                "});"
            )
        )

        branch_key = self._branch_key(root_id=root_id, branch_id=branch_id)
        self.conn.execute(
            (
                "CREATE (:Branch {"
                f"id: '{self._esc(branch_key)}', "
                f"root_id: '{self._esc(root_id)}', "
                f"branch_id: '{self._esc(branch_id)}', "
                f"branch_root_id: '{self._esc(branch_root_id)}'"
                "});"
            )
        )

        act_rows = self.conn.execute(
            (
                "MATCH (a:Act) "
                f"WHERE a.root_id = '{self._esc(root_id)}' "
                f"AND a.branch_id = '{DEFAULT_BRANCH_ID}' "
                "RETURN a.id, a.act_index, a.disaster;"
            )
        )
        act_map: dict[str, str] = {}
        for row in act_rows:
            old_id = str(row[0])
            new_id = str(uuid4())
            act_map[old_id] = new_id
            self.conn.execute(
                (
                    "CREATE (:Act {"
                    f"id: '{self._esc(new_id)}', "
                    f"origin_id: '{self._esc(old_id)}', "
                    f"branch_id: '{self._esc(branch_id)}', "
                    f"root_id: '{self._esc(root_id)}', "
                    f"act_index: {row[1]}, "
                    f"disaster: {self._sql_str(row[2])}"
                    "});"
                )
            )

        entity_rows = self.conn.execute(
            (
                "MATCH (e:Entity) "
                f"WHERE e.root_id = '{self._esc(root_id)}' "
                f"AND e.branch_id = '{DEFAULT_BRANCH_ID}' "
                "RETURN e.id, e.name, e.entity_type, e.tags_json, e.arc_status, "
                "e.semantic_states_json, e.ambition, e.conflict, e.epiphany, e.voice_dna;"
            )
        )
        entity_map: dict[str, str] = {}
        for row in entity_rows:
            old_id = str(row[0])
            if old_id == f"{root_id}::{DEFAULT_BRANCH_ID}::anchor":
                new_id = f"{root_id}::{branch_id}::anchor"
            else:
                new_id = str(uuid4())
            entity_map[old_id] = new_id
            self.conn.execute(
                (
                    "CREATE (:Entity {"
                    f"id: '{self._esc(new_id)}', "
                    f"origin_id: '{self._esc(old_id)}', "
                    f"branch_id: '{self._esc(branch_id)}', "
                    f"root_id: '{self._esc(root_id)}', "
                    f"name: {self._sql_str(row[1])}, "
                    f"entity_type: {self._sql_str(row[2])}, "
                    f"tags_json: {self._sql_str(row[3])}, "
                    f"arc_status: {self._sql_str(row[4])}, "
                    f"semantic_states_json: {self._sql_str(row[5])}, "
                    f"ambition: {self._sql_str(row[6])}, "
                    f"conflict: {self._sql_str(row[7])}, "
                    f"epiphany: {self._sql_str(row[8])}, "
                    f"voice_dna: {self._sql_str(row[9])}"
                    "});"
                )
            )

        scene_rows = self.conn.execute(
            (
                "MATCH (s:Scene) "
                f"WHERE s.root_id = '{self._esc(root_id)}' "
                f"AND s.branch_id = '{DEFAULT_BRANCH_ID}' "
                "RETURN s.id, s.status, s.pov_character_id, s.expected_outcome, "
                "s.conflict_type, s.actual_outcome, s.logic_exception, "
                "s.logic_exception_reason, s.parent_act_id, s.dirty, s.summary, "
                "s.rendered_content;"
            )
        )
        scene_map: dict[str, str] = {}
        for row in scene_rows:
            old_id = str(row[0])
            new_id = str(uuid4())
            scene_map[old_id] = new_id
            pov_character_id = row[2]
            mapped_pov = (
                entity_map.get(str(pov_character_id)) if pov_character_id else None
            )
            if pov_character_id and not mapped_pov:
                raise ValueError(
                    "Scene pov_character_id missing mapping: "
                    f"root_id={root_id} branch_id={branch_id} entity_id={pov_character_id}"
                )
            parent_act_id = row[8]
            mapped_act = act_map.get(str(parent_act_id)) if parent_act_id else None
            if parent_act_id and not mapped_act:
                raise ValueError(
                    "Scene parent_act_id missing mapping: "
                    f"root_id={root_id} branch_id={branch_id} act_id={parent_act_id}"
                )
            self.conn.execute(
                (
                    "CREATE (:Scene {"
                    f"id: '{self._esc(new_id)}', "
                    f"origin_id: '{self._esc(old_id)}', "
                    f"branch_id: '{self._esc(branch_id)}', "
                    f"status: {self._sql_str(row[1])}, "
                    f"pov_character_id: {self._sql_str(mapped_pov)}, "
                    f"expected_outcome: {self._sql_str(row[3])}, "
                    f"conflict_type: {self._sql_str(row[4])}, "
                    f"actual_outcome: {self._sql_str(row[5])}, "
                    f"logic_exception: {self._sql_bool(row[6])}, "
                    f"logic_exception_reason: {self._sql_str(row[7])}, "
                    f"parent_act_id: {self._sql_str(mapped_act)}, "
                    f"root_id: '{self._esc(root_id)}', "
                    f"dirty: {self._sql_bool(row[9])}, "
                    f"summary: {self._sql_str(row[10])}, "
                    f"rendered_content: {self._sql_str(row[11])}"
                    "});"
                )
            )

        for old_act_id, new_act_id in act_map.items():
            self.conn.execute(
                (
                    "MATCH (r:Root), (a:Act) "
                    f"WHERE r.id = '{self._esc(branch_root_id)}' "
                    f"AND r.branch_id = '{self._esc(branch_id)}' "
                    f"AND a.id = '{self._esc(new_act_id)}' "
                    f"MERGE (r)-[:RootContainsAct {{branch_id: '{self._esc(branch_id)}'}}]->(a);"
                )
            )

        act_scene_rows = self.conn.execute(
            (
                "MATCH (a:Act)-[:ActContainsScene]->(s:Scene) "
                f"WHERE a.root_id = '{self._esc(root_id)}' "
                f"AND a.branch_id = '{DEFAULT_BRANCH_ID}' "
                "RETURN a.id, s.id;"
            )
        )
        for row in act_scene_rows:
            new_act_id = act_map.get(str(row[0]))
            new_scene_id = scene_map.get(str(row[1]))
            if not new_act_id or not new_scene_id:
                raise ValueError(
                    "Act/Scene relation missing mapping: "
                    f"root_id={root_id} branch_id={branch_id}"
                )
            self.conn.execute(
                (
                    "MATCH (a:Act), (s:Scene) "
                    f"WHERE a.id = '{self._esc(new_act_id)}' "
                    f"AND s.id = '{self._esc(new_scene_id)}' "
                    f"MERGE (a)-[:ActContainsScene {{branch_id: '{self._esc(branch_id)}'}}]->(s);"
                )
            )

        scene_entity_rows = self.conn.execute(
            (
                "MATCH (s:Scene)-[:SceneEntity]->(e:Entity) "
                f"WHERE s.root_id = '{self._esc(root_id)}' "
                f"AND s.branch_id = '{DEFAULT_BRANCH_ID}' "
                "RETURN s.id, e.id;"
            )
        )
        for row in scene_entity_rows:
            new_scene_id = scene_map.get(str(row[0]))
            new_entity_id = entity_map.get(str(row[1]))
            if not new_scene_id or not new_entity_id:
                raise ValueError(
                    "Scene/Entity relation missing mapping: "
                    f"root_id={root_id} branch_id={branch_id}"
                )
            self.conn.execute(
                (
                    "MATCH (s:Scene), (e:Entity) "
                    f"WHERE s.id = '{self._esc(new_scene_id)}' "
                    f"AND e.id = '{self._esc(new_entity_id)}' "
                    f"MERGE (s)-[:SceneEntity {{branch_id: '{self._esc(branch_id)}'}}]->(e);"
                )
            )

        scene_next_rows = self.conn.execute(
            (
                "MATCH (s:Scene)-[:SceneNext]->(n:Scene) "
                f"WHERE s.root_id = '{self._esc(root_id)}' "
                f"AND s.branch_id = '{DEFAULT_BRANCH_ID}' "
                "RETURN s.id, n.id;"
            )
        )
        for row in scene_next_rows:
            new_scene_id = scene_map.get(str(row[0]))
            new_next_id = scene_map.get(str(row[1]))
            if not new_scene_id or not new_next_id:
                raise ValueError(
                    "SceneNext relation missing mapping: "
                    f"root_id={root_id} branch_id={branch_id}"
                )
            self.conn.execute(
                (
                    "MATCH (a:Scene), (b:Scene) "
                    f"WHERE a.id = '{self._esc(new_scene_id)}' "
                    f"AND b.id = '{self._esc(new_next_id)}' "
                    f"MERGE (a)-[:SceneNext {{branch_id: '{self._esc(branch_id)}'}}]->(b);"
                )
            )

        relation_rows = self.conn.execute(
            (
                "MATCH (a:Entity)-[r:EntityRelation]->(b:Entity) "
                f"WHERE a.root_id = '{self._esc(root_id)}' "
                f"AND a.branch_id = '{DEFAULT_BRANCH_ID}' "
                "RETURN a.id, b.id, r.relation_type, r.tension;"
            )
        )
        for row in relation_rows:
            new_from = entity_map.get(str(row[0]))
            new_to = entity_map.get(str(row[1]))
            if not new_from or not new_to:
                raise ValueError(
                    "EntityRelation missing mapping: "
                    f"root_id={root_id} branch_id={branch_id}"
                )
            self.conn.execute(
                (
                    "MATCH (a:Entity), (b:Entity) "
                    f"WHERE a.id = '{self._esc(new_from)}' "
                    f"AND b.id = '{self._esc(new_to)}' "
                    "CREATE (a)-[:EntityRelation {"
                    f"branch_id: '{self._esc(branch_id)}', "
                    f"relation_type: {self._sql_str(row[2])}, "
                    f"tension: {row[3]}"
                    "}]->(b);"
                )
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
                "MATCH (b:Branch) "
                f"WHERE b.root_id = '{self._esc(root_id)}' "
                f"AND b.branch_id = '{self._esc(branch_id)}' "
                "DELETE b;"
            )
        )

    def _purge_branch_data(
        self, *, root_id: str, branch_id: str, branch_root_id: str | None = None
    ) -> None:
        if branch_id == DEFAULT_BRANCH_ID:
            raise ValueError("branch_id must not be default branch")
        self.require_branch(root_id=root_id, branch_id=branch_id)
        if branch_root_id is None:
            branch_root_id = self._resolve_root_node_id(
                root_id=root_id, branch_id=branch_id
            )

        self.conn.execute(
            (
                "MATCH (r:Root)-[rel:RootContainsAct]->(a:Act) "
                f"WHERE r.id = '{self._esc(branch_root_id)}' "
                f"AND r.branch_id = '{self._esc(branch_id)}' "
                "DELETE rel;"
            )
        )
        self.conn.execute(
            (
                "MATCH (a:Act)-[rel:ActContainsScene]->(s:Scene) "
                f"WHERE a.root_id = '{self._esc(root_id)}' "
                f"AND a.branch_id = '{self._esc(branch_id)}' "
                "DELETE rel;"
            )
        )
        self.conn.execute(
            (
                "MATCH (s:Scene)-[rel:SceneEntity]->(e:Entity) "
                f"WHERE s.root_id = '{self._esc(root_id)}' "
                f"AND s.branch_id = '{self._esc(branch_id)}' "
                "DELETE rel;"
            )
        )
        self.conn.execute(
            (
                "MATCH (s:Scene)-[rel:SceneNext]->(n:Scene) "
                f"WHERE s.root_id = '{self._esc(root_id)}' "
                f"AND s.branch_id = '{self._esc(branch_id)}' "
                "DELETE rel;"
            )
        )
        self.conn.execute(
            (
                "MATCH (a:Entity)-[rel:EntityRelation]->(b:Entity) "
                f"WHERE a.root_id = '{self._esc(root_id)}' "
                f"AND a.branch_id = '{self._esc(branch_id)}' "
                "DELETE rel;"
            )
        )
        self.conn.execute(
            (
                "MATCH (s:Scene) "
                f"WHERE s.root_id = '{self._esc(root_id)}' "
                f"AND s.branch_id = '{self._esc(branch_id)}' "
                "DELETE s;"
            )
        )
        self.conn.execute(
            (
                "MATCH (a:Act) "
                f"WHERE a.root_id = '{self._esc(root_id)}' "
                f"AND a.branch_id = '{self._esc(branch_id)}' "
                "DELETE a;"
            )
        )
        self.conn.execute(
            (
                "MATCH (e:Entity) "
                f"WHERE e.root_id = '{self._esc(root_id)}' "
                f"AND e.branch_id = '{self._esc(branch_id)}' "
                "DELETE e;"
            )
        )
        self.conn.execute(
            (
                "MATCH (r:Root) "
                f"WHERE r.id = '{self._esc(branch_root_id)}' "
                f"AND r.branch_id = '{self._esc(branch_id)}' "
                "DELETE r;"
            )
        )

        self._remove_branch(root_id=root_id, branch_id=branch_id)

    def _apply_branch_snapshot_to_main(self, *, root_id: str, branch_id: str) -> None:
        if branch_id == DEFAULT_BRANCH_ID:
            raise ValueError("branch_id must not be default branch")
        self.require_branch(root_id=root_id, branch_id=branch_id)
        branch_root_id = self._resolve_root_node_id(
            root_id=root_id, branch_id=branch_id
        )

        root_rows = self.conn.execute(
            (
                "MATCH (r:Root) "
                f"WHERE r.id = '{self._esc(branch_root_id)}' "
                f"AND r.branch_id = '{self._esc(branch_id)}' "
                "RETURN r.logline, r.theme, r.ending;"
            )
        )
        root_row = next(iter(root_rows), None)
        if not root_row:
            raise KeyError(f"Root not found: root_id={root_id} branch_id={branch_id}")

        update_result = self.conn.execute(
            (
                "MATCH (r:Root) "
                f"WHERE r.id = '{self._esc(root_id)}' "
                f"AND r.branch_id = '{DEFAULT_BRANCH_ID}' "
                f"SET r.logline = {self._sql_str(root_row[0])}, "
                f"r.theme = {self._sql_str(root_row[1])}, "
                f"r.ending = {self._sql_str(root_row[2])} "
                "RETURN r.id LIMIT 1;"
            )
        )
        if not next(iter(update_result), None):
            raise KeyError(
                f"Root not found: root_id={root_id} branch_id={DEFAULT_BRANCH_ID}"
            )

        act_rows = self.conn.execute(
            (
                "MATCH (a:Act) "
                f"WHERE a.root_id = '{self._esc(root_id)}' "
                f"AND a.branch_id = '{self._esc(branch_id)}' "
                "RETURN a.id, a.origin_id, a.act_index, a.disaster;"
            )
        )
        act_map: dict[str, str] = {}
        acts: list[tuple[str, int | None, str | None]] = []
        for row in act_rows:
            branch_act_id = str(row[0])
            origin_id = row[1]
            if not origin_id:
                raise ValueError(
                    "Act origin_id missing: "
                    f"root_id={root_id} branch_id={branch_id} act_id={branch_act_id}"
                )
            origin_id_str = str(origin_id)
            act_map[branch_act_id] = origin_id_str
            acts.append((origin_id_str, row[2], row[3]))

        entity_rows = self.conn.execute(
            (
                "MATCH (e:Entity) "
                f"WHERE e.root_id = '{self._esc(root_id)}' "
                f"AND e.branch_id = '{self._esc(branch_id)}' "
                "RETURN e.id, e.origin_id, e.name, e.entity_type, e.tags_json, "
                "e.arc_status, e.semantic_states_json, e.ambition, e.conflict, "
                "e.epiphany, e.voice_dna;"
            )
        )
        entity_map: dict[str, str] = {}
        entities: list[tuple[str, tuple]] = []
        for row in entity_rows:
            branch_entity_id = str(row[0])
            origin_id = row[1]
            if not origin_id:
                raise ValueError(
                    "Entity origin_id missing: "
                    f"root_id={root_id} branch_id={branch_id} entity_id={branch_entity_id}"
                )
            origin_id_str = str(origin_id)
            entity_map[branch_entity_id] = origin_id_str
            entities.append((origin_id_str, row))

        scene_rows = self.conn.execute(
            (
                "MATCH (s:Scene) "
                f"WHERE s.root_id = '{self._esc(root_id)}' "
                f"AND s.branch_id = '{self._esc(branch_id)}' "
                "RETURN s.id, s.origin_id, s.status, s.pov_character_id, "
                "s.expected_outcome, s.conflict_type, s.actual_outcome, "
                "s.logic_exception, s.logic_exception_reason, s.parent_act_id, "
                "s.dirty, s.summary, s.rendered_content;"
            )
        )
        scene_map: dict[str, str] = {}
        scenes: list[dict[str, Any]] = []
        for row in scene_rows:
            branch_scene_id = str(row[0])
            origin_id = row[1]
            if not origin_id:
                raise ValueError(
                    "Scene origin_id missing: "
                    f"root_id={root_id} branch_id={branch_id} scene_id={branch_scene_id}"
                )
            origin_id_str = str(origin_id)
            scene_map[branch_scene_id] = origin_id_str
            scenes.append(
                {
                    "origin_id": origin_id_str,
                    "status": row[2],
                    "pov_character_id": row[3],
                    "expected_outcome": row[4],
                    "conflict_type": row[5],
                    "actual_outcome": row[6],
                    "logic_exception": row[7],
                    "logic_exception_reason": row[8],
                    "parent_act_id": row[9],
                    "dirty": row[10],
                    "summary": row[11],
                    "rendered_content": row[12],
                }
            )

        self.conn.execute(
            (
                "MATCH (r:Root)-[rel:RootContainsAct]->(a:Act) "
                f"WHERE r.id = '{self._esc(root_id)}' "
                f"AND r.branch_id = '{DEFAULT_BRANCH_ID}' "
                "DELETE rel;"
            )
        )
        self.conn.execute(
            (
                "MATCH (a:Act)-[rel:ActContainsScene]->(s:Scene) "
                f"WHERE a.root_id = '{self._esc(root_id)}' "
                f"AND a.branch_id = '{DEFAULT_BRANCH_ID}' "
                "DELETE rel;"
            )
        )
        self.conn.execute(
            (
                "MATCH (s:Scene)-[rel:SceneEntity]->(e:Entity) "
                f"WHERE s.root_id = '{self._esc(root_id)}' "
                f"AND s.branch_id = '{DEFAULT_BRANCH_ID}' "
                "DELETE rel;"
            )
        )
        self.conn.execute(
            (
                "MATCH (s:Scene)-[rel:SceneNext]->(n:Scene) "
                f"WHERE s.root_id = '{self._esc(root_id)}' "
                f"AND s.branch_id = '{DEFAULT_BRANCH_ID}' "
                "DELETE rel;"
            )
        )
        self.conn.execute(
            (
                "MATCH (a:Entity)-[rel:EntityRelation]->(b:Entity) "
                f"WHERE a.root_id = '{self._esc(root_id)}' "
                f"AND a.branch_id = '{DEFAULT_BRANCH_ID}' "
                "DELETE rel;"
            )
        )
        self.conn.execute(
            (
                "MATCH (s:Scene) "
                f"WHERE s.root_id = '{self._esc(root_id)}' "
                f"AND s.branch_id = '{DEFAULT_BRANCH_ID}' "
                "DELETE s;"
            )
        )
        self.conn.execute(
            (
                "MATCH (a:Act) "
                f"WHERE a.root_id = '{self._esc(root_id)}' "
                f"AND a.branch_id = '{DEFAULT_BRANCH_ID}' "
                "DELETE a;"
            )
        )
        self.conn.execute(
            (
                "MATCH (e:Entity) "
                f"WHERE e.root_id = '{self._esc(root_id)}' "
                f"AND e.branch_id = '{DEFAULT_BRANCH_ID}' "
                "DELETE e;"
            )
        )

        for origin_id, act_index, disaster in acts:
            self.conn.execute(
                (
                    "CREATE (:Act {"
                    f"id: '{self._esc(origin_id)}', "
                    f"origin_id: '{self._esc(origin_id)}', "
                    f"branch_id: '{DEFAULT_BRANCH_ID}', "
                    f"root_id: '{self._esc(root_id)}', "
                    f"act_index: {act_index}, "
                    f"disaster: {self._sql_str(disaster)}"
                    "});"
                )
            )

        for origin_id, row in entities:
            self.conn.execute(
                (
                    "CREATE (:Entity {"
                    f"id: '{self._esc(origin_id)}', "
                    f"origin_id: '{self._esc(origin_id)}', "
                    f"branch_id: '{DEFAULT_BRANCH_ID}', "
                    f"root_id: '{self._esc(root_id)}', "
                    f"name: {self._sql_str(row[2])}, "
                    f"entity_type: {self._sql_str(row[3])}, "
                    f"tags_json: {self._sql_str(row[4])}, "
                    f"arc_status: {self._sql_str(row[5])}, "
                    f"semantic_states_json: {self._sql_str(row[6])}, "
                    f"ambition: {self._sql_str(row[7])}, "
                    f"conflict: {self._sql_str(row[8])}, "
                    f"epiphany: {self._sql_str(row[9])}, "
                    f"voice_dna: {self._sql_str(row[10])}"
                    "});"
                )
            )

        for scene in scenes:
            pov_character_id = scene["pov_character_id"]
            mapped_pov = (
                entity_map.get(str(pov_character_id)) if pov_character_id else None
            )
            if pov_character_id and not mapped_pov:
                raise ValueError(
                    "Scene pov_character_id missing mapping: "
                    f"root_id={root_id} branch_id={branch_id} entity_id={pov_character_id}"
                )
            parent_act_id = scene["parent_act_id"]
            mapped_act = act_map.get(str(parent_act_id)) if parent_act_id else None
            if parent_act_id and not mapped_act:
                raise ValueError(
                    "Scene parent_act_id missing mapping: "
                    f"root_id={root_id} branch_id={branch_id} act_id={parent_act_id}"
                )
            self.conn.execute(
                (
                    "CREATE (:Scene {"
                    f"id: '{self._esc(scene['origin_id'])}', "
                    f"origin_id: '{self._esc(scene['origin_id'])}', "
                    f"branch_id: '{DEFAULT_BRANCH_ID}', "
                    f"status: {self._sql_str(scene['status'])}, "
                    f"pov_character_id: {self._sql_str(mapped_pov)}, "
                    f"expected_outcome: {self._sql_str(scene['expected_outcome'])}, "
                    f"conflict_type: {self._sql_str(scene['conflict_type'])}, "
                    f"actual_outcome: {self._sql_str(scene['actual_outcome'])}, "
                    f"logic_exception: {self._sql_bool(scene['logic_exception'])}, "
                    f"logic_exception_reason: {self._sql_str(scene['logic_exception_reason'])}, "
                    f"parent_act_id: {self._sql_str(mapped_act)}, "
                    f"root_id: '{self._esc(root_id)}', "
                    f"dirty: {self._sql_bool(scene['dirty'])}, "
                    f"summary: {self._sql_str(scene['summary'])}, "
                    f"rendered_content: {self._sql_str(scene['rendered_content'])}"
                    "});"
                )
            )

        for origin_id, _act_index, _disaster in acts:
            self.conn.execute(
                (
                    "MATCH (r:Root), (a:Act) "
                    f"WHERE r.id = '{self._esc(root_id)}' "
                    f"AND r.branch_id = '{DEFAULT_BRANCH_ID}' "
                    f"AND a.id = '{self._esc(origin_id)}' "
                    f"MERGE (r)-[:RootContainsAct {{branch_id: '{DEFAULT_BRANCH_ID}'}}]->(a);"
                )
            )

        act_scene_rows = self.conn.execute(
            (
                "MATCH (a:Act)-[:ActContainsScene]->(s:Scene) "
                f"WHERE a.root_id = '{self._esc(root_id)}' "
                f"AND a.branch_id = '{self._esc(branch_id)}' "
                "RETURN a.id, s.id;"
            )
        )
        for row in act_scene_rows:
            main_act_id = act_map.get(str(row[0]))
            main_scene_id = scene_map.get(str(row[1]))
            if not main_act_id or not main_scene_id:
                raise ValueError(
                    "Act/Scene relation missing mapping: "
                    f"root_id={root_id} branch_id={branch_id}"
                )
            self.conn.execute(
                (
                    "MATCH (a:Act), (s:Scene) "
                    f"WHERE a.id = '{self._esc(main_act_id)}' "
                    f"AND s.id = '{self._esc(main_scene_id)}' "
                    f"MERGE (a)-[:ActContainsScene {{branch_id: '{DEFAULT_BRANCH_ID}'}}]->(s);"
                )
            )

        scene_entity_rows = self.conn.execute(
            (
                "MATCH (s:Scene)-[:SceneEntity]->(e:Entity) "
                f"WHERE s.root_id = '{self._esc(root_id)}' "
                f"AND s.branch_id = '{self._esc(branch_id)}' "
                "RETURN s.id, e.id;"
            )
        )
        for row in scene_entity_rows:
            main_scene_id = scene_map.get(str(row[0]))
            main_entity_id = entity_map.get(str(row[1]))
            if not main_scene_id or not main_entity_id:
                raise ValueError(
                    "Scene/Entity relation missing mapping: "
                    f"root_id={root_id} branch_id={branch_id}"
                )
            self.conn.execute(
                (
                    "MATCH (s:Scene), (e:Entity) "
                    f"WHERE s.id = '{self._esc(main_scene_id)}' "
                    f"AND e.id = '{self._esc(main_entity_id)}' "
                    f"MERGE (s)-[:SceneEntity {{branch_id: '{DEFAULT_BRANCH_ID}'}}]->(e);"
                )
            )

        scene_next_rows = self.conn.execute(
            (
                "MATCH (s:Scene)-[:SceneNext]->(n:Scene) "
                f"WHERE s.root_id = '{self._esc(root_id)}' "
                f"AND s.branch_id = '{self._esc(branch_id)}' "
                "RETURN s.id, n.id;"
            )
        )
        for row in scene_next_rows:
            main_scene_id = scene_map.get(str(row[0]))
            main_next_id = scene_map.get(str(row[1]))
            if not main_scene_id or not main_next_id:
                raise ValueError(
                    "SceneNext relation missing mapping: "
                    f"root_id={root_id} branch_id={branch_id}"
                )
            self.conn.execute(
                (
                    "MATCH (a:Scene), (b:Scene) "
                    f"WHERE a.id = '{self._esc(main_scene_id)}' "
                    f"AND b.id = '{self._esc(main_next_id)}' "
                    f"MERGE (a)-[:SceneNext {{branch_id: '{DEFAULT_BRANCH_ID}'}}]->(b);"
                )
            )

        relation_rows = self.conn.execute(
            (
                "MATCH (a:Entity)-[r:EntityRelation]->(b:Entity) "
                f"WHERE a.root_id = '{self._esc(root_id)}' "
                f"AND a.branch_id = '{self._esc(branch_id)}' "
                "RETURN a.id, b.id, r.relation_type, r.tension;"
            )
        )
        for row in relation_rows:
            main_from = entity_map.get(str(row[0]))
            main_to = entity_map.get(str(row[1]))
            if not main_from or not main_to:
                raise ValueError(
                    "EntityRelation missing mapping: "
                    f"root_id={root_id} branch_id={branch_id}"
                )
            self.conn.execute(
                (
                    "MATCH (a:Entity), (b:Entity) "
                    f"WHERE a.id = '{self._esc(main_from)}' "
                    f"AND b.id = '{self._esc(main_to)}' "
                    "CREATE (a)-[:EntityRelation {"
                    f"branch_id: '{DEFAULT_BRANCH_ID}', "
                    f"relation_type: {self._sql_str(row[2])}, "
                    f"tension: {row[3]}"
                    "}]->(b);"
                )
            )

        self._purge_branch_data(
            root_id=root_id, branch_id=branch_id, branch_root_id=branch_root_id
        )

    def merge_branch(self, *, root_id: str, branch_id: str) -> None:
        self._apply_branch_snapshot_to_main(root_id=root_id, branch_id=branch_id)

    def revert_branch(self, *, root_id: str, branch_id: str) -> None:
        self._purge_branch_data(root_id=root_id, branch_id=branch_id)

    def require_root(self, *, root_id: str, branch_id: str) -> None:
        root_node_id = self._resolve_root_node_id(root_id=root_id, branch_id=branch_id)
        root_rows = self.conn.execute(
            (
                "MATCH (r:Root) "
                f"WHERE r.id = '{self._esc(root_node_id)}' "
                f"AND r.branch_id = '{self._esc(branch_id)}' "
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
        entity_id = str(uuid4())
        arc_status_value = (
            "NULL" if arc_status is None else f"'{self._esc(arc_status)}'"
        )
        self.conn.execute(
            (
                "CREATE (:Entity {"
                f"id: '{entity_id}', "
                f"origin_id: '{entity_id}', "
                f"branch_id: '{self._esc(branch_id)}', "
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
        result = self.conn.execute(
            (
                "MATCH (e:Entity) "
                f"WHERE e.id = '{self._esc(entity_id)}' "
                f"AND e.root_id = '{self._esc(root_id)}' "
                f"AND e.branch_id = '{self._esc(branch_id)}' "
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
        result = self.conn.execute(
            (
                "MATCH (e:Entity) "
                f"WHERE e.id = '{self._esc(entity_id)}' "
                f"AND e.root_id = '{self._esc(root_id)}' "
                f"AND e.branch_id = '{self._esc(branch_id)}' "
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
        payload = self._esc(json.dumps(semantic_states, ensure_ascii=False))
        result = self.conn.execute(
            (
                "MATCH (e:Entity) "
                f"WHERE e.id = '{self._esc(entity_id)}' "
                f"AND e.root_id = '{self._esc(root_id)}' "
                f"AND e.branch_id = '{self._esc(branch_id)}' "
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
        result = self.conn.execute(
            (
                "MATCH (e:Entity) "
                f"WHERE e.root_id = '{self._esc(root_id)}' "
                f"AND e.branch_id = '{self._esc(branch_id)}' "
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
        root_node_id = self._resolve_root_node_id(root_id=root_id, branch_id=branch_id)
        root_rows = self.conn.execute(
            (
                "MATCH (r:Root) "
                f"WHERE r.id = '{self._esc(root_node_id)}' "
                f"AND r.branch_id = '{self._esc(branch_id)}' "
                "RETURN r.id, r.branch_id, r.logline, r.theme, r.ending;"
            )
        )
        root_row = next(iter(root_rows), None)
        if not root_row:
            raise KeyError(f"Root not found: root_id={root_id} branch_id={branch_id}")

        scenes_rows = self.conn.execute(
            (
                "MATCH (s:Scene) "
                f"WHERE s.root_id = '{self._esc(root_id)}' "
                f"AND s.branch_id = '{self._esc(branch_id)}' "
                "RETURN s.id, s.branch_id, s.status, s.pov_character_id, s.expected_outcome, "
                "s.conflict_type, s.actual_outcome, s.logic_exception, s.logic_exception_reason, s.dirty;"
            )
        )
        scenes = [
            {
                "id": row[0],
                "branch_id": row[1],
                "status": row[2],
                "pov_character_id": row[3],
                "expected_outcome": row[4],
                "conflict_type": row[5],
                "actual_outcome": row[6],
                "logic_exception": row[7],
                "logic_exception_reason": row[8],
                "is_dirty": row[9],
            }
            for row in scenes_rows
        ]

        characters_rows = self.conn.execute(
            (
                "MATCH (e:Entity) "
                f"WHERE e.root_id = '{self._esc(root_id)}' "
                f"AND e.branch_id = '{self._esc(branch_id)}' "
                "AND e.entity_type = 'character' "
                "RETURN e.id, e.name;"
            )
        )
        characters = [{"entity_id": row[0], "name": row[1]} for row in characters_rows]

        relations_rows = self.conn.execute(
            (
                "MATCH (a:Entity)-[r:EntityRelation]->(b:Entity) "
                f"WHERE a.root_id = '{self._esc(root_id)}' "
                f"AND a.branch_id = '{self._esc(branch_id)}' "
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
            "logline": root_row[2],
            "theme": root_row[3],
            "ending": root_row[4],
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
        result = self.conn.execute(
            (
                "MATCH (a:Entity), (b:Entity) "
                f"WHERE a.id = '{self._esc(from_entity_id)}' "
                f"AND b.id = '{self._esc(to_entity_id)}' "
                f"AND a.root_id = '{self._esc(root_id)}' "
                f"AND b.root_id = '{self._esc(root_id)}' "
                f"AND a.branch_id = '{self._esc(branch_id)}' "
                f"AND b.branch_id = '{self._esc(branch_id)}' "
                f"MERGE (a)-[r:EntityRelation {{branch_id: '{self._esc(branch_id)}', relation_type: '{self._esc(relation_type)}'}}]->(b) "
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
        if not scene_ids:
            raise ValueError("scene_ids must not be empty")
        scene_id_list = list(scene_ids)
        if len(scene_id_list) != len(set(scene_id_list)):
            raise ValueError("scene_ids must be unique")
        self.require_root(root_id=root_id, branch_id=branch_id)

        existing_rows = self.conn.execute(
            (
                "MATCH (s:Scene) "
                f"WHERE s.root_id = '{self._esc(root_id)}' "
                f"AND s.branch_id = '{self._esc(branch_id)}' "
                "RETURN s.id;"
            )
        )
        existing_ids = [str(row[0]) for row in existing_rows]
        if not existing_ids:
            raise KeyError(f"Scenes not found: root_id={root_id} branch_id={branch_id}")
        existing_set = set(existing_ids)
        requested_set = set(scene_id_list)
        if existing_set != requested_set:
            missing = sorted(existing_set - requested_set)
            extra = sorted(requested_set - existing_set)
            raise ValueError(
                "scene_ids must match existing scenes: "
                f"missing={missing} extra={extra}"
            )

        root_node_id = self._resolve_root_node_id(root_id=root_id, branch_id=branch_id)
        act_rows = self.conn.execute(
            (
                "MATCH (r:Root)-[:RootContainsAct]->(a:Act) "
                f"WHERE r.id = '{self._esc(root_node_id)}' "
                f"AND r.branch_id = '{self._esc(branch_id)}' "
                "RETURN a.id, a.act_index ORDER BY a.act_index;"
            )
        )
        acts = [row[0] for row in act_rows]
        if not acts:
            raise KeyError(f"Act not found: root_id={root_id} branch_id={branch_id}")

        self.conn.execute(
            (
                "MATCH (a:Act)-[rel:ActContainsScene]->(s:Scene) "
                f"WHERE a.root_id = '{self._esc(root_id)}' "
                f"AND a.branch_id = '{self._esc(branch_id)}' "
                "DELETE rel;"
            )
        )

        total_scenes = len(scene_id_list)
        act_count = len(acts)
        for index, scene_id in enumerate(scene_id_list):
            act_idx = (index * act_count) // max(total_scenes, 1)
            act_id = acts[act_idx]
            result = self.conn.execute(
                (
                    "MATCH (s:Scene) "
                    f"WHERE s.id = '{self._esc(scene_id)}' "
                    f"AND s.root_id = '{self._esc(root_id)}' "
                    f"AND s.branch_id = '{self._esc(branch_id)}' "
                    f"SET s.parent_act_id = {self._sql_str(str(act_id))} "
                    "RETURN s.id LIMIT 1;"
                )
            )
            if not next(iter(result), None):
                raise KeyError(
                    "Scene not found: "
                    f"root_id={root_id} branch_id={branch_id} scene_id={scene_id}"
                )
            self.conn.execute(
                (
                    "MATCH (a:Act), (s:Scene) "
                    f"WHERE a.id = '{self._esc(act_id)}' "
                    f"AND a.branch_id = '{self._esc(branch_id)}' "
                    f"AND s.id = '{self._esc(scene_id)}' "
                    f"MERGE (a)-[:ActContainsScene {{branch_id: '{self._esc(branch_id)}'}}]->(s);"
                )
            )

        self.conn.execute(
            (
                "MATCH (s:Scene)-[rel:SceneNext]->(n:Scene) "
                f"WHERE s.root_id = '{self._esc(root_id)}' "
                f"AND s.branch_id = '{self._esc(branch_id)}' "
                "DELETE rel;"
            )
        )
        for current_id, next_id in zip(scene_id_list, scene_id_list[1:]):
            self.conn.execute(
                (
                    "MATCH (a:Scene), (b:Scene) "
                    f"WHERE a.id = '{self._esc(current_id)}' "
                    f"AND b.id = '{self._esc(next_id)}' "
                    f"MERGE (a)-[:SceneNext {{branch_id: '{self._esc(branch_id)}'}}]->(b);"
                )
            )

    def list_structure_tree(self, *, root_id: str, branch_id: str) -> dict[str, Any]:
        self.require_root(root_id=root_id, branch_id=branch_id)
        root_node_id = self._resolve_root_node_id(root_id=root_id, branch_id=branch_id)
        acts_rows = self.conn.execute(
            (
                "MATCH (r:Root)-[:RootContainsAct]->(a:Act) "
                f"WHERE r.id = '{self._esc(root_node_id)}' "
                f"AND r.branch_id = '{self._esc(branch_id)}' "
                "RETURN a.id, a.act_index, a.disaster "
                "ORDER BY a.act_index;"
            )
        )
        acts: list[dict[str, Any]] = []
        for row in acts_rows:
            act_id = row[0]
            scenes_rows = self.conn.execute(
                (
                    "MATCH (a:Act)-[:ActContainsScene]->(s:Scene) "
                    f"WHERE a.id = '{self._esc(act_id)}' "
                    f"AND a.branch_id = '{self._esc(branch_id)}' "
                    "RETURN s.id;"
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
        scene_rows = self.conn.execute(
            (
                "MATCH (s:Scene) "
                f"WHERE s.id = '{self._esc(scene_id)}' "
                f"AND s.branch_id = '{self._esc(branch_id)}' "
                "RETURN s.root_id, s.expected_outcome;"
            )
        )
        scene_row = next(iter(scene_rows), None)
        if not scene_row:
            raise KeyError(f"Scene not found: scene_id={scene_id} branch_id={branch_id}")
        root_id = scene_row[0]
        expected_outcome = scene_row[1]
        if not expected_outcome or not str(expected_outcome).strip():
            raise ValueError(
                f"Scene expected_outcome missing: scene_id={scene_id} branch_id={branch_id}"
            )

        next_rows = self.conn.execute(
            (
                "MATCH (s:Scene)-[:SceneNext]->(n:Scene) "
                f"WHERE s.id = '{self._esc(scene_id)}' "
                f"AND s.branch_id = '{self._esc(branch_id)}' "
                "RETURN n.id LIMIT 1;"
            )
        )
        next_row = next(iter(next_rows), None)

        prev_rows = self.conn.execute(
            (
                "MATCH (p:Scene)-[:SceneNext]->(s:Scene) "
                f"WHERE s.id = '{self._esc(scene_id)}' "
                f"AND s.branch_id = '{self._esc(branch_id)}' "
                "RETURN p.id, p.summary LIMIT 1;"
            )
        )
        prev_row = next(iter(prev_rows), None)
        prev_scene_id = prev_row[0] if prev_row else None
        summary = ""
        if prev_row:
            summary = prev_row[1]
            if summary is None or not str(summary).strip():
                raise ValueError(
                    "Scene summary missing for previous scene: "
                    f"scene_id={scene_id} branch_id={branch_id} prev_scene_id={prev_scene_id}"
                )

        scene_entities_rows = self.conn.execute(
            (
                "MATCH (s:Scene)-[:SceneEntity]->(e:Entity) "
                f"WHERE s.id = '{self._esc(scene_id)}' "
                f"AND s.branch_id = '{self._esc(branch_id)}' "
                "RETURN e.id, e.name, e.entity_type, e.semantic_states_json;"
            )
        )
        scene_entities = []
        semantic_states: dict[str, Any] = {}
        for row in scene_entities_rows:
            raw_states = row[3]
            if raw_states is None:
                raise ValueError(
                    "Scene entity semantic_states missing: "
                    f"scene_id={scene_id} branch_id={branch_id} entity_id={row[0]}"
                )
            states = json.loads(raw_states)
            if not isinstance(states, dict):
                raise ValueError(
                    "Scene entity semantic_states must be a JSON object: "
                    f"scene_id={scene_id} branch_id={branch_id} entity_id={row[0]}"
                )
            scene_entities.append(
                {
                    "entity_id": row[0],
                    "name": row[1],
                    "entity_type": row[2],
                    "semantic_states": states,
                }
            )
            semantic_states[str(row[0])] = states

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
            "next_scene_id": next_row[0] if next_row else None,
        }

    def get_scene_snapshot(self, *, scene_id: str, branch_id: str) -> dict[str, Any]:
        rows = self.conn.execute(
            (
                "MATCH (s:Scene) "
                f"WHERE s.id = '{self._esc(scene_id)}' "
                f"AND s.branch_id = '{self._esc(branch_id)}' "
                "RETURN s.id, s.expected_outcome, s.summary, s.rendered_content, s.dirty;"
            )
        )
        row = next(iter(rows), None)
        if not row:
            raise KeyError(f"Scene not found: scene_id={scene_id} branch_id={branch_id}")
        return {
            "scene_id": row[0],
            "expected_outcome": row[1],
            "summary": row[2],
            "rendered_content": row[3],
            "is_dirty": row[4],
        }

    def is_scene_logic_exception(
        self, *, root_id: str, branch_id: str, scene_id: str
    ) -> bool:
        rows = self.conn.execute(
            (
                "MATCH (s:Scene) "
                f"WHERE s.id = '{self._esc(scene_id)}' "
                f"AND s.root_id = '{self._esc(root_id)}' "
                f"AND s.branch_id = '{self._esc(branch_id)}' "
                "RETURN s.logic_exception;"
            )
        )
        row = next(iter(rows), None)
        if not row:
            raise KeyError(
                f"Scene not found: root_id={root_id} branch_id={branch_id} scene_id={scene_id}"
            )
        return bool(row[0])

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
        result = self.conn.execute(
            (
                "MATCH (s:Scene) "
                f"WHERE s.id = '{self._esc(scene_id)}' "
                f"AND s.root_id = '{self._esc(root_id)}' "
                f"AND s.branch_id = '{self._esc(branch_id)}' "
                "SET s.logic_exception = true, "
                f"s.logic_exception_reason = '{self._esc(reason)}' "
                "RETURN s.id LIMIT 1;"
            )
        )
        if not next(iter(result), None):
            raise KeyError(
                f"Scene not found: root_id={root_id} branch_id={branch_id} scene_id={scene_id}"
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
        result = self.conn.execute(
            (
                "MATCH (s:Scene) "
                f"WHERE s.id = '{self._esc(scene_id)}' "
                f"AND s.branch_id = '{self._esc(branch_id)}' "
                f"SET s.actual_outcome = '{self._esc(actual_outcome)}', "
                f"s.summary = '{self._esc(summary)}', "
                "s.status = 'committed' "
                "RETURN s.id LIMIT 1;"
            )
        )
        if not next(iter(result), None):
            raise KeyError(f"Scene not found: scene_id={scene_id} branch_id={branch_id}")

    def save_scene_render(self, *, scene_id: str, branch_id: str, content: str) -> None:
        if not content.strip():
            raise ValueError("content must not be blank")
        result = self.conn.execute(
            (
                "MATCH (s:Scene) "
                f"WHERE s.id = '{self._esc(scene_id)}' "
                f"AND s.branch_id = '{self._esc(branch_id)}' "
                f"SET s.rendered_content = '{self._esc(content)}' "
                "RETURN s.id LIMIT 1;"
            )
        )
        if not next(iter(result), None):
            raise KeyError(f"Scene not found: scene_id={scene_id} branch_id={branch_id}")

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
        current_rows = self.conn.execute(
            (
                "MATCH (s:Scene) "
                f"WHERE s.id = '{self._esc(scene_id)}' "
                f"AND s.root_id = '{self._esc(root_id)}' "
                f"AND s.branch_id = '{self._esc(branch_id)}' "
                "RETURN s.expected_outcome;"
            )
        )
        current_row = next(iter(current_rows), None)
        if not current_row:
            raise KeyError(
                "Scene not found: "
                f"root_id={root_id} branch_id={branch_id} scene_id={scene_id}"
            )
        base_outcome = current_row[0]
        if base_outcome is None or not str(base_outcome).strip():
            raise ValueError(
                "Scene expected_outcome missing for local fix: "
                f"scene_id={scene_id} branch_id={branch_id}"
            )

        target_ids = self.mark_next_scenes_dirty(
            root_id=root_id,
            branch_id=branch_id,
            scene_id=scene_id,
            limit=limit,
        )

        for target_id in target_ids:
            target_rows = self.conn.execute(
                (
                    "MATCH (s:Scene) "
                    f"WHERE s.id = '{self._esc(target_id)}' "
                    f"AND s.root_id = '{self._esc(root_id)}' "
                    f"AND s.branch_id = '{self._esc(branch_id)}' "
                    "RETURN s.expected_outcome;"
                )
            )
            target_row = next(iter(target_rows), None)
            if not target_row:
                raise KeyError(
                    "Scene not found: "
                    f"root_id={root_id} branch_id={branch_id} scene_id={target_id}"
                )
            target_outcome = target_row[0]
            if target_outcome is None or not str(target_outcome).strip():
                raise ValueError(
                    "Scene expected_outcome missing for local fix: "
                    f"scene_id={target_id} branch_id={branch_id}"
                )
            updated_expected_outcome = (
                f"{target_outcome} [local_fix:{base_outcome}]"
            )
            updated_summary = f"local_fix from {scene_id}: {base_outcome}"
            result = self.conn.execute(
                (
                    "MATCH (s:Scene) "
                    f"WHERE s.id = '{self._esc(target_id)}' "
                    f"AND s.root_id = '{self._esc(root_id)}' "
                    f"AND s.branch_id = '{self._esc(branch_id)}' "
                    f"SET s.expected_outcome = '{self._esc(updated_expected_outcome)}', "
                    f"s.summary = '{self._esc(updated_summary)}' "
                    "RETURN s.id LIMIT 1;"
                )
            )
            if not next(iter(result), None):
                raise KeyError(
                    "Scene not found: "
                    f"root_id={root_id} branch_id={branch_id} scene_id={target_id}"
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
        exists = self.conn.execute(
            (
                "MATCH (s:Scene) "
                f"WHERE s.id = '{self._esc(scene_id)}' "
                f"AND s.root_id = '{self._esc(root_id)}' "
                f"AND s.branch_id = '{self._esc(branch_id)}' "
                "RETURN s.id LIMIT 1;"
            )
        )
        if not next(iter(exists), None):
            raise KeyError(
                "Scene not found: "
                f"root_id={root_id} branch_id={branch_id} scene_id={scene_id}"
            )

        dirty_ids: list[str] = []
        current_id = scene_id
        for _ in range(limit):
            next_rows = self.conn.execute(
                (
                    "MATCH (s:Scene)-[:SceneNext]->(n:Scene) "
                    f"WHERE s.id = '{self._esc(current_id)}' "
                    f"AND s.root_id = '{self._esc(root_id)}' "
                    f"AND s.branch_id = '{self._esc(branch_id)}' "
                    f"AND n.root_id = '{self._esc(root_id)}' "
                    f"AND n.branch_id = '{self._esc(branch_id)}' "
                    "RETURN n.id LIMIT 1;"
                )
            )
            next_row = next(iter(next_rows), None)
            if not next_row:
                break
            next_id = next_row[0]
            dirty_ids.append(next_id)
            current_id = next_id

        for dirty_id in dirty_ids:
            result = self.conn.execute(
                (
                    "MATCH (s:Scene) "
                    f"WHERE s.id = '{self._esc(dirty_id)}' "
                    f"AND s.root_id = '{self._esc(root_id)}' "
                    f"AND s.branch_id = '{self._esc(branch_id)}' "
                    "SET s.dirty = true "
                    "RETURN s.id LIMIT 1;"
                )
            )
            if not next(iter(result), None):
                raise KeyError(
                    "Scene not found: "
                    f"root_id={root_id} branch_id={branch_id} scene_id={dirty_id}"
                )
        return dirty_ids

    def mark_future_scenes_dirty(
        self,
        *,
        root_id: str,
        branch_id: str,
        scene_id: str,
    ) -> list[str]:
        exists = self.conn.execute(
            (
                "MATCH (s:Scene) "
                f"WHERE s.id = '{self._esc(scene_id)}' "
                f"AND s.root_id = '{self._esc(root_id)}' "
                f"AND s.branch_id = '{self._esc(branch_id)}' "
                "RETURN s.id LIMIT 1;"
            )
        )
        if not next(iter(exists), None):
            raise KeyError(
                "Scene not found: "
                f"root_id={root_id} branch_id={branch_id} scene_id={scene_id}"
            )

        dirty_ids: list[str] = []
        current_id = scene_id
        while True:
            next_rows = self.conn.execute(
                (
                    "MATCH (s:Scene)-[:SceneNext]->(n:Scene) "
                    f"WHERE s.id = '{self._esc(current_id)}' "
                    f"AND s.root_id = '{self._esc(root_id)}' "
                    f"AND s.branch_id = '{self._esc(branch_id)}' "
                    f"AND n.root_id = '{self._esc(root_id)}' "
                    f"AND n.branch_id = '{self._esc(branch_id)}' "
                    "RETURN n.id LIMIT 1;"
                )
            )
            next_row = next(iter(next_rows), None)
            if not next_row:
                break
            next_id = next_row[0]
            dirty_ids.append(next_id)
            current_id = next_id

        for dirty_id in dirty_ids:
            result = self.conn.execute(
                (
                    "MATCH (s:Scene) "
                    f"WHERE s.id = '{self._esc(dirty_id)}' "
                    f"AND s.root_id = '{self._esc(root_id)}' "
                    f"AND s.branch_id = '{self._esc(branch_id)}' "
                    "SET s.dirty = true "
                    "RETURN s.id LIMIT 1;"
                )
            )
            if not next(iter(result), None):
                raise KeyError(
                    "Scene not found: "
                    f"root_id={root_id} branch_id={branch_id} scene_id={dirty_id}"
                )
        return dirty_ids

    def mark_scene_dirty(self, *, scene_id: str, branch_id: str) -> None:
        result = self.conn.execute(
            (
                "MATCH (s:Scene) "
                f"WHERE s.id = '{self._esc(scene_id)}' "
                f"AND s.branch_id = '{self._esc(branch_id)}' "
                "SET s.dirty = true "
                "RETURN s.id LIMIT 1;"
            )
        )
        if not next(iter(result), None):
            raise KeyError(f"Scene not found: scene_id={scene_id} branch_id={branch_id}")

    def list_dirty_scenes(self, *, root_id: str, branch_id: str) -> list[str]:
        result = self.conn.execute(
            (
                "MATCH (s:Scene) "
                f"WHERE s.root_id = '{self._esc(root_id)}' "
                f"AND s.branch_id = '{self._esc(branch_id)}' "
                "AND s.dirty = true "
                "RETURN s.id;"
            )
        )
        return [row[0] for row in result]

    def count_scenes(self, root_id: str) -> int:
        result = self.conn.execute(
            (
                "MATCH (s:Scene) "
                f"WHERE s.root_id='{root_id}' "
                f"AND s.branch_id='{DEFAULT_BRANCH_ID}' "
                "RETURN COUNT(*)"
            )
        )
        rows = [row[0] for row in result]
        return rows[0] if rows else 0

    def fetch_scene_ids(self, root_id: str) -> list[str]:
        result = self.conn.execute(
            (
                "MATCH (s:Scene) "
                f"WHERE s.root_id='{root_id}' "
                f"AND s.branch_id='{DEFAULT_BRANCH_ID}' "
                "RETURN s.id"
            )
        )
        return [row[0] for row in result]
