import tempfile
from uuid import uuid4

import kuzu
import pytest
import tempfile

from app.models import CharacterSheet, SceneNode, SnowflakeRoot
from app.storage.graph import DEFAULT_BRANCH_ID, DEFAULT_SCENE_STATUS, GraphStorage


def test_graph_storage_persists_snowflake():
    tmpdir = tempfile.mkdtemp()
    storage = GraphStorage(db_path=f"{tmpdir}/snowflake.db")

    root = SnowflakeRoot(
        logline="Test story",
        three_disasters=["D1", "D2", "D3"],
        ending="End",
        theme="Testing",
    )
    characters = [
        CharacterSheet(
            entity_id=uuid4(),
            name="Hero",
            ambition="Save world",
            conflict="Weakness",
            epiphany="Strength",
            voice_dna="Bold",
        )
    ]
    scenes = [
        SceneNode(
            branch_id=DEFAULT_BRANCH_ID,
            expected_outcome="Outcome 1",
            conflict_type="internal",
            actual_outcome="",
            parent_act_id=None,
            is_dirty=False,
            pov_character_id=characters[0].entity_id,
        ),
        SceneNode(
            branch_id=DEFAULT_BRANCH_ID,
            expected_outcome="Outcome 2",
            conflict_type="external",
            actual_outcome="",
            parent_act_id=None,
            is_dirty=False,
            pov_character_id=characters[0].entity_id,
        ),
    ]

    root_id = storage.save_snowflake(root, characters, scenes)

    assert storage.count_scenes(root_id) == 2
    ids = storage.fetch_scene_ids(root_id)
    assert len(ids) == 2
    assert len(set(ids)) == 2
    snapshot = storage.get_root_snapshot(
        root_id=root_id, branch_id=DEFAULT_BRANCH_ID
    )
    for scene in snapshot["scenes"]:
        assert scene["branch_id"] == DEFAULT_BRANCH_ID
        assert scene["actual_outcome"] == ""
        assert scene["is_dirty"] is False


def _seed_storage(storage: GraphStorage) -> tuple[str, CharacterSheet, SceneNode]:
    root = SnowflakeRoot(
        logline="Test story",
        three_disasters=["D1", "D2", "D3"],
        ending="End",
        theme="Testing",
    )
    character = CharacterSheet(
        name="Hero",
        ambition="Save world",
        conflict="Weakness",
        epiphany="Strength",
        voice_dna="Bold",
    )
    scene = SceneNode(
        branch_id=DEFAULT_BRANCH_ID,
        expected_outcome="Outcome 1",
        conflict_type="internal",
        actual_outcome="",
        parent_act_id=None,
        is_dirty=False,
        pov_character_id=character.entity_id,
    )
    root_id = storage.save_snowflake(root, [character], [scene])
    return root_id, character, scene


def _seed_storage_with_scenes(
    storage: GraphStorage, count: int
) -> tuple[str, CharacterSheet, list[SceneNode]]:
    if count < 2:
        raise ValueError("count must be >= 2 for multi-scene seed")
    root = SnowflakeRoot(
        logline="Test story",
        three_disasters=["D1", "D2", "D3"],
        ending="End",
        theme="Testing",
    )
    character = CharacterSheet(
        name="Hero",
        ambition="Save world",
        conflict="Weakness",
        epiphany="Strength",
        voice_dna="Bold",
    )
    scenes: list[SceneNode] = []
    for idx in range(count):
        scenes.append(
            SceneNode(
                branch_id=DEFAULT_BRANCH_ID,
                expected_outcome=f"Outcome {idx + 1}",
                conflict_type="internal" if idx % 2 == 0 else "external",
                actual_outcome="",
                parent_act_id=None,
                is_dirty=False,
                pov_character_id=character.entity_id,
            )
        )
    root_id = storage.save_snowflake(root, [character], scenes)
    return root_id, character, scenes


def _seed_legacy_scene_chain(
    storage: GraphStorage,
    *,
    root_id: str,
    branch_id: str,
    count: int = 3,
    scene_id_prefix: str = "legacy-scene",
    create_root: bool = True,
) -> list[str]:
    if count < 1:
        raise ValueError("count must be >= 1")
    if create_root:
        storage.conn.execute(
            (
                "CREATE (:Root {"
                f"id: '{storage._esc(root_id)}', "
                f"branch_id: '{storage._esc(branch_id)}', "
                "logline: 'legacy logline', "
                "theme: 'legacy theme', "
                "ending: 'legacy ending'"
                "});"
            )
        )
    scene_ids: list[str] = []
    for idx in range(count):
        scene_id = f"{scene_id_prefix}-{idx + 1}"
        origin_id = f"legacy-origin-{idx + 1}"
        storage.conn.execute(
            (
                "CREATE (:Scene {"
                f"id: '{storage._esc(scene_id)}', "
                f"origin_id: '{storage._esc(origin_id)}', "
                f"branch_id: '{storage._esc(branch_id)}', "
                f"root_id: '{storage._esc(root_id)}', "
                f"status: '{DEFAULT_SCENE_STATUS}', "
                "pov_character_id: 'legacy-character', "
                f"expected_outcome: 'Outcome {idx + 1}', "
                "conflict_type: 'internal', "
                "actual_outcome: '', "
                f"summary: 'Summary {idx + 1}', "
                "rendered_content: NULL, "
                "logic_exception: false, "
                "logic_exception_reason: NULL, "
                "parent_act_id: 'legacy-act', "
                "dirty: false"
                "});"
            )
        )
        scene_ids.append(scene_id)
    for idx in range(count - 1):
        from_id = scene_ids[idx]
        to_id = scene_ids[idx + 1]
        storage.conn.execute(
            (
                "MATCH (a:Scene), (b:Scene) "
                f"WHERE a.id = '{storage._esc(from_id)}' "
                f"AND b.id = '{storage._esc(to_id)}' "
                f"CREATE (a)-[:SceneNext {{branch_id: '{storage._esc(branch_id)}'}}]->(b);"
            )
        )
    return scene_ids


def test_graph_storage_missing_required_fields_fails(tmp_path):
    db_path = tmp_path / "snowflake.db"
    db = kuzu.Database(str(db_path))
    conn = kuzu.Connection(db)
    conn.execute("CREATE NODE TABLE Scene(id STRING, PRIMARY KEY (id));")
    conn.execute("CREATE (:Scene {id: 'scene-1'});")
    conn.close()
    db.close()

    storage = GraphStorage(db_path=db_path)
    try:
        with pytest.raises(ValueError) as excinfo:
            storage.run_backfill_migrations()
        assert "required fields" in str(excinfo.value)
        assert "branch_id" in str(excinfo.value)
    finally:
        storage.close()


def test_graph_storage_missing_anchor_fails(tmp_path):
    db_path = tmp_path / "snowflake.db"
    db = kuzu.Database(str(db_path))
    conn = kuzu.Connection(db)
    conn.execute(
        "CREATE NODE TABLE Scene("
        "id STRING, "
        "branch_id STRING, "
        "root_id STRING, "
        "status STRING, "
        "dirty BOOLEAN, "
        "PRIMARY KEY (id)"
        ");"
    )
    conn.execute(
        (
            "CREATE (:Scene {"
            "id: 'scene-1', "
            f"branch_id: '{DEFAULT_BRANCH_ID}', "
            "root_id: 'root-1', "
            f"status: '{DEFAULT_SCENE_STATUS}', "
            "dirty: false"
            "});"
        )
    )
    conn.close()
    db.close()

    storage = GraphStorage(db_path=db_path)
    try:
        with pytest.raises(ValueError) as excinfo:
            storage.run_backfill_migrations()
        message = str(excinfo.value)
        assert "anchor" in message
        assert "root_id=root-1" in message
    finally:
        storage.close()


def test_apply_semantic_states_patch():
    tmpdir = tempfile.mkdtemp()
    storage = GraphStorage(db_path=f"{tmpdir}/snowflake.db")
    try:
        root_id, character, _scene = _seed_storage(storage)
        updated = storage.apply_semantic_states_patch(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            entity_id=str(character.entity_id),
            patch={"hp": "90%"},
        )
        assert updated["hp"] == "90%"
        current = storage.get_entity_semantic_states(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            entity_id=str(character.entity_id),
        )
        assert current["hp"] == "90%"
    finally:
        storage.close()


def test_mark_scene_dirty_and_list():
    tmpdir = tempfile.mkdtemp()
    storage = GraphStorage(db_path=f"{tmpdir}/snowflake.db")
    try:
        root_id, _character, scene = _seed_storage(storage)
        storage.mark_scene_dirty(scene_id=str(scene.id), branch_id=DEFAULT_BRANCH_ID)
        dirty = storage.list_dirty_scenes(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        assert str(scene.id) in dirty
    finally:
        storage.close()


def test_mark_next_scenes_dirty_supports_limit_three(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id, _character, scenes = _seed_storage_with_scenes(storage, count=4)
        dirty_ids = storage.mark_next_scenes_dirty(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            scene_id=str(scenes[0].id),
            limit=3,
        )
        expected = {str(scene.id) for scene in scenes[1:4]}
        assert set(dirty_ids) == expected
        dirty_list = storage.list_dirty_scenes(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        assert expected.issubset(set(dirty_list))
    finally:
        storage.close()


def test_mark_scene_logic_exception_persists_reason():
    tmpdir = tempfile.mkdtemp()
    storage = GraphStorage(db_path=f"{tmpdir}/snowflake.db")
    try:
        root_id, _character, scene = _seed_storage(storage)
        storage.mark_scene_logic_exception(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            scene_id=str(scene.id),
            reason="force execute",
        )
        snapshot = storage.get_root_snapshot(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        snapshot_scene = next(
            item for item in snapshot["scenes"] if item["id"] == str(scene.id)
        )
        assert snapshot_scene["logic_exception"] is True
        assert snapshot_scene["logic_exception_reason"] == "force execute"
    finally:
        storage.close()


def test_get_entity_semantic_states_roundtrip():
    tmpdir = tempfile.mkdtemp()
    storage = GraphStorage(db_path=f"{tmpdir}/snowflake.db")
    try:
        root_id, character, _scene = _seed_storage(storage)
        storage.set_entity_semantic_states(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            entity_id=str(character.entity_id),
            semantic_states={"mood": "steady"},
        )
        current = storage.get_entity_semantic_states(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            entity_id=str(character.entity_id),
        )
        assert current == {"mood": "steady"}
    finally:
        storage.close()


def test_branch_metadata_roundtrip(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id, _character, _scene = _seed_storage(storage)
        branches = storage.list_branches(root_id=root_id)
        assert branches == [DEFAULT_BRANCH_ID]

        storage.create_branch(root_id=root_id, branch_id="dev")
        branches = storage.list_branches(root_id=root_id)
        assert set(branches) == {DEFAULT_BRANCH_ID, "dev"}
        storage.require_branch(root_id=root_id, branch_id="dev")
        with pytest.raises(KeyError):
            storage.require_branch(root_id=root_id, branch_id="ghost")
    finally:
        storage.close()


def test_create_branch_duplicate_fails(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id, _character, _scene = _seed_storage(storage)
        storage.create_branch(root_id=root_id, branch_id="dev")
        with pytest.raises(ValueError):
            storage.create_branch(root_id=root_id, branch_id="dev")
    finally:
        storage.close()


def test_branch_isolation_after_create(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id, _character, _scene = _seed_storage(storage)
        storage.create_branch(root_id=root_id, branch_id="dev")
        main_snapshot = storage.get_root_snapshot(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        branch_snapshot = storage.get_root_snapshot(root_id=root_id, branch_id="dev")
        assert len(branch_snapshot["scenes"]) == len(main_snapshot["scenes"])
        assert len(branch_snapshot["characters"]) == len(main_snapshot["characters"])

        branch_scene_id = branch_snapshot["scenes"][0]["id"]
        storage.complete_scene(
            scene_id=branch_scene_id,
            branch_id="dev",
            actual_outcome="branch outcome",
            summary="branch summary",
        )
        updated_branch = storage.get_root_snapshot(root_id=root_id, branch_id="dev")
        updated_main = storage.get_root_snapshot(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        branch_scene = next(
            item for item in updated_branch["scenes"] if item["id"] == branch_scene_id
        )
        main_scene = updated_main["scenes"][0]
        assert branch_scene["actual_outcome"] == "branch outcome"
        assert main_scene["actual_outcome"] == ""
    finally:
        storage.close()


def test_merge_branch_applies_branch_snapshot(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id, _character, scene = _seed_storage(storage)
        storage.create_branch(root_id=root_id, branch_id="dev")
        branch_snapshot = storage.get_root_snapshot(root_id=root_id, branch_id="dev")
        branch_scene_id = branch_snapshot["scenes"][0]["id"]
        storage.complete_scene(
            scene_id=branch_scene_id,
            branch_id="dev",
            actual_outcome="merged outcome",
            summary="merged summary",
        )

        storage.merge_branch(root_id=root_id, branch_id="dev")
        assert storage.list_branches(root_id=root_id) == [DEFAULT_BRANCH_ID]
        snapshot = storage.get_root_snapshot(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        merged_scene = next(
            item for item in snapshot["scenes"] if item["id"] == str(scene.id)
        )
        assert merged_scene["actual_outcome"] == "merged outcome"
    finally:
        storage.close()


def test_revert_branch_discards_branch_and_preserves_main(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id, _character, scene = _seed_storage(storage)
        storage.create_branch(root_id=root_id, branch_id="dev")
        branch_snapshot = storage.get_root_snapshot(root_id=root_id, branch_id="dev")
        branch_scene_id = branch_snapshot["scenes"][0]["id"]
        storage.complete_scene(
            scene_id=branch_scene_id,
            branch_id="dev",
            actual_outcome="branch outcome",
            summary="branch summary",
        )
        storage.complete_scene(
            scene_id=str(scene.id),
            branch_id=DEFAULT_BRANCH_ID,
            actual_outcome="main outcome",
            summary="main summary",
        )

        storage.revert_branch(root_id=root_id, branch_id="dev")
        assert storage.list_branches(root_id=root_id) == [DEFAULT_BRANCH_ID]
        after_revert = storage.get_root_snapshot(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        preserved_scene = next(
            item for item in after_revert["scenes"] if item["id"] == str(scene.id)
        )
        assert preserved_scene["actual_outcome"] == "main outcome"
    finally:
        storage.close()


def test_branch_name_validation(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id, _character, _scene = _seed_storage(storage)
        with pytest.raises(ValueError):
            storage.create_branch(root_id=root_id, branch_id="bad name")
        with pytest.raises(ValueError):
            storage.create_branch(root_id=root_id, branch_id="中文")
        with pytest.raises(ValueError):
            storage.create_branch(root_id=root_id, branch_id="a" * 65)
        storage.create_branch(root_id=root_id, branch_id="branch_A-1")
    finally:
        storage.close()


def test_commit_scene_rejects_no_change(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id, _character, scene = _seed_storage(storage)
        with pytest.raises(ValueError) as excinfo:
            storage.commit_scene(
                root_id=root_id,
                branch_id=DEFAULT_BRANCH_ID,
                scene_origin_id=str(scene.id),
                content={"actual_outcome": ""},
                message="noop",
            )
        assert "NO_CHANGES" in str(excinfo.value)
    finally:
        storage.close()


def test_commit_scene_concurrent_modification(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id, _character, scene = _seed_storage(storage)
        head = storage.get_branch_head(root_id=root_id, branch_id=DEFAULT_BRANCH_ID)
        storage.commit_scene(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            scene_origin_id=str(scene.id),
            content={
                "actual_outcome": "A",
                "summary": "summary A",
                "status": "committed",
            },
            message="commit A",
            expected_head_version=head["version"],
        )
        with pytest.raises(ValueError) as excinfo:
            storage.commit_scene(
                root_id=root_id,
                branch_id=DEFAULT_BRANCH_ID,
                scene_origin_id=str(scene.id),
                content={
                    "actual_outcome": "B",
                    "summary": "summary B",
                    "status": "committed",
                },
                message="commit B",
                expected_head_version=head["version"],
            )
        assert "CONCURRENT_MODIFICATION" in str(excinfo.value)
    finally:
        storage.close()


def test_get_scene_at_commit_and_diff(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id, _character, scene = _seed_storage(storage)
        initial_head = storage.get_branch_head(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        storage.commit_scene(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            scene_origin_id=str(scene.id),
            content={
                "actual_outcome": "Outcome X",
                "summary": "Summary X",
                "status": "committed",
            },
            message="update scene",
        )
        updated_head = storage.get_branch_head(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        before = storage.get_scene_at_commit(
            scene_origin_id=str(scene.id),
            commit_id=initial_head["head_commit_id"],
        )
        after = storage.get_scene_at_commit(
            scene_origin_id=str(scene.id),
            commit_id=updated_head["head_commit_id"],
        )
        assert before["actual_outcome"] == ""
        assert after["actual_outcome"] == "Outcome X"
        diff = storage.diff_scene_versions(
            scene_origin_id=str(scene.id),
            from_commit_id=initial_head["head_commit_id"],
            to_commit_id=updated_head["head_commit_id"],
        )
        assert diff["actual_outcome"] == {"from": "", "to": "Outcome X"}
        assert diff["summary"]["to"] == "Summary X"
    finally:
        storage.close()


def test_fork_from_historic_commit_and_delete_branch(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id, _character, scene = _seed_storage(storage)
        initial_head = storage.get_branch_head(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        storage.commit_scene(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            scene_origin_id=str(scene.id),
            content={
                "actual_outcome": "main update",
                "summary": "main summary",
                "status": "committed",
            },
            message="main update",
        )
        storage.fork_from_commit(
            root_id=root_id,
            source_commit_id=initial_head["head_commit_id"],
            new_branch_id="branch_A",
            parent_branch_id=DEFAULT_BRANCH_ID,
        )
        branch_snapshot = storage.get_root_snapshot(
            root_id=root_id, branch_id="branch_A"
        )
        branch_scene = next(
            item for item in branch_snapshot["scenes"] if item["id"] == str(scene.id)
        )
        assert branch_scene["actual_outcome"] == ""
        storage.revert_branch(root_id=root_id, branch_id="branch_A")
        main_snapshot = storage.get_root_snapshot(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        main_scene = next(
            item for item in main_snapshot["scenes"] if item["id"] == str(scene.id)
        )
        assert main_scene["actual_outcome"] == "main update"
    finally:
        storage.close()


def test_fork_from_scene_at_fifth_scene(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id, _character, scenes = _seed_storage_with_scenes(storage, count=5)
        storage.fork_from_scene(
            root_id=root_id,
            source_branch_id=DEFAULT_BRANCH_ID,
            scene_origin_id=str(scenes[4].id),
            new_branch_id="branch_A",
        )
        main_head = storage.get_branch_head(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        branch_head = storage.get_branch_head(root_id=root_id, branch_id="branch_A")
        assert branch_head["head_commit_id"] == main_head["head_commit_id"]
    finally:
        storage.close()


def test_fork_from_scene_allows_head_without_scene_update(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id, _character, scenes = _seed_storage_with_scenes(storage, count=2)
        commit = storage.commit_scene(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            scene_origin_id=str(scenes[1].id),
            content={
                "actual_outcome": "Scene 2 update",
                "summary": "Scene 2 summary",
                "status": "committed",
            },
            message="scene 2 update",
        )
        main_head = storage.get_branch_head(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        assert main_head["head_commit_id"] == commit["commit_id"]
        detail = storage.get_commit_detail(
            root_id=root_id, commit_id=main_head["head_commit_id"]
        )
        scene_origin_ids = {
            version["scene_origin_id"] for version in detail["scene_versions"]
        }
        assert str(scenes[0].id) not in scene_origin_ids

        storage.fork_from_scene(
            root_id=root_id,
            source_branch_id=DEFAULT_BRANCH_ID,
            scene_origin_id=str(scenes[0].id),
            new_branch_id="branch_A",
        )
        branch_head = storage.get_branch_head(root_id=root_id, branch_id="branch_A")
        assert branch_head["head_commit_id"] == main_head["head_commit_id"]
    finally:
        storage.close()


def test_scene_version_count_under_threshold(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id, _character, scenes = _seed_storage_with_scenes(storage, count=100)
        for idx in range(10):
            branch_id = f"branch_{idx}"
            storage.create_branch(root_id=root_id, branch_id=branch_id)
            storage.commit_scene(
                root_id=root_id,
                branch_id=branch_id,
                scene_origin_id=str(scenes[idx].id),
                content={
                    "actual_outcome": f"branch update {idx}",
                    "summary": f"summary {idx}",
                    "status": "committed",
                },
                message=f"update {idx}",
            )
        assert storage.count_scene_versions(root_id) <= 110
    finally:
        storage.close()


def test_create_scene_origin_requires_parent_act(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id, character, _scene = _seed_storage(storage)
        tree = storage.list_structure_tree(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        act_id = tree["acts"][0]["act_id"]
        content = {
            "pov_character_id": str(character.entity_id),
            "expected_outcome": "Outcome 2",
            "conflict_type": "internal",
            "actual_outcome": "",
            "dirty": False,
        }
        result = storage.create_scene_origin(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            title="New scene",
            parent_act_id=act_id,
            content=content,
        )
        assert result["scene_origin_id"]

        with pytest.raises(ValueError) as excinfo:
            storage.create_scene_origin(
                root_id=root_id,
                branch_id=DEFAULT_BRANCH_ID,
                title="Bad scene",
                parent_act_id="missing",
                content=content,
            )
        assert "INVALID_SCENE_PARENT_ACT" in str(excinfo.value)
    finally:
        storage.close()


def test_create_scene_origin_allows_non_default_branch(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id, character, _scene = _seed_storage(storage)
        storage.create_branch(root_id=root_id, branch_id="dev")
        tree = storage.list_structure_tree(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        act_id = tree["acts"][0]["act_id"]
        content = {
            "pov_character_id": str(character.entity_id),
            "expected_outcome": "Outcome branch",
            "conflict_type": "internal",
            "actual_outcome": "",
            "dirty": False,
        }
        result = storage.create_scene_origin(
            root_id=root_id,
            branch_id="dev",
            title="Branch scene",
            parent_act_id=act_id,
            content=content,
        )
        assert result["commit_id"]
        assert result["scene_version_id"]
        head = storage.get_branch_head(root_id=root_id, branch_id="dev")
        assert head["head_commit_id"] == result["commit_id"]
        versions = storage.list_scene_versions(
            scene_origin_id=result["scene_origin_id"]
        )
        assert versions[0]["commit_id"] == result["commit_id"]
    finally:
        storage.close()


def test_delete_scene_origin_archives_scene(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id, _character, scene = _seed_storage(storage)
        result = storage.delete_scene_origin(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            scene_origin_id=str(scene.id),
            message="archive scene",
        )
        head = storage.get_branch_head(root_id=root_id, branch_id=DEFAULT_BRANCH_ID)
        assert result["commit_id"] == head["head_commit_id"]
        version = storage.get_scene_at_commit(
            scene_origin_id=str(scene.id), commit_id=head["head_commit_id"]
        )
        assert version["status"] == "archived"
    finally:
        storage.close()


def test_fork_from_scene_requires_commit_contains_version(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id, _character, scenes = _seed_storage_with_scenes(storage, count=2)
        commit = storage.commit_scene(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            scene_origin_id=str(scenes[1].id),
            content={
                "actual_outcome": "Scene 2 update",
                "summary": "Scene 2 summary",
                "status": "committed",
            },
            message="scene 2 update",
        )
        with pytest.raises(KeyError) as excinfo:
            storage.fork_from_scene(
                root_id=root_id,
                source_branch_id=DEFAULT_BRANCH_ID,
                scene_origin_id=str(scenes[0].id),
                new_branch_id="branch_A",
                commit_id=commit["commit_id"],
            )
        assert "SCENE_NOT_FOUND" in str(excinfo.value)
    finally:
        storage.close()


def test_reset_branch_head_updates_head(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id, _character, scene = _seed_storage(storage)
        initial_head = storage.get_branch_head(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        commit_a = storage.commit_scene(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            scene_origin_id=str(scene.id),
            content={
                "actual_outcome": "A",
                "summary": "summary A",
                "status": "committed",
            },
            message="commit A",
        )
        commit_b = storage.commit_scene(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            scene_origin_id=str(scene.id),
            content={
                "actual_outcome": "B",
                "summary": "summary B",
                "status": "committed",
            },
            message="commit B",
        )
        storage.reset_branch_head(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            commit_id=commit_a["commit_id"],
        )
        head = storage.get_branch_head(root_id=root_id, branch_id=DEFAULT_BRANCH_ID)
        assert head["head_commit_id"] == commit_a["commit_id"]
        assert head["head_commit_id"] != commit_b["commit_id"]
        assert head["fork_point_commit_id"] == initial_head["fork_point_commit_id"]
    finally:
        storage.close()


def test_reset_branch_head_rejects_non_ancestor(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id, _character, scene = _seed_storage(storage)
        storage.create_branch(root_id=root_id, branch_id="dev")
        commit = storage.commit_scene(
            root_id=root_id,
            branch_id="dev",
            scene_origin_id=str(scene.id),
            content={
                "actual_outcome": "branch update",
                "summary": "branch summary",
                "status": "committed",
            },
            message="branch commit",
        )
        with pytest.raises(ValueError) as excinfo:
            storage.reset_branch_head(
                root_id=root_id,
                branch_id=DEFAULT_BRANCH_ID,
                commit_id=commit["commit_id"],
            )
        assert "INVALID_RESET" in str(excinfo.value)
    finally:
        storage.close()


def test_reset_branch_head_rejects_before_fork_point(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id, _character, scene = _seed_storage(storage)
        commit_a = storage.commit_scene(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            scene_origin_id=str(scene.id),
            content={
                "actual_outcome": "A",
                "summary": "summary A",
                "status": "committed",
            },
            message="commit A",
        )
        commit_b = storage.commit_scene(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            scene_origin_id=str(scene.id),
            content={
                "actual_outcome": "B",
                "summary": "summary B",
                "status": "committed",
            },
            message="commit B",
        )
        storage.create_branch(root_id=root_id, branch_id="dev")
        storage.commit_scene(
            root_id=root_id,
            branch_id="dev",
            scene_origin_id=str(scene.id),
            content={
                "actual_outcome": "C",
                "summary": "summary C",
                "status": "committed",
            },
            message="commit C",
        )
        with pytest.raises(ValueError) as excinfo:
            storage.reset_branch_head(
                root_id=root_id,
                branch_id="dev",
                commit_id=commit_a["commit_id"],
            )
        assert "INVALID_RESET" in str(excinfo.value)
    finally:
        storage.close()


def test_get_branch_history_limit(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id, _character, scene = _seed_storage(storage)
        commit_a = storage.commit_scene(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            scene_origin_id=str(scene.id),
            content={
                "actual_outcome": "A",
                "summary": "summary A",
                "status": "committed",
            },
            message="commit A",
        )
        commit_b = storage.commit_scene(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            scene_origin_id=str(scene.id),
            content={
                "actual_outcome": "B",
                "summary": "summary B",
                "status": "committed",
            },
            message="commit B",
        )
        history = storage.get_branch_history(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID, limit=2
        )
        assert [item["id"] for item in history] == [
            commit_b["commit_id"],
            commit_a["commit_id"],
        ]
    finally:
        storage.close()


def test_get_commit_detail_returns_scene_versions(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id, _character, scene = _seed_storage(storage)
        commit = storage.commit_scene(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            scene_origin_id=str(scene.id),
            content={
                "actual_outcome": "A",
                "summary": "summary A",
                "status": "committed",
            },
            message="commit A",
        )
        detail = storage.get_commit_detail(
            root_id=root_id, commit_id=commit["commit_id"]
        )
        assert detail["commit"]["id"] == commit["commit_id"]
        assert len(detail["scene_versions"]) == 1
        assert detail["scene_versions"][0]["scene_origin_id"] == str(scene.id)
        assert detail["scene_versions"][0]["commit_id"] == commit["commit_id"]
    finally:
        storage.close()


def test_list_scene_versions_orders_desc(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id, _character, scene = _seed_storage(storage)
        commit_a = storage.commit_scene(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            scene_origin_id=str(scene.id),
            content={
                "actual_outcome": "A",
                "summary": "summary A",
                "status": "committed",
            },
            message="commit A",
        )
        commit_b = storage.commit_scene(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            scene_origin_id=str(scene.id),
            content={
                "actual_outcome": "B",
                "summary": "summary B",
                "status": "committed",
            },
            message="commit B",
        )
        versions = storage.list_scene_versions(scene_origin_id=str(scene.id))
        assert versions[0]["commit_id"] == commit_b["commit_id"]
        assert versions[1]["commit_id"] == commit_a["commit_id"]
    finally:
        storage.close()


def test_gc_orphan_commits_removes_unreachable(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id, _character, scene = _seed_storage(storage)
        initial_head = storage.get_branch_head(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        commit_a = storage.commit_scene(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            scene_origin_id=str(scene.id),
            content={
                "actual_outcome": "A",
                "summary": "summary A",
                "status": "committed",
            },
            message="commit A",
        )
        commit_b = storage.commit_scene(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            scene_origin_id=str(scene.id),
            content={
                "actual_outcome": "B",
                "summary": "summary B",
                "status": "committed",
            },
            message="commit B",
        )
        storage.reset_branch_head(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            commit_id=initial_head["head_commit_id"],
        )
        gc_result = storage.gc_orphan_commits(retention_days=0)
        deleted_commits = set(gc_result["deleted_commit_ids"])
        assert commit_a["commit_id"] in deleted_commits
        assert commit_b["commit_id"] in deleted_commits
        deleted_versions = set(gc_result["deleted_scene_version_ids"])
        assert set(commit_a["scene_version_ids"]).issubset(deleted_versions)
        assert set(commit_b["scene_version_ids"]).issubset(deleted_versions)
        versions = storage.list_scene_versions(scene_origin_id=str(scene.id))
        assert len(versions) == 1
        with pytest.raises(KeyError):
            storage.get_commit_detail(root_id=root_id, commit_id=commit_a["commit_id"])
    finally:
        storage.close()


def test_delete_branch_removes_metadata_only(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id, _character, scene = _seed_storage(storage)
        storage.create_branch(root_id=root_id, branch_id="dev")
        commit = storage.commit_scene(
            root_id=root_id,
            branch_id="dev",
            scene_origin_id=str(scene.id),
            content={
                "actual_outcome": "branch update",
                "summary": "branch summary",
                "status": "committed",
            },
            message="branch commit",
        )
        storage.delete_branch(root_id=root_id, branch_id="dev")
        assert storage.list_branches(root_id=root_id) == [DEFAULT_BRANCH_ID]
        detail = storage.get_commit_detail(
            root_id=root_id, commit_id=commit["commit_id"]
        )
        assert detail["commit"]["id"] == commit["commit_id"]
    finally:
        storage.close()


def test_delete_empty_branch(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id, _character, _scene = _seed_storage(storage)
        storage.create_branch(root_id=root_id, branch_id="empty")
        storage.delete_branch(root_id=root_id, branch_id="empty")
        assert storage.list_branches(root_id=root_id) == [DEFAULT_BRANCH_ID]
    finally:
        storage.close()


def test_fork_from_orphan_commit(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id, _character, scene = _seed_storage(storage)
        initial_head = storage.get_branch_head(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        storage.commit_scene(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            scene_origin_id=str(scene.id),
            content={
                "actual_outcome": "A",
                "summary": "summary A",
                "status": "committed",
            },
            message="commit A",
        )
        commit_b = storage.commit_scene(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            scene_origin_id=str(scene.id),
            content={
                "actual_outcome": "B",
                "summary": "summary B",
                "status": "committed",
            },
            message="commit B",
        )
        storage.reset_branch_head(
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            commit_id=initial_head["head_commit_id"],
        )
        storage.fork_from_commit(
            root_id=root_id,
            source_commit_id=commit_b["commit_id"],
            new_branch_id="orphan",
            parent_branch_id=DEFAULT_BRANCH_ID,
        )
        orphan_head = storage.get_branch_head(root_id=root_id, branch_id="orphan")
        assert orphan_head["head_commit_id"] == commit_b["commit_id"]
    finally:
        storage.close()


def test_deep_nested_branch_forks(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id, _character, scene = _seed_storage(storage)
        storage.create_branch(root_id=root_id, branch_id="branch_a")
        storage.commit_scene(
            root_id=root_id,
            branch_id="branch_a",
            scene_origin_id=str(scene.id),
            content={
                "actual_outcome": "branch A update",
                "summary": "branch A summary",
                "status": "committed",
            },
            message="branch A update",
        )
        head_a = storage.get_branch_head(root_id=root_id, branch_id="branch_a")
        storage.fork_from_commit(
            root_id=root_id,
            source_commit_id=head_a["head_commit_id"],
            new_branch_id="branch_b",
            parent_branch_id="branch_a",
        )
        storage.commit_scene(
            root_id=root_id,
            branch_id="branch_b",
            scene_origin_id=str(scene.id),
            content={
                "actual_outcome": "branch B update",
                "summary": "branch B summary",
                "status": "committed",
            },
            message="branch B update",
        )
        head_b = storage.get_branch_head(root_id=root_id, branch_id="branch_b")
        storage.fork_from_commit(
            root_id=root_id,
            source_commit_id=head_b["head_commit_id"],
            new_branch_id="branch_c",
            parent_branch_id="branch_b",
        )
        head_c = storage.get_branch_head(root_id=root_id, branch_id="branch_c")
        assert head_c["head_commit_id"] == head_b["head_commit_id"]
        assert set(storage.list_branches(root_id=root_id)) == {
            DEFAULT_BRANCH_ID,
            "branch_a",
            "branch_b",
            "branch_c",
        }
    finally:
        storage.close()


def test_scene_version_migration_non_default_branch_independent_commit(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id = str(uuid4())
        _seed_legacy_scene_chain(
            storage,
            root_id=root_id,
            branch_id=DEFAULT_BRANCH_ID,
            count=2,
            scene_id_prefix="legacy-main",
            create_root=True,
        )
        _seed_legacy_scene_chain(
            storage,
            root_id=root_id,
            branch_id="dev",
            count=2,
            scene_id_prefix="legacy-dev",
            create_root=False,
        )
        storage.migrate_scene_versions()
        head = storage.get_branch_head(root_id=root_id, branch_id="dev")
        detail = storage.get_commit_detail(
            root_id=root_id, commit_id=head["head_commit_id"]
        )
        assert detail["commit"]["parent_id"] is None
        assert head["fork_point_commit_id"] is None
    finally:
        storage.close()


def test_scene_version_migration_idempotent(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id = str(uuid4())
        _seed_legacy_scene_chain(
            storage, root_id=root_id, branch_id=DEFAULT_BRANCH_ID, count=3
        )
        storage.migrate_scene_versions()
        origin_count = storage.count_scenes(root_id)
        version_count = storage.count_scene_versions(root_id)
        head = storage.get_branch_head(root_id=root_id, branch_id=DEFAULT_BRANCH_ID)
        storage.migrate_scene_versions()
        assert storage.count_scenes(root_id) == origin_count
        assert storage.count_scene_versions(root_id) == version_count
        head_after = storage.get_branch_head(
            root_id=root_id, branch_id=DEFAULT_BRANCH_ID
        )
        assert head_after["head_commit_id"] == head["head_commit_id"]
    finally:
        storage.close()


def test_scene_version_migration_rollback(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id = str(uuid4())
        scene_ids = _seed_legacy_scene_chain(
            storage, root_id=root_id, branch_id=DEFAULT_BRANCH_ID, count=2
        )
        storage.migrate_scene_versions()
        storage.rollback_scene_version_migration(root_id=root_id)
        assert storage.count_scenes(root_id) == 0
        assert storage.count_scene_versions(root_id) == 0
        with pytest.raises(KeyError):
            storage.get_branch_head(root_id=root_id, branch_id=DEFAULT_BRANCH_ID)
        rows = list(
            storage.conn.execute(
                (
                    "MATCH (s:Scene) "
                    f"WHERE s.root_id = '{storage._esc(root_id)}' "
                    "RETURN COUNT(*)"
                )
            )
        )
        assert rows[0][0] == len(scene_ids)
    finally:
        storage.close()


def test_scene_version_migration_rejects_multiple_starts(tmp_path):
    storage = GraphStorage(db_path=tmp_path / "snowflake.db")
    try:
        root_id = str(uuid4())
        _seed_legacy_scene_chain(
            storage, root_id=root_id, branch_id=DEFAULT_BRANCH_ID, count=2
        )
        storage.conn.execute("MATCH ()-[r:SceneNext]->() DELETE r;")
        with pytest.raises(ValueError):
            storage.migrate_scene_versions()
    finally:
        storage.close()
