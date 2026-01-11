import tempfile
from uuid import uuid4

import kuzu
import pytest

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


def test_graph_storage_missing_required_fields_fails(tmp_path):
    db_path = tmp_path / "snowflake.db"
    db = kuzu.Database(str(db_path))
    conn = kuzu.Connection(db)
    conn.execute("CREATE NODE TABLE Scene(id STRING, PRIMARY KEY (id));")
    conn.execute("CREATE (:Scene {id: 'scene-1'});")
    conn.close()
    db.close()

    with pytest.raises(ValueError) as excinfo:
        GraphStorage(db_path=db_path)
    assert "required fields" in str(excinfo.value)
    assert "branch_id" in str(excinfo.value)


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

    with pytest.raises(ValueError) as excinfo:
        GraphStorage(db_path=db_path)
    message = str(excinfo.value)
    assert "anchor" in message
    assert "root_id=root-1" in message


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
