"""Minimal end-to-end smoke test against the Snowflake backend using a cyberpunk story idea.

Run with: python scripts/cyberpunk_integration_test.py
Set BACKEND_BASE_URL to override the default http://127.0.0.1:8000.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests

BASE_URL = os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT = float(os.getenv("BACKEND_TIMEOUT", "30"))
PACE_SECONDS = float(os.getenv("BACKEND_PACE_SECONDS", "0"))
FULL = os.getenv("INTEGRATION_FULL", "0").strip().lower() in {"1", "true", "yes"}


def _post(path: str, payload: Dict[str, Any]) -> Any:
    url = f"{BASE_URL}{path}"
    resp = requests.post(url, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    url = f"{BASE_URL}{path}"
    resp = requests.get(url, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    print(f"Using backend: {BASE_URL}")
    if PACE_SECONDS < 0:
        raise ValueError("BACKEND_PACE_SECONDS must be >= 0")
    print(
        "Config:",
        f"timeout={TIMEOUT}s",
        f"pace={PACE_SECONDS}s",
        f"full={FULL}",
    )

    idea = "在霓虹雨夜的赛博朋克城，黑客诗人要偷走企业之神的意识备份以唤醒被禁锢的街区灵魂。"
    print("\n[Step1] Generating loglines...")
    loglines: List[str] = _post("/api/v1/snowflake/step1", {"idea": idea})
    if PACE_SECONDS:
        time.sleep(PACE_SECONDS)
    chosen_logline = loglines[0]
    print("Chosen logline:", chosen_logline)

    print("\n[Step2] Generating root structure...")
    root = _post("/api/v1/snowflake/step2", {"logline": chosen_logline})
    if PACE_SECONDS:
        time.sleep(PACE_SECONDS)
    print(json.dumps(root, ensure_ascii=False, indent=2))

    print("\n[Step3] Generating characters...")
    characters = _post("/api/v1/snowflake/step3", root)
    if PACE_SECONDS:
        time.sleep(PACE_SECONDS)
    print(json.dumps(characters, ensure_ascii=False, indent=2))

    print("\n[Step4] Generating scenes...")
    step4 = _post(
        "/api/v1/snowflake/step4",
        {"root": root, "characters": characters},
    )
    if PACE_SECONDS:
        time.sleep(PACE_SECONDS)
    root_id = step4["root_id"]
    branch_id = step4["branch_id"]
    scenes = step4["scenes"]
    print("Persisted root_id:", root_id, "branch_id:", branch_id)
    print(f"Generated scenes: {len(scenes)} (showing first 2)")
    print(json.dumps(scenes[:2], ensure_ascii=False, indent=2))

    print("\n[Root Graph] Loading persisted graph snapshot from Kùzu...")
    graph = _get(f"/api/v1/roots/{root_id}", params={"branch_id": branch_id})
    if graph.get("root_id") != root_id:
        raise RuntimeError("graph.root_id mismatch")
    if not graph.get("relations"):
        raise RuntimeError("graph.relations is empty (expected persisted entity relations)")
    print(
        "Loaded graph:",
        f"characters={len(graph.get('characters', []))}",
        f"scenes={len(graph.get('scenes', []))}",
        f"relations={len(graph.get('relations', []))}",
    )

    print("\n[Entity] Creating a non-character entity and reloading entities list...")
    create_entity_payload = {
        "name": "霓虹雨夜广场",
        "entity_type": "location",
        "tags": ["赛博朋克", "地标"],
        "arc_status": None,
        "semantic_states": {"lighting": "neon", "weather": "rain"},
    }
    created = requests.post(
        f"{BASE_URL}/api/v1/roots/{root_id}/entities",
        params={"branch_id": branch_id},
        json=create_entity_payload,
        timeout=TIMEOUT,
    )
    created.raise_for_status()
    created_entity = created.json()
    entities = _get(f"/api/v1/roots/{root_id}/entities", params={"branch_id": branch_id})
    if not any(e.get("entity_id") == created_entity.get("entity_id") for e in entities):
        raise RuntimeError("created entity not found in entities list")

    entity_ids = [c["entity_id"] for c in characters]
    entity_ids.append(created_entity["entity_id"])
    if any(eid in (None, "") for eid in entity_ids):
        raise RuntimeError("entity_ids contains empty values")

    if scenes:
        print("\n[Context Load] Fetching context for first scene...")
        first_scene_id = scenes[0]["id"]
        if scenes[0].get("pov_character_id") in (None, ""):
            raise RuntimeError("first scene missing pov_character_id (expected POV assignment)")
        context = _get(
            f"/api/v1/scenes/{first_scene_id}/context",
            params={"branch_id": branch_id},
        )
        if context.get("root_id") != root_id:
            raise RuntimeError("context.root_id mismatch")
        summary = {
            "root_id": context.get("root_id"),
            "branch_id": context.get("branch_id"),
            "scene_id": first_scene_id,
            "scene_entity_count": len(context.get("scene_entities", [])),
            "character_count": len(context.get("characters", [])),
            "relation_count": len(context.get("relations", [])),
            "prev_scene_id": context.get("prev_scene_id"),
            "next_scene_id": context.get("next_scene_id"),
        }
        if summary["next_scene_id"] in (None, ""):
            raise RuntimeError("context.next_scene_id missing (expected SceneNext edge)")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    if FULL:
        print("\n[Logic Check] Simulating reasoning with potential force-execute...")
        logic_payload = {
            "outline_requirement": "主角应在本场遭遇挫折，并保留与反派的距离",
            "world_state": {"hero_hp": "100%", "enemy_distance": "20km"},
            "user_intent": "让主角立刻黑入反派中枢并正面交锋",
            "mode": "force_execute",
        }
        logic_result = _post("/api/v1/logic/check", logic_payload)
        if PACE_SECONDS:
            time.sleep(PACE_SECONDS)
        print(json.dumps(logic_result, ensure_ascii=False, indent=2))

        print("\n[State Extract] Extracting state deltas from generated prose...")
        sample_prose = (
            "雨水顺着霓虹落下，黑客诗人将神经接口插入暗巷终端。"
            "他的生命值从满格骤降，只剩下20%，但仍咬牙完成上传。"
        )
        extract_payload = {"content": sample_prose, "entity_ids": entity_ids}
        proposals = _post("/api/v1/state/extract", extract_payload)
        if PACE_SECONDS:
            time.sleep(PACE_SECONDS)
        print(json.dumps(proposals, ensure_ascii=False, indent=2))

        if not proposals:
            raise RuntimeError("state/extract returned empty proposals (expected >= 1).")

        print("\n[State Commit] Committing first proposal back to graph...")
        commit_payload = [proposals[0]]
        commit_result = requests.post(
            f"{BASE_URL}/api/v1/state/commit",
            params={"root_id": root_id, "branch_id": branch_id},
            json=commit_payload,
            timeout=TIMEOUT,
        )
        commit_result.raise_for_status()
        commit_result = commit_result.json()
        print(commit_result)

    if scenes:
        print("\n[Dirty Flag] Marking first scene as dirty for lazy repair...")
        dirty_result = requests.post(
            f"{BASE_URL}/api/v1/scenes/{first_scene_id}/dirty",
            params={"branch_id": branch_id},
            json={},
            timeout=TIMEOUT,
        )
        dirty_result.raise_for_status()
        dirty_result = dirty_result.json()
        print(dirty_result)
        dirty_scenes = _get(
            f"/api/v1/roots/{root_id}/dirty_scenes", params={"branch_id": branch_id}
        )
        if first_scene_id not in dirty_scenes:
            raise RuntimeError("dirty_scenes does not contain the marked scene")
        print(f"Dirty scenes: {len(dirty_scenes)} (first={dirty_scenes[0]})")

    print("\nAll requests completed. Check backend logs for detailed behavior.")


if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as exc:
        print(f"HTTP error: {exc.response.status_code} {exc.response.text}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - runtime safety
        print(f"Unexpected error: {exc}", file=sys.stderr)
        sys.exit(1)
