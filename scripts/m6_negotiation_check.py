"""Minimal end-to-end check for Milestone M6 negotiation loop behaviors.

Run with: python scripts/m6_negotiation_check.py
Set BACKEND_BASE_URL to override the default http://127.0.0.1:8000.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, Optional

import requests

BASE_URL = os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT = float(os.getenv("BACKEND_TIMEOUT", "30"))
PACE_SECONDS = float(os.getenv("BACKEND_PACE_SECONDS", "0"))


def _post(path: str, payload: Any, *, params: Optional[Dict[str, Any]] = None) -> Any:
    url = f"{BASE_URL}{path}"
    resp = requests.post(url, params=params, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _get(path: str, *, params: Optional[Dict[str, Any]] = None) -> Any:
    url = f"{BASE_URL}{path}"
    resp = requests.get(url, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _find_entity_semantic_states(
    entities: list[dict[str, Any]],
    *,
    entity_id: str,
) -> dict[str, Any]:
    for entity in entities:
        if entity.get("entity_id") == entity_id:
            return entity.get("semantic_states", {})
    raise RuntimeError(f"entity not found in entities list: entity_id={entity_id}")


def _find_scene(
    scenes: list[dict[str, Any]],
    *,
    scene_id: str,
) -> dict[str, Any]:
    for scene in scenes:
        if scene.get("id") == scene_id:
            return scene
    raise RuntimeError(f"scene not found in root snapshot: scene_id={scene_id}")


def main() -> None:
    print(f"Using backend: {BASE_URL}")
    if PACE_SECONDS < 0:
        raise ValueError("BACKEND_PACE_SECONDS must be >= 0")

    idea = "M6 协商回归：ForceExecute 标记 + State Diff + 提交后场景完成态。"
    print("\n[Step1] Generating loglines...")
    loglines = _post("/api/v1/snowflake/step1", {"idea": idea})
    if not loglines:
        raise RuntimeError("step1 returned empty loglines")
    chosen_logline = loglines[0]
    print("Chosen logline:", chosen_logline)
    if PACE_SECONDS:
        time.sleep(PACE_SECONDS)

    print("\n[Step2] Generating root structure...")
    root = _post("/api/v1/snowflake/step2", {"logline": chosen_logline})
    print(json.dumps(root, ensure_ascii=False, indent=2))
    if PACE_SECONDS:
        time.sleep(PACE_SECONDS)

    print("\n[Step3] Generating characters...")
    characters = _post("/api/v1/snowflake/step3", root)
    if not characters:
        raise RuntimeError("step3 returned empty characters")
    if PACE_SECONDS:
        time.sleep(PACE_SECONDS)

    print("\n[Step4] Generating scenes...")
    step4 = _post("/api/v1/snowflake/step4", {"root": root, "characters": characters})
    root_id = step4["root_id"]
    branch_id = step4["branch_id"]
    scenes = step4["scenes"]
    if not scenes:
        raise RuntimeError("step4 returned empty scenes")
    first_scene_id = scenes[0]["id"]
    print("Persisted root_id:", root_id, "branch_id:", branch_id)
    print("First scene_id:", first_scene_id)
    if PACE_SECONDS:
        time.sleep(PACE_SECONDS)

    print("\n[M6-T1] ForceExecute should persist logic_exception + reason...")
    force_reason = "HITL: 强制执行以推进协商循环"
    logic_payload = {
        "outline_requirement": "本场应保持风险可控",
        "world_state": {"hero_hp": "100%"},
        "user_intent": "直接进入高风险行动（用于触发 force_execute）",
        "mode": "force_execute",
        "root_id": root_id,
        "branch_id": branch_id,
        "scene_id": first_scene_id,
        "force_reason": force_reason,
    }
    logic_result = _post("/api/v1/logic/check", logic_payload)
    print(json.dumps(logic_result, ensure_ascii=False, indent=2))
    if PACE_SECONDS:
        time.sleep(PACE_SECONDS)

    graph = _get(f"/api/v1/roots/{root_id}", params={"branch_id": branch_id})
    scene = _find_scene(graph.get("scenes", []), scene_id=first_scene_id)
    if scene.get("logic_exception") is not True:
        raise RuntimeError("expected scene.logic_exception=true after force_execute")
    if scene.get("logic_exception_reason") != force_reason:
        raise RuntimeError("expected scene.logic_exception_reason to match force_reason")
    print("OK: logic_exception + reason persisted and visible in root snapshot")

    print("\n[M6-T2] State extract should return diff and must be read-only...")
    target_entity_id = characters[0]["entity_id"]
    entities_before = _get(
        f"/api/v1/roots/{root_id}/entities", params={"branch_id": branch_id}
    )
    before_states = _find_entity_semantic_states(
        entities_before,
        entity_id=target_entity_id,
    )

    sample_prose = "主角生命值从满格骤降，只剩下20%，但仍继续前进。"
    extract_payload = {
        "content": sample_prose,
        "entity_ids": [target_entity_id],
        "root_id": root_id,
        "branch_id": branch_id,
    }
    proposals = _post("/api/v1/state/extract", extract_payload)
    if not proposals:
        raise RuntimeError("state/extract returned empty proposals")
    first = proposals[0]
    if "semantic_states_before" not in first or "semantic_states_after" not in first:
        raise RuntimeError("state/extract missing semantic_states_before/after for diff view")
    if first["semantic_states_before"] != before_states:
        raise RuntimeError("semantic_states_before mismatch")
    if first["semantic_states_after"].get("hero_hp") != "20%":
        raise RuntimeError("semantic_states_after missing expected patch value hero_hp=20%")

    entities_after_extract = _get(
        f"/api/v1/roots/{root_id}/entities", params={"branch_id": branch_id}
    )
    after_extract_states = _find_entity_semantic_states(
        entities_after_extract,
        entity_id=target_entity_id,
    )
    if after_extract_states != before_states:
        raise RuntimeError("state/extract should be read-only, but semantic_states changed")
    print("OK: extract returned diff and did not persist changes")

    print("\n[M6-T2] Commit + scene complete should persist actual_outcome + status...")
    commit_result = _post(
        "/api/v1/state/commit",
        [first],
        params={"root_id": root_id, "branch_id": branch_id},
    )
    print(commit_result)

    entities_after_commit = _get(
        f"/api/v1/roots/{root_id}/entities", params={"branch_id": branch_id}
    )
    after_commit_states = _find_entity_semantic_states(
        entities_after_commit,
        entity_id=target_entity_id,
    )
    if after_commit_states.get("hero_hp") != "20%":
        raise RuntimeError("state/commit did not persist semantic_states patch")

    actual_outcome = "主角带伤撤离，并拿到关键线索"
    summary = "主角带伤撤离，获得关键线索"
    complete_result = _post(
        f"/api/v1/scenes/{first_scene_id}/complete",
        {"actual_outcome": actual_outcome, "summary": summary},
        params={"branch_id": branch_id},
    )
    print(complete_result)

    graph2 = _get(f"/api/v1/roots/{root_id}", params={"branch_id": branch_id})
    scene2 = _find_scene(graph2.get("scenes", []), scene_id=first_scene_id)
    if scene2.get("status") != "committed":
        raise RuntimeError("expected scene.status=committed after complete")
    if scene2.get("actual_outcome") != actual_outcome:
        raise RuntimeError("expected scene.actual_outcome to be persisted after complete")
    print("OK: commit + complete persisted and visible in root snapshot")

    print("\nM6 negotiation checks passed.")


if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as exc:
        print(f"HTTP error: {exc.response.status_code} {exc.response.text}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - runtime safety
        print(f"Unexpected error: {exc}", file=sys.stderr)
        sys.exit(1)
