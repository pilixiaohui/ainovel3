#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.models import CharacterSheet, SceneNode, SnowflakeRoot  # noqa: E402
from app.storage.graph import DEFAULT_BRANCH_ID, GraphStorage  # noqa: E402


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    index = int(math.ceil(ratio * len(ordered))) - 1
    index = max(0, min(index, len(ordered) - 1))
    return ordered[index]


def _measure_ms(action) -> float:
    start = time.perf_counter()
    action()
    return (time.perf_counter() - start) * 1000


def _build_seed() -> tuple[SnowflakeRoot, CharacterSheet, list[SceneNode]]:
    root = SnowflakeRoot(
        logline="Perf benchmark",
        three_disasters=["D1", "D2", "D3"],
        ending="End",
        theme="Theme",
    )
    character = CharacterSheet(
        name="Benchmark",
        ambition="A",
        conflict="C",
        epiphany="E",
        voice_dna="VD",
    )
    scenes = [
        SceneNode(
            branch_id=DEFAULT_BRANCH_ID,
            expected_outcome="Outcome 1",
            conflict_type="internal",
            actual_outcome="",
            parent_act_id=None,
            is_dirty=False,
            pov_character_id=character.entity_id,
        ),
        SceneNode(
            branch_id=DEFAULT_BRANCH_ID,
            expected_outcome="Outcome 2",
            conflict_type="external",
            actual_outcome="",
            parent_act_id=None,
            is_dirty=False,
            pov_character_id=character.entity_id,
        ),
    ]
    return root, character, scenes


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark branch creation, single-scene commit, and get_scene_context P50/P99."
        )
    )
    parser.add_argument("--db", dest="db_path", help="Optional Kùzu db path")
    parser.add_argument(
        "--branch-threshold-ms",
        type=float,
        default=100.0,
        help="Threshold for create_branch in ms",
    )
    parser.add_argument(
        "--commit-threshold-ms",
        type=float,
        default=50.0,
        help="Threshold for commit_scene in ms",
    )
    parser.add_argument(
        "--p50-threshold-ms",
        type=float,
        default=100.0,
        help="Threshold for P50 get_scene_context in ms",
    )
    parser.add_argument(
        "--p99-threshold-ms",
        type=float,
        default=300.0,
        help="Threshold for P99 get_scene_context in ms",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1000,
        help="Commit depth for get_scene_context chain",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Sample count for get_scene_context timing",
    )
    args = parser.parse_args()

    if args.depth < 1:
        raise ValueError("depth must be >= 1")
    if args.samples < 1:
        raise ValueError("samples must be >= 1")

    if args.db_path:
        db_path = Path(args.db_path)
        if not db_path.is_absolute():
            db_path = REPO_ROOT / db_path
    else:
        tmpdir = tempfile.mkdtemp(prefix="scene_version_bench_")
        db_path = Path(tmpdir) / "bench.db"

    storage = GraphStorage(db_path=db_path)
    try:
        root, _character, scenes = _build_seed()
        root_id = storage.save_snowflake(root, [_character], scenes)
        scene_a_id = str(scenes[0].id)
        scene_b_id = str(scenes[1].id)

        branch_time = _measure_ms(
            lambda: storage.create_branch(root_id=root_id, branch_id="bench")
        )
        print(
            f"create_branch: {branch_time:.2f}ms "
            f"(threshold {args.branch_threshold_ms:.2f}ms)"
        )
        if branch_time > args.branch_threshold_ms:
            raise RuntimeError("create_branch exceeded threshold")

        commit_time = _measure_ms(
            lambda: storage.commit_scene(
                root_id=root_id,
                branch_id="bench",
                scene_origin_id=scene_a_id,
                content={
                    "actual_outcome": "benchmark",
                    "summary": "benchmark",
                    "status": "committed",
                },
                message="benchmark",
            )
        )
        print(
            f"commit_scene: {commit_time:.2f}ms "
            f"(threshold {args.commit_threshold_ms:.2f}ms)"
        )
        if commit_time > args.commit_threshold_ms:
            raise RuntimeError("commit_scene exceeded threshold")

        for idx in range(args.depth):
            storage.commit_scene(
                root_id=root_id,
                branch_id=DEFAULT_BRANCH_ID,
                scene_origin_id=scene_b_id,
                content={
                    "actual_outcome": f"depth {idx}",
                    "summary": f"depth {idx}",
                    "status": "committed",
                },
                message=f"depth {idx}",
            )

        storage.get_scene_context(
            scene_id=scene_a_id, branch_id=DEFAULT_BRANCH_ID
        )
        context_times: list[float] = []
        for _ in range(args.samples):
            context_times.append(
                _measure_ms(
                    lambda: storage.get_scene_context(
                        scene_id=scene_a_id,
                        branch_id=DEFAULT_BRANCH_ID,
                    )
                )
            )

        p50 = _percentile(context_times, 0.50)
        p99 = _percentile(context_times, 0.99)
        print(
            "get_scene_context_p50: "
            f"{p50:.2f}ms (threshold {args.p50_threshold_ms:.2f}ms)"
        )
        print(
            "get_scene_context_p99: "
            f"{p99:.2f}ms (threshold {args.p99_threshold_ms:.2f}ms)"
        )
        if p50 > args.p50_threshold_ms:
            raise RuntimeError("get_scene_context P50 exceeded threshold")
        if p99 > args.p99_threshold_ms:
            raise RuntimeError("get_scene_context P99 exceeded threshold")

        print("Benchmark passed.")
    finally:
        storage.close()


if __name__ == "__main__":
    main()
