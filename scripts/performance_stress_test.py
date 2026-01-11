"""Manual performance/stress checks for state/extract.

Run with:
  python project/scripts/performance_stress_test.py \
    --p0-threshold-seconds 30 \
    --p1-threshold-seconds 180

Requires:
  - Backend running with SNOWFLAKE_ENGINE=gemini
  - Valid TOPONE_API_KEY
  - BACKEND_BASE_URL/ BACKEND_TIMEOUT optional overrides
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Any

import requests

BASE_URL = os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT = float(os.getenv("BACKEND_TIMEOUT", "120"))


def _build_text(word_count: int) -> str:
    if word_count <= 0:
        raise ValueError("word_count must be > 0")
    return ("word " * word_count).strip()


def _post_state_extract(payload: dict[str, Any]) -> None:
    resp = requests.post(
        f"{BASE_URL}/api/v1/state/extract",
        json=payload,
        timeout=TIMEOUT,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"state/extract failed: {resp.status_code} {resp.text}")


def _run_case(name: str, *, word_count: int, threshold_seconds: float) -> float:
    payload = {
        "content": _build_text(word_count),
        "entity_ids": ["entity-1"],
    }
    start = time.perf_counter()
    _post_state_extract(payload)
    elapsed = time.perf_counter() - start
    if elapsed > threshold_seconds:
        raise RuntimeError(
            f"{name} exceeded threshold: {elapsed:.2f}s > {threshold_seconds:.2f}s"
        )
    print(f"{name}: {elapsed:.2f}s (threshold {threshold_seconds:.2f}s)")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P0/P1 performance checks for /api/v1/state/extract."
    )
    parser.add_argument(
        "--p0-threshold-seconds",
        type=float,
        required=True,
        help="P0 response time threshold in seconds.",
    )
    parser.add_argument(
        "--p1-threshold-seconds",
        type=float,
        required=True,
        help="P1 response time threshold in seconds.",
    )
    parser.add_argument(
        "--p0-words",
        type=int,
        default=2000,
        help="Word count for P0 baseline payload.",
    )
    parser.add_argument(
        "--p1-words",
        type=int,
        default=200000,
        help="Word count for P1 stress payload.",
    )
    args = parser.parse_args()

    if args.p0_threshold_seconds <= 0:
        raise ValueError("p0_threshold_seconds must be > 0")
    if args.p1_threshold_seconds <= 0:
        raise ValueError("p1_threshold_seconds must be > 0")

    print(f"Using backend: {BASE_URL}")
    print(
        "Config:",
        f"timeout={TIMEOUT}s",
        f"p0_words={args.p0_words}",
        f"p1_words={args.p1_words}",
    )

    _run_case(
        "P0",
        word_count=args.p0_words,
        threshold_seconds=args.p0_threshold_seconds,
    )
    _run_case(
        "P1",
        word_count=args.p1_words,
        threshold_seconds=args.p1_threshold_seconds,
    )

    print("Performance checks passed.")


if __name__ == "__main__":
    main()
