#!/usr/bin/env python3
"""End-to-end validation script for the Imaging Intelligence Agent.

Runs 5 validation checks:
  (a) Milvus connection and collection counts
  (b) RAG query returns results
  (c) NIM service status
  (d) API health endpoint responds
  (e) Workflow registry has 4 workflows

Usage:
    python scripts/validate_e2e.py
    python scripts/validate_e2e.py --api-url http://localhost:8524
    python scripts/validate_e2e.py --host 10.0.0.1 --port 19530

Author: Adam Jones
Date: February 2026
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger


def check_milvus(host, port):
    """Check (a): Milvus connection and collection counts."""
    try:
        from src.collections import ImagingCollectionManager, OWNED_COLLECTION_SCHEMAS

        manager = ImagingCollectionManager(host=host, port=port)
        manager.connect()

        stats = manager.get_collection_stats()
        existing = {name: count for name, count in stats.items() if count >= 0}
        total_records = sum(stats.values())

        # Verify all 10 owned collections exist
        expected = set(OWNED_COLLECTION_SCHEMAS.keys())
        # Check which collections actually exist (count >= 0 means they were queried)
        from pymilvus import utility
        found_collections = set()
        for name in expected:
            if utility.has_collection(name, using="imaging"):
                found_collections.add(name)

        missing = expected - found_collections
        manager.disconnect()

        if missing:
            return (
                "FAIL",
                f"Missing collections: {missing}. "
                f"Found {len(found_collections)}/{len(expected)} collections, "
                f"{total_records:,} total records.",
            )

        return (
            "PASS",
            f"All {len(expected)} collections exist. "
            f"{total_records:,} total records across all collections.",
        )

    except Exception as e:
        return "FAIL", f"Milvus connection failed: {e}"


def check_rag_query(host, port):
    """Check (b): RAG query returns results."""
    try:
        from sentence_transformers import SentenceTransformer

        from config.settings import settings
        from src.collections import ImagingCollectionManager

        embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
        manager = ImagingCollectionManager(host=host, port=port)
        manager.connect()

        # Run a simple vector search on imaging_findings
        test_query = "CT head hemorrhage detection"
        query_embedding = embedder.encode(
            test_query, normalize_embeddings=True
        ).tolist()

        results = manager.search(
            collection_name="imaging_findings",
            query_embedding=query_embedding,
            top_k=3,
        )

        manager.disconnect()

        if results:
            top_score = results[0]["score"]
            return (
                "PASS",
                f"RAG search returned {len(results)} results. "
                f"Top score: {top_score:.3f}.",
            )
        else:
            # No results could mean empty collection, which is OK for initial setup
            return (
                "WARN",
                "RAG search returned 0 results. "
                "Collections may be empty -- run seed scripts first.",
            )

    except Exception as e:
        return "FAIL", f"RAG query failed: {e}"


def check_nim_services():
    """Check (c): NIM service status."""
    try:
        from config.settings import settings
        from src.nim.service_manager import NIMServiceManager

        nim_manager = NIMServiceManager(settings)
        status = nim_manager.check_all_services()

        available = sum(1 for s in status.values() if s == "available")
        mock = sum(1 for s in status.values() if s == "mock")
        unavailable = sum(1 for s in status.values() if s == "unavailable")

        status_str = ", ".join(f"{k}={v}" for k, v in status.items())

        if unavailable == 0:
            return (
                "PASS",
                f"All NIM services OK ({available} available, {mock} mock). "
                f"{status_str}",
            )
        elif available + mock > 0:
            return (
                "WARN",
                f"Partial NIM availability ({available} available, {mock} mock, "
                f"{unavailable} unavailable). {status_str}",
            )
        else:
            return (
                "FAIL",
                f"No NIM services available. {status_str}",
            )

    except Exception as e:
        return "FAIL", f"NIM service check failed: {e}"


def check_api_health(api_url):
    """Check (d): API health endpoint responds."""
    try:
        import requests

        url = f"{api_url.rstrip('/')}/health"
        logger.info(f"Checking API health at {url}")
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            status_str = data.get("status", "unknown")
            return "PASS", f"API health endpoint returned 200. Status: {status_str}."
        else:
            return (
                "FAIL",
                f"API health endpoint returned HTTP {response.status_code}.",
            )

    except Exception as e:
        return (
            "WARN",
            f"API health endpoint unreachable at {api_url}: {e}. "
            f"API server may not be running.",
        )


def check_workflow_registry():
    """Check (e): Workflow registry has 4 workflows."""
    try:
        from src.workflows import WORKFLOW_REGISTRY

        count = len(WORKFLOW_REGISTRY)
        names = list(WORKFLOW_REGISTRY.keys())

        if count == 4:
            return (
                "PASS",
                f"Workflow registry has {count} workflows: {names}.",
            )
        elif count > 0:
            return (
                "WARN",
                f"Workflow registry has {count} workflows (expected 4): {names}.",
            )
        else:
            return "FAIL", "Workflow registry is empty."

    except Exception as e:
        return "FAIL", f"Workflow registry check failed: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end validation for Imaging Intelligence Agent"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8524",
        help="API server URL (default: http://localhost:8524)",
    )
    parser.add_argument(
        "--host", default=None, help="Milvus host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Milvus port (default: 19530)"
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  Imaging Intelligence Agent — End-to-End Validation")
    print("=" * 65)
    print(f"  API URL: {args.api_url}")
    print()

    checks = [
        ("(a) Milvus Collections", lambda: check_milvus(args.host, args.port)),
        ("(b) RAG Query", lambda: check_rag_query(args.host, args.port)),
        ("(c) NIM Services", check_nim_services),
        ("(d) API Health", lambda: check_api_health(args.api_url)),
        ("(e) Workflow Registry", check_workflow_registry),
    ]

    results = []

    for check_name, check_fn in checks:
        logger.info(f"Running check: {check_name}")
        start = time.time()

        try:
            status, detail = check_fn()
        except Exception as e:
            status, detail = "FAIL", f"Unexpected error: {e}"

        elapsed = (time.time() - start) * 1000
        results.append((check_name, status, detail, elapsed))

        symbol = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}.get(
            status, "[????]"
        )
        print(f"  {symbol} {check_name} ({elapsed:.0f}ms)")
        logger.info(f"  {detail}")

    # --- Summary ---
    print()
    print("=" * 65)
    print("  Validation Summary")
    print("=" * 65)

    for check_name, status, detail, elapsed in results:
        symbol = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}.get(
            status, "[????]"
        )
        print(f"  {symbol} {check_name}")
        print(f"         {detail}")

    passed = sum(1 for _, s, _, _ in results if s == "PASS")
    warned = sum(1 for _, s, _, _ in results if s == "WARN")
    failed = sum(1 for _, s, _, _ in results if s == "FAIL")
    total = len(results)

    print()
    print(f"  Passed: {passed}/{total}  |  Warnings: {warned}  |  Failed: {failed}")

    if failed == 0:
        print("  Overall: PASS")
    elif passed > 0:
        print("  Overall: PARTIAL")
    else:
        print("  Overall: FAIL")

    print("=" * 65)

    # Exit with non-zero if any checks failed
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
