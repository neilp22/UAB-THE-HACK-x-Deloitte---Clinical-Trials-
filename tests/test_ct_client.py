"""
Validation tests for src/retrieval/ct_client.py.

Run with: python tests/test_ct_client.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.ct_client import search_trials


def test_basic_query() -> None:
    print("TEST: basic CT API query (lung cancer, 5 results)...")
    trials = search_trials(query_cond="lung cancer", max_results=5, page_size=5)
    assert len(trials) > 0, "Expected at least 1 result"
    assert len(trials) <= 5

    t = trials[0]
    required_fields = ["nct_id", "title", "status", "brief_summary",
                       "eligibility_criteria", "sex", "min_age", "max_age",
                       "std_ages", "phases", "conditions"]
    for field in required_fields:
        assert field in t, f"Missing field: {field}"
    assert t["nct_id"].startswith("NCT"), f"Invalid NCT ID: {t['nct_id']}"

    print(f"  PASS — got {len(trials)} trials")
    print(f"  Sample: {t['nct_id']} | {t['title'][:60]}")
    print(f"  Status: {t['status']}, Phases: {t['phases']}")


def test_query_term_only() -> None:
    print("TEST: query_term only...")
    trials = search_trials(query_term="EGFR mutation non-small cell lung", max_results=10)
    assert isinstance(trials, list)
    print(f"  PASS — got {len(trials)} trials")


def test_no_query_raises() -> None:
    print("TEST: missing query raises ValueError...")
    try:
        search_trials()
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  PASS — ValueError raised as expected")


def test_pagination_ceiling() -> None:
    print("TEST: max_results ceiling respected...")
    trials = search_trials(query_cond="cancer", max_results=7, page_size=10)
    assert len(trials) <= 7, f"Got {len(trials)}, expected <= 7"
    print(f"  PASS — got {len(trials)} trials (ceiling=7)")


if __name__ == "__main__":
    passed = 0
    failed = 0
    for name, fn in [
        ("basic_query", test_basic_query),
        ("query_term_only", test_query_term_only),
        ("no_query_raises", test_no_query_raises),
        ("pagination_ceiling", test_pagination_ceiling),
    ]:
        try:
            fn()
            passed += 1
        except Exception as exc:
            print(f"  FAIL — {exc}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
