"""Tests for per-run artifact folder utilities."""

from __future__ import annotations

import json
from pathlib import Path

from src.utils.run_artifacts import (
    build_rankings,
    make_run_dir,
    materialize_run_artifacts,
    write_config_snapshot,
)


def _fake_predictions() -> list[dict]:
    return [
        {
            "patient_id": "1",
            "ranked_trials": [
                {
                    "rank": 1,
                    "nct_id": "NCT001",
                    "score": 0.75,
                    "title": "Trial A",
                    "phase": "PHASE2",
                    "status": "RECRUITING",
                    "score_breakdown": {
                        "inclusion_met_ratio": 0.8,
                        "exclusion_penalty": 0.0,
                        "final_score": 0.75,
                    },
                    "score_explanation": "Ranked based on strong inclusion match.",
                },
                {
                    "rank": 2,
                    "nct_id": "NCT002",
                    "score": 0.40,
                    "title": "Trial B",
                    "phase": "PHASE1",
                    "status": "COMPLETED",
                    "score_breakdown": {
                        "inclusion_met_ratio": 0.4,
                        "exclusion_penalty": 0.0,
                        "final_score": 0.40,
                    },
                    "score_explanation": "Ranked based on partial inclusion match.",
                },
            ],
        }
    ]


def test_make_run_dir_creates_incremental_directories(tmp_path: Path):
    first = make_run_dir(tmp_path)
    second = make_run_dir(tmp_path)

    assert first.exists()
    assert second.exists()
    assert first != second
    assert first.name.endswith("_run_001")
    assert second.name.endswith("_run_002")


def test_write_config_snapshot_creates_yaml_file(tmp_path: Path):
    run_dir = make_run_dir(tmp_path)

    path = write_config_snapshot(
        run_dir,
        {
            "year": 2021,
            "retrieval_mode": "combined",
            "topics_limit": 1,
        },
    )

    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "year: 2021" in text
    assert "retrieval_mode: combined" in text
    assert "topics_limit: 1" in text


def test_build_rankings_extracts_compact_rows():
    rows = build_rankings(_fake_predictions())

    assert len(rows) == 2
    assert rows[0]["patient_id"] == "1"
    assert rows[0]["nct_id"] == "NCT001"
    assert rows[0]["rank"] == 1
    assert rows[0]["score_breakdown"]["final_score"] == 0.75
    assert "strong inclusion match" in rows[0]["score_explanation"]


def test_materialize_run_artifacts_writes_expected_files(tmp_path: Path):
    run_dir = make_run_dir(tmp_path)

    materialize_run_artifacts(
        run_dir,
        predictions=_fake_predictions(),
        metrics={
            "recall_20": 0.1,
            "micro_f1": 0.2,
            "ndcg_cut_10": 0.3,
        },
        summary={
            "topics_evaluated": 1,
            "failed_topics": 0,
        },
    )

    expected_files = [
        "query_plans.jsonl",
        "retrieved_trials.jsonl",
        "parsed_criteria.jsonl",
        "criterion_evaluations.jsonl",
        "rankings.json",
        "predictions.json",
        "metrics.json",
        "error_analysis.md",
    ]

    for name in expected_files:
        assert (run_dir / name).exists(), name

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["recall_20"] == 0.1
    assert metrics["micro_f1"] == 0.2
    assert metrics["ndcg_cut_10"] == 0.3

    rankings = json.loads((run_dir / "rankings.json").read_text(encoding="utf-8"))
    assert rankings[0]["nct_id"] == "NCT001"
    assert rankings[0]["score_breakdown"]["final_score"] == 0.75

    predictions = json.loads(
        (run_dir / "predictions.json").read_text(encoding="utf-8")
    )
    assert predictions[0]["patient_id"] == "1"

    retrieved_text = (run_dir / "retrieved_trials.jsonl").read_text(
        encoding="utf-8"
    )
    assert "NCT001" in retrieved_text

    error_analysis = (run_dir / "error_analysis.md").read_text(encoding="utf-8")
    assert "# Error Analysis" in error_analysis
    assert "topics_evaluated" in error_analysis