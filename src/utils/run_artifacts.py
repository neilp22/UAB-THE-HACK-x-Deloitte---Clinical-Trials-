"""Utilities for per-run artifact folders.

These helpers create reproducible run folders such as:

runs/YYYY-MM-DD_run_001/
├── config_snapshot.yaml
├── query_plans.jsonl
├── retrieved_trials.jsonl
├── parsed_criteria.jsonl
├── criterion_evaluations.jsonl
├── rankings.json
├── predictions.json
├── metrics.json
└── error_analysis.md

No LLM call is used here.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


INTERMEDIATE_FILES = [
    "query_plans.jsonl",
    "retrieved_trials.jsonl",
    "parsed_criteria.jsonl",
    "criterion_evaluations.jsonl",
]


def make_run_dir(root: str | Path = "runs", run_id: str | None = None) -> Path:
    """Create and return a new run directory.

    If run_id is provided, it is used directly.
    Otherwise, create YYYY-MM-DD_run_001, YYYY-MM-DD_run_002, ...
    """

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    if run_id:
        run_dir = root_path / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    prefix = datetime.now().strftime("%Y-%m-%d")
    counter = 1

    while True:
        candidate = root_path / f"{prefix}_run_{counter:03d}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        counter += 1


def _to_plain(value: Any) -> Any:
    """Convert common non-serializable values to plain JSON/YAML-friendly values."""

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]

    return value


def write_json(path: str | Path, data: Any) -> None:
    """Write pretty JSON."""

    Path(path).write_text(
        json.dumps(_to_plain(data), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    """Write JSON Lines."""

    with Path(path).open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(_to_plain(row), ensure_ascii=False) + "\n")


def write_config_snapshot(run_dir: str | Path, config: dict[str, Any]) -> Path:
    """Write a simple YAML-like config snapshot without requiring PyYAML."""

    run_dir = Path(run_dir)
    path = run_dir / "config_snapshot.yaml"

    lines = ["# Auto-generated run configuration snapshot\n"]

    for key, value in _to_plain(config).items():
        if isinstance(value, (dict, list)):
            lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}\n")
        else:
            lines.append(f"{key}: {value}\n")

    path.write_text("".join(lines), encoding="utf-8")
    return path


def ensure_intermediate_files(run_dir: str | Path) -> None:
    """Create empty intermediate JSONL files if they do not exist yet."""

    run_dir = Path(run_dir)
    for name in INTERMEDIATE_FILES:
        path = run_dir / name
        if not path.exists():
            path.write_text("", encoding="utf-8")


def build_rankings(predictions: list[dict]) -> list[dict]:
    """Build compact ranking rows from predictions."""

    rankings: list[dict] = []

    for entry in predictions:
        patient_id = entry.get("patient_id")
        for trial in entry.get("ranked_trials", []):
            rankings.append(
                {
                    "patient_id": patient_id,
                    "rank": trial.get("rank"),
                    "nct_id": trial.get("nct_id"),
                    "score": trial.get("score"),
                    "title": trial.get("title", ""),
                    "phase": trial.get("phase", ""),
                    "status": trial.get("status", ""),
                    "score_breakdown": trial.get("score_breakdown", {}),
                    "score_explanation": trial.get("score_explanation", ""),
                }
            )

    return rankings


def build_retrieved_trials_rows(predictions: list[dict]) -> list[dict]:
    """Build JSONL rows describing ranked/retrieved trials.

    This is a lightweight artifact derived from the final ranking.
    Deeper retrieval traces can be added later in P1.
    """

    rows: list[dict] = []

    for entry in predictions:
        patient_id = entry.get("patient_id")
        for trial in entry.get("ranked_trials", []):
            rows.append(
                {
                    "patient_id": patient_id,
                    "nct_id": trial.get("nct_id"),
                    "rank": trial.get("rank"),
                    "score": trial.get("score"),
                    "retrieval_trace": "derived_from_ranked_trials",
                }
            )

    return rows


def copy_if_exists(source: str | Path, target: str | Path) -> None:
    """Copy a file if it exists."""

    source_path = Path(source)
    target_path = Path(target)

    if source_path.exists() and source_path.is_file():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def copy_dossiers_if_exists(source_dir: str | Path, target_dir: str | Path) -> None:
    """Copy generated dossiers into the run folder if they exist."""

    source_path = Path(source_dir)
    target_path = Path(target_dir)

    if source_path.exists() and source_path.is_dir():
        shutil.copytree(source_path, target_path, dirs_exist_ok=True)


def write_error_analysis_stub(
    run_dir: str | Path,
    summary: dict[str, Any],
) -> Path:
    """Write a basic error analysis stub to be filled after review."""

    run_dir = Path(run_dir)
    path = run_dir / "error_analysis.md"

    lines = [
        "# Error Analysis\n\n",
        "## Run summary\n\n",
    ]

    for key, value in _to_plain(summary).items():
        lines.append(f"- **{key}**: {value}\n")

    lines.extend(
        [
            "\n## Observed issues\n\n",
            "- TODO: inspect low Recall@20 topics.\n",
            "- TODO: inspect wrong top-ranked trials.\n",
            "- TODO: inspect NEI-heavy trials.\n",
            "\n## Hypotheses for next iteration\n\n",
            "- TODO: add retrieval/query/rescue hypotheses here.\n",
        ]
    )

    path.write_text("".join(lines), encoding="utf-8")
    return path


def materialize_run_artifacts(
    run_dir: str | Path,
    *,
    predictions: list[dict],
    metrics: dict[str, Any],
    run_file_path: str | Path | None = None,
    predictions_path: str | Path | None = None,
    dossiers_dir: str | Path | None = None,
    summary: dict[str, Any] | None = None,
) -> None:
    """Write the standard run artifacts folder."""

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    ensure_intermediate_files(run_dir)

    write_json(run_dir / "predictions.json", predictions)
    write_json(run_dir / "rankings.json", build_rankings(predictions))
    write_jsonl(
        run_dir / "retrieved_trials.jsonl",
        build_retrieved_trials_rows(predictions),
    )
    write_json(run_dir / "metrics.json", metrics)

    if run_file_path is not None:
        copy_if_exists(run_file_path, run_dir / "trec_run.txt")

    if predictions_path is not None:
        copy_if_exists(predictions_path, run_dir / "predictions.copy.json")

    if dossiers_dir is not None:
        copy_dossiers_if_exists(dossiers_dir, run_dir / "dossiers")

    write_error_analysis_stub(run_dir, summary or {})