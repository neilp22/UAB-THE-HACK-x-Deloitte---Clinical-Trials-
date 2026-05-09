#!/usr/bin/env python3
"""
Grid search over scoring weights using cached TREC 2021 predictions.

Reads data/predictions/full_pipeline_2021.json (which includes phase, status,
and eligibility_summary per trial) and re-scores without any new LLM calls.

Usage:
    python scripts/tune_weights.py
    python scripts/tune_weights.py --predictions data/predictions/full_pipeline_2021.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytrec_eval

from src.config import DATA_DIR, TREC_2021_DIR
from src.matching.label_deriver import derive_label

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

GRID = {
    "A": [0.35, 0.45, 0.55],   # inclusion_met weight
    "C": [0.05, 0.10, 0.15],   # phase_bonus weight
    "D": [0.10, 0.15, 0.20],   # recruiting_bonus weight
    "E": [0.10, 0.20, 0.30],   # nei_penalty weight
}

_PHASE_MAP = {"PHASE3": 1.0, "PHASE2": 0.6, "PHASE1": 0.3}


def _norm(raw: str) -> str:
    return raw.upper().replace(" ", "").replace("_", "") if raw else ""


def rescore(trial: dict, A: float, C: float, D: float, E: float) -> float:
    s = trial["eligibility_summary"]
    inc_met = s.get("inclusion_met", 0)
    inc_not_met = s.get("inclusion_not_met", 0)
    inc_nei = s.get("inclusion_nei", 0)
    excl_viol = s.get("exclusion_violations", 0)

    total_inclusion = inc_met + inc_not_met + inc_nei
    total_criteria = total_inclusion + excl_viol  # approx (excl_met not stored)

    inclusion_met_ratio = inc_met / max(total_inclusion, 1)
    exclusion_penalty = 1.0 if excl_viol > 0 else 0.0
    nei_ratio = (inc_nei) / max(total_criteria, 1)

    phase_raw = _norm(trial.get("phase", ""))
    phase_bonus = _PHASE_MAP.get(phase_raw, 0.1)
    status_raw = _norm(trial.get("status", ""))
    recruiting_bonus = 1.0 if status_raw == "RECRUITING" else 0.0

    raw = (
        A * inclusion_met_ratio
        - 1.00 * exclusion_penalty
        + C * phase_bonus
        + D * recruiting_bonus
        - E * nei_ratio
    )
    return max(raw, 0.0)


def compute_ndcg10(run: dict[str, dict[str, float]],
                   qrels: dict[str, dict[str, int]]) -> float:
    eval_qrels = {tid: qrels[tid] for tid in run if tid in qrels}
    if not eval_qrels:
        return 0.0
    evaluator = pytrec_eval.RelevanceEvaluator(eval_qrels, {"ndcg_cut_10", "recall_20", "map"})
    per_topic = evaluator.evaluate(run)
    return statistics.mean(m["ndcg_cut_10"] for m in per_topic.values())


def compute_composite(run, qrels, t2_micro_f1: float) -> float:
    eval_qrels = {tid: qrels[tid] for tid in run if tid in qrels}
    if not eval_qrels:
        return 0.0
    evaluator = pytrec_eval.RelevanceEvaluator(eval_qrels, {"ndcg_cut_10", "recall_20"})
    per_topic = evaluator.evaluate(run)
    t1 = statistics.mean(m["recall_20"] for m in per_topic.values())
    t3 = statistics.mean(m["ndcg_cut_10"] for m in per_topic.values())
    return 0.20 * t1 + 0.30 * t2_micro_f1 + 0.25 * t3


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            tid, _, nct_id, grade = parts[0], parts[1], parts[2], int(parts[3])
            qrels.setdefault(tid, {})[nct_id] = grade
    return qrels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=str(DATA_DIR / "predictions" / "full_pipeline_2021.json"))
    parser.add_argument("--qrels", default=str(TREC_2021_DIR / "qrels.txt"))
    args = parser.parse_args()

    predictions = json.loads(Path(args.predictions).read_text())
    qrels = load_qrels(Path(args.qrels))

    # Baseline T2 (fixed — not affected by score weights)
    from sklearn.metrics import f1_score
    _GRADE_TO_LABEL = {2: "MET", 1: "NOT_MET", 0: "NEI"}

    y_true, y_pred_base = [], []
    for entry in predictions:
        tid = str(entry["patient_id"])
        for trial in entry["ranked_trials"]:
            if trial["nct_id"] in qrels.get(tid, {}):
                y_true.append(_GRADE_TO_LABEL[qrels[tid][trial["nct_id"]]])
                y_pred_base.append(derive_label(trial["eligibility_summary"]))
    t2_micro_f1 = f1_score(y_true, y_pred_base, average="micro",
                            labels=["MET", "NOT_MET", "NEI"], zero_division=0) if y_true else 0.0

    # Build baseline run for reference
    baseline_run: dict[str, dict[str, float]] = {}
    for entry in predictions:
        tid = str(entry["patient_id"])
        baseline_run[tid] = {t["nct_id"]: t["score"] for t in entry["ranked_trials"]}

    baseline_ndcg = compute_ndcg10(baseline_run, qrels)
    baseline_composite = compute_composite(baseline_run, qrels, t2_micro_f1)

    print(f"Baseline — NDCG@10={baseline_ndcg:.4f}  Composite={baseline_composite:.4f}  T2={t2_micro_f1:.4f}")
    print(f"Grid size: {len(GRID['A']) * len(GRID['C']) * len(GRID['D']) * len(GRID['E'])} combinations\n")

    best_ndcg = baseline_ndcg
    best_composite = baseline_composite
    best_weights: dict = {}

    for A, C, D, E in product(GRID["A"], GRID["C"], GRID["D"], GRID["E"]):
        run: dict[str, dict[str, float]] = {}
        for entry in predictions:
            tid = str(entry["patient_id"])
            run[tid] = {
                t["nct_id"]: rescore(t, A, C, D, E)
                for t in entry["ranked_trials"]
            }
        ndcg = compute_ndcg10(run, qrels)
        composite = compute_composite(run, qrels, t2_micro_f1)
        if composite > best_composite:
            best_composite = composite
            best_ndcg = ndcg
            best_weights = {"A": A, "C": C, "D": D, "E": E}

    if best_weights:
        print(f"Best weights found : A={best_weights['A']}  C={best_weights['C']}  "
              f"D={best_weights['D']}  E={best_weights['E']}")
        print(f"Best NDCG@10       : {best_ndcg:.4f}")
        print(f"Best Composite     : {best_composite:.4f}")
        improvement = best_composite - baseline_composite
        print(f"Improvement vs baseline : {improvement:+.4f}")
    else:
        print("No improvement found — baseline weights are optimal.")
        best_weights = {"A": 0.45, "C": 0.10, "D": 0.15, "E": 0.20}

    return best_weights


if __name__ == "__main__":
    result = main()
    print(f"\nFinal weights to apply: {result}")
