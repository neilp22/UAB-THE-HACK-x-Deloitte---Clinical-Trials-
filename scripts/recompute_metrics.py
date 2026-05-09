"""
Recompute T2 Micro-F1 from cached predictions — zero LLM cost.

Loads data/predictions/full_pipeline_2021.json (eligibility_summary raw counts),
applies both OLD and NEW _derive_label() logic, and prints a before/after table.
T1 (Recall@20) and T3 (NDCG@10) are loaded from the run file (scores unchanged).

Usage:
    python scripts/recompute_metrics.py [--predictions PATH] [--run PATH] [--qrels PATH]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import pytrec_eval
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.matching.label_deriver import derive_label as _new_derive_label

# ---------------------------------------------------------------------------
# Label derivation — two versions
# ---------------------------------------------------------------------------

def _old_derive_label(s: dict) -> str:
    """Original (pre-gate) logic kept for before/after comparison only."""
    if s.get("exclusion_violations", 0) > 0:
        return "NOT_MET"
    if s.get("inclusion_met", 0) > 0:
        return "MET"
    return "NEI"


_GRADE_TO_LABEL = {2: "MET", 1: "NOT_MET", 0: "NEI"}

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_t2(predictions: list[dict], qrels: dict[str, dict[str, int]],
               derive_fn) -> tuple[float, dict]:
    y_true: list[str] = []
    y_pred: list[str] = []
    for entry in predictions:
        tid = str(entry["patient_id"])
        topic_qrels = qrels.get(tid, {})
        for trial in entry["ranked_trials"]:
            nct_id = trial["nct_id"]
            if nct_id not in topic_qrels:
                continue
            grade = topic_qrels[nct_id]
            y_true.append(_GRADE_TO_LABEL[grade])
            y_pred.append(derive_fn(trial["eligibility_summary"]))
    if not y_true:
        return 0.0, {}
    micro_f1 = f1_score(y_true, y_pred, average="micro",
                        labels=["MET", "NOT_MET", "NEI"], zero_division=0)
    report = classification_report(y_true, y_pred,
                                   labels=["MET", "NOT_MET", "NEI"],
                                   zero_division=0, output_dict=True)
    return micro_f1, report


def load_run(run_path: Path) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = {}
    with open(run_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            tid, _, nct_id, _, score, _ = parts
            run.setdefault(tid, {})[nct_id] = float(score)
    return run


def compute_retrieval_metrics(run: dict[str, dict[str, float]],
                              qrels: dict[str, dict[str, int]]) -> dict[str, float]:
    eval_qrels = {tid: qrels[tid] for tid in run if tid in qrels}
    if not eval_qrels:
        return {"ndcg_cut_10": 0.0, "recall_20": 0.0, "map": 0.0}
    evaluator = pytrec_eval.RelevanceEvaluator(eval_qrels, {"ndcg_cut_10", "recall_20", "map"})
    per_topic = evaluator.evaluate(run)
    return {
        m: statistics.mean(v[m] for v in per_topic.values())
        for m in ("ndcg_cut_10", "recall_20", "map")
    }


def load_qrels(qrels_path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(qrels_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            tid, _, nct_id, grade = parts[0], parts[1], parts[2], int(parts[3])
            qrels.setdefault(tid, {})[nct_id] = grade
    return qrels


def label_distribution(predictions: list[dict], qrels: dict[str, dict[str, int]],
                        derive_fn) -> dict[str, int]:
    dist: dict[str, int] = {"MET": 0, "NOT_MET": 0, "NEI": 0}
    for entry in predictions:
        tid = str(entry["patient_id"])
        topic_qrels = qrels.get(tid, {})
        for trial in entry["ranked_trials"]:
            if trial["nct_id"] not in topic_qrels:
                continue
            lbl = derive_fn(trial["eligibility_summary"])
            dist[lbl] = dist.get(lbl, 0) + 1
    return dist


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="data/predictions/full_pipeline_2021.json")
    parser.add_argument("--run", default="data/runs/full_pipeline_2021.txt")
    parser.add_argument("--qrels", default="data/trec/2021/qrels.txt")
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    run_path = Path(args.run)
    qrels_path = Path(args.qrels)

    if not predictions_path.exists():
        print(f"ERROR: predictions file not found: {predictions_path}", file=sys.stderr)
        sys.exit(1)

    predictions = json.loads(predictions_path.read_text())
    qrels = load_qrels(qrels_path)

    topic_ids_in_preds = [str(e["patient_id"]) for e in predictions]
    print(f"\nPredictions loaded: {len(predictions)} topic(s) — {topic_ids_in_preds}")

    # T1 / T3: from run file (scores unchanged by label derivation)
    if run_path.exists():
        run = load_run(run_path)
        # Restrict run to topics present in predictions
        run_filtered = {tid: run[tid] for tid in topic_ids_in_preds if tid in run}
        ret = compute_retrieval_metrics(run_filtered, qrels)
        t1 = ret["recall_20"]
        t3 = ret["ndcg_cut_10"]
        map_ = ret["map"]
    else:
        print(f"WARNING: run file not found ({run_path}), T1/T3 will be 0.0", file=sys.stderr)
        t1 = t3 = map_ = 0.0

    # T2: before and after
    t2_old, report_old = compute_t2(predictions, qrels, _old_derive_label)
    t2_new, report_new = compute_t2(predictions, qrels, _new_derive_label)

    comp_old = 0.20 * t1 + 0.30 * t2_old + 0.25 * t3
    comp_new = 0.20 * t1 + 0.30 * t2_new + 0.25 * t3

    def _delta(a: float, b: float) -> str:
        d = b - a
        return f"{d:+.4f}"

    print()
    print("=" * 62)
    print("  METRICS WITH NEW LABEL DERIVATION")
    print("=" * 62)
    print(f"  {'':20s}  {'Before':>10}  {'After':>10}  {'Delta':>10}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}")
    print(f"  {'T1 Recall@20':20s}  {t1:>10.4f}  {t1:>10.4f}  {'(unchanged)':>10}")
    print(f"  {'T2 Micro-F1':20s}  {t2_old:>10.4f}  {t2_new:>10.4f}  {_delta(t2_old, t2_new):>10}")
    print(f"  {'T3 NDCG@10':20s}  {t3:>10.4f}  {t3:>10.4f}  {'(unchanged)':>10}")
    print(f"  {'MAP':20s}  {map_:>10.4f}  {map_:>10.4f}  {'(unchanged)':>10}")
    print(f"  {'Composite':20s}  {comp_old:>10.4f}  {comp_new:>10.4f}  {_delta(comp_old, comp_new):>10}")
    print()

    labels = ["MET", "NOT_MET", "NEI"]

    def _row(lbl: str, r_old: dict, r_new: dict) -> str:
        def _fmt(r: dict, l: str) -> str:
            if l not in r:
                return "  P=n/a  R=n/a  F1=n/a"
            p, rv, f = r[l]["precision"], r[l]["recall"], r[l]["f1-score"]
            return f"  P={p:.3f}  R={rv:.3f}  F1={f:.3f}"
        return f"  {lbl:<9}  before:{_fmt(r_old, lbl)}  →  after:{_fmt(r_new, lbl)}"

    print("  Per-class T2:")
    for lbl in labels:
        print(_row(lbl, report_old, report_new))

    # Label distribution shift
    dist_old = label_distribution(predictions, qrels, _old_derive_label)
    dist_new = label_distribution(predictions, qrels, _new_derive_label)
    total = sum(dist_old.values())
    print()
    print("  Label distribution (judged trials only):")
    print(f"  {'Label':<10}  {'Before':>8}  {'After':>8}  {'Delta':>8}")
    for lbl in labels:
        o, n = dist_old[lbl], dist_new[lbl]
        print(f"  {lbl:<10}  {o:>8}  {n:>8}  {n-o:>+8}")
    print(f"  {'Total':<10}  {total:>8}  {total:>8}")
    print("=" * 62)


if __name__ == "__main__":
    main()
