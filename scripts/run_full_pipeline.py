#!/usr/bin/env python3
"""
Full clinical trial matching pipeline — TREC 2021 evaluation.

For each of 75 TREC 2021 topics:
  1. PatientNormalizer  → PatientProfile
  2. QueryBuilder       → ≤500 candidate trials (CT API, cached)
  3. BM25 rerank        → top-50 hard cap
  4. CriteriaParser     → ParsedCriteria per trial (cached)
  5. HardFilter         → eliminate deterministic exclusion violations
  6. EligibilityReasoner→ list[(type, CriterionVerdict)] per survivor
  7. Scorer             → float score
  8. Sort descending    → TREC run file + JSON predictions

Outputs
-------
  data/runs/full_pipeline_2021.txt        TREC run file
  data/predictions/full_pipeline_2021.json  structured predictions

Metrics
-------
  T2: Micro-F1 over MET / NOT_MET / NEI  (3-way, pure-logic label derivation)
  T3: NDCG@10  (pytrec_eval)

Usage
-----
  python scripts/run_full_pipeline.py
  python scripts/run_full_pipeline.py --topic 1          # single topic debug
  python scripts/run_full_pipeline.py --topics-limit 5   # first N topics
  python scripts/run_full_pipeline.py --no-cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import diskcache
import pytrec_eval
from sklearn.metrics import classification_report, f1_score

from src.config import CACHE_DIR, DATA_DIR, TREC_2021_DIR, TREC_2022_DIR
from src.llm_client import get_usage
from src.matching.eligibility_reasoner import evaluate_trial
from src.matching.hard_filter import apply_hard_filter
from src.output.dossier_generator import generate_dossier
from src.output.nei_question_generator import generate_nei_question
from src.parsing.criteria_parser import parse_criteria
from src.parsing.patient_normalizer import normalize_patient
from src.ranking.scorer import score_trial
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_builder import build_queries_clinical, retrieve_candidates
from src.retrieval.semantic_retriever import SemanticRetriever
from src.retrieval.trec_index_retriever import TrecIndexRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SYSTEM_NAME = "clinical_agent"
CANDIDATE_CAP = 100
DOSSIER_TOP_K = 10  # generate dossiers only for top-K ranked trials per topic
RUN_PATH = DATA_DIR / "runs" / "full_pipeline_2021.txt"
PREDICTIONS_PATH = DATA_DIR / "predictions" / "full_pipeline_2021.json"
DOSSIERS_DIR = DATA_DIR / "dossiers"

# gpt-4o-mini pricing (USD per token)
_COST_PER_INPUT_TOKEN = 0.15 / 1_000_000
_COST_PER_OUTPUT_TOKEN = 0.60 / 1_000_000


def _llm_cost_usd() -> float:
    u = get_usage()
    return u["prompt_tokens"] * _COST_PER_INPUT_TOKEN + u["completion_tokens"] * _COST_PER_OUTPUT_TOKEN


# Grade → eligibility label (T2 mapping, confirmed)
_GRADE_TO_LABEL = {2: "MET", 1: "NOT_MET", 0: "NEI"}


# ---------------------------------------------------------------------------
# TREC file helpers
# ---------------------------------------------------------------------------

def load_topics(path: Path) -> dict[str, str]:
    """Parse topics.xml → {topic_id_str: patient_text}."""
    tree = ET.parse(str(path))
    topics: dict[str, str] = {}
    for el in tree.getroot().findall("topic"):
        num = (el.get("number") or "").strip()
        text = (el.text or "").strip()
        if num and text:
            topics[num] = text
    logger.info("Loaded %d topics from %s", len(topics), path.name)
    return topics


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Parse qrels.txt → {topic_id_str: {nct_id: grade}}."""
    qrels: dict[str, dict[str, int]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            tid, _, nct_id, grade = parts[0], parts[1], parts[2], int(parts[3])
            qrels.setdefault(tid, {})[nct_id] = grade
    logger.info(
        "Loaded qrels: %d topics, %d judgments",
        len(qrels), sum(len(v) for v in qrels.values()),
    )
    return qrels


def _write_topic_lines(fh, topic_id: str, scored: dict[str, float]) -> None:
    """Append TREC run lines for one topic to an open file handle."""
    ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
    for rank, (nct_id, score) in enumerate(ranked, 1):
        fh.write(f"{topic_id} Q0 {nct_id} {rank} {score:.6f} {SYSTEM_NAME}\n")


# ---------------------------------------------------------------------------
# Per-trial helper
# ---------------------------------------------------------------------------

def _eligibility_summary(typed_verdicts: list, eliminated: bool) -> dict:
    inc_met = sum(1 for t, v in typed_verdicts if t == "inclusion" and v.verdict == "MET")
    inc_not_met = sum(1 for t, v in typed_verdicts if t == "inclusion" and v.verdict == "NOT_MET")
    inc_nei = sum(1 for t, v in typed_verdicts if t == "inclusion" and v.verdict == "NEI")
    excl_viol = sum(1 for t, v in typed_verdicts if t == "exclusion" and v.verdict == "NOT_MET")
    if eliminated:
        excl_viol = max(excl_viol, 1)  # hard filter found a violation
    return {
        "inclusion_met": inc_met,
        "inclusion_not_met": inc_not_met,
        "inclusion_nei": inc_nei,
        "exclusion_violations": excl_viol,
    }


def _derive_label(summary: dict) -> str:
    """T2 label with NEI gates to fix over-confident MET labelling."""
    inc_met  = summary.get("inclusion_met", 0)
    inc_not_met = summary.get("inclusion_not_met", 0)
    inc_nei  = summary.get("inclusion_nei", 0)
    excl_viol = summary.get("exclusion_violations", 0)

    inc_total = max(inc_met + inc_not_met + inc_nei, 1)
    inc_ratio = inc_met / inc_total
    nei_ratio = inc_nei / inc_total

    if excl_viol > 0:
        return "NOT_MET"

    # Gate 1: majority uncertain → NEI
    if nei_ratio > 0.5:
        return "NEI"

    # Gate 2: weak inclusion match → NEI
    if inc_ratio < 0.3:
        return "NEI"

    if inc_met > 0:
        return "MET"

    return "NEI"


# ---------------------------------------------------------------------------
# Per-topic pipeline
# ---------------------------------------------------------------------------

def run_topic(
    topic_id: str,
    patient_text: str,
    use_cache: bool,
    qb_cache: diskcache.Cache,
    criteria_cache: diskcache.Cache,
    reasoner_cache: diskcache.Cache,
    retriever=None,              # any of: TrecIndexRetriever | SemanticRetriever | HybridRetriever | None
    max_workers: int = 5,
    use_clinical_planner: bool = False,
) -> tuple[dict[str, float], dict, int, int, int]:
    """
    Full pipeline for one topic.

    Returns (scores, predictions_entry, survivors, criteria_hits, criteria_total).
    scores:              {nct_id: float}  for TREC run file
    predictions_entry:   dict per spec
    survivors:           count of trials surviving hard filter
    criteria_hits/total: for cache-hit-rate diagnostic
    """
    # 1. Normalise patient
    profile = normalize_patient(patient_text, use_cache=use_cache)

    # 2. Retrieve candidates
    if retriever is not None:
        # Index-based path (TrecIndex / Semantic / Hybrid): query → NCT IDs → trial dicts
        nct_ids = retriever.query(patient_text, top_k=CANDIDATE_CAP)
        # SemanticRetriever.query() returns list[dict]; others return list[str]
        if nct_ids and isinstance(nct_ids[0], dict):
            nct_ids = [r["nct_id"] for r in nct_ids]
        top_trials = [
            t for nid in nct_ids
            if (t := retriever.get_trial(nid)) and t.get("nct_id")
        ]
        if not top_trials:
            logger.warning("Topic %s: no candidates from retriever.", topic_id)
            return {}, {"patient_id": topic_id, "ranked_trials": []}, 0, 0, 0
    elif use_clinical_planner:
        # Clinical query planner path (structured queries + MeSH, no TREC index)
        nct_ids = build_queries_clinical(
            patient_text=patient_text,
            profile=profile,
            max_candidates=CANDIDATE_CAP,
            use_cache=use_cache,
            cache=qb_cache,
        )
        top_trials = [
            t for nid in nct_ids
            if (t := qb_cache.get(f"trial:{nid}")) and isinstance(t, dict)
        ]
        # Fall back to CT API fetch for any NCT IDs not in local cache
        missing = [nid for nid in nct_ids if not qb_cache.get(f"trial:{nid}")]
        if missing:
            from src.retrieval.trec_index_retriever import TrecIndexRetriever as _TI
            _tc = diskcache.Cache(str(CACHE_DIR / "trial_data"))
            for nid in missing[:CANDIDATE_CAP]:
                td = _tc.get(nid)
                if td:
                    top_trials.append(td)
        top_trials = top_trials[:CANDIDATE_CAP]
        if not top_trials:
            logger.warning("Topic %s: no candidates from clinical planner.", topic_id)
            return {}, {"patient_id": topic_id, "ranked_trials": []}, 0, 0, 0
    else:
        # Original CT API path
        conditions = profile.conditions or [patient_text.split()[0]]
        trials = retrieve_candidates(
            conditions, max_total=500, use_cache=use_cache, cache=qb_cache
        )
        if not trials:
            logger.warning("Topic %s: no candidates retrieved.", topic_id)
            return {}, {"patient_id": topic_id, "ranked_trials": []}, 0, 0, 0
        try:
            bm25 = BM25Retriever(trials)
            top_trials = bm25.query(patient_text, top_k=CANDIDATE_CAP)
        except ValueError:
            top_trials = trials[:CANDIDATE_CAP]

    # 4-7. Per-trial pipeline — parallel with ThreadPoolExecutor
    scored: list[dict] = []
    survivors = 0
    criteria_hits = 0
    criteria_total = 0

    def _process_trial(trial: dict) -> dict:
        nct_id = trial.get("nct_id", "")
        if not nct_id:
            return {}
        try:
            crit_text = trial.get("eligibility_criteria", "")
            ck = hashlib.md5(crit_text.encode()).hexdigest() if crit_text else ""
            cache_hit = bool(ck and criteria_cache.get(ck) is not None)

            parsed = parse_criteria(crit_text, use_cache=use_cache, cache=criteria_cache)

            excl_det = [c for c in parsed.criteria
                        if c.type == "exclusion" and c.deterministic]
            survived = apply_hard_filter(profile, excl_det)

            if not survived:
                return {
                    "nct_id": nct_id, "score": 0.0,
                    "summary": _eligibility_summary([], eliminated=True),
                    "title": trial.get("title", ""),
                    "_cache_hit": cache_hit, "_survived": False,
                }

            typed_verdicts = evaluate_trial(
                profile, parsed.criteria,
                nct_id=nct_id,
                use_cache=use_cache,
                cache=reasoner_cache,
            )

            phase_list = trial.get("phases") or []
            metadata = {
                "phase": phase_list[0] if phase_list else "",
                "status": trial.get("status", ""),
            }
            score = score_trial(typed_verdicts, metadata)
            summary = _eligibility_summary(typed_verdicts, eliminated=False)
            enriched = [
                (ctype, c.text, v)
                for c, (ctype, v) in zip(parsed.criteria, typed_verdicts)
            ]
            return {
                "nct_id": nct_id, "score": score, "summary": summary,
                "title": trial.get("title", ""),
                "trial_meta": trial, "enriched_verdicts": enriched,
                "_cache_hit": cache_hit, "_survived": True,
            }
        except Exception as exc:
            logger.warning("Trial %s failed (topic %s): %s", nct_id, topic_id, exc)
            return {
                "nct_id": nct_id, "score": 0.0,
                "summary": _eligibility_summary([], eliminated=False),
                "title": trial.get("title", ""),
                "_cache_hit": False, "_survived": False,
            }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_trial, trial): trial for trial in top_trials}
        for future in as_completed(futures):
            result = future.result()
            if not result:
                continue
            scored.append(result)
            criteria_total += 1
            if result.get("_cache_hit"):
                criteria_hits += 1
            if result.get("_survived"):
                survivors += 1

    # 8. Sort descending
    scored.sort(key=lambda x: x["score"], reverse=True)

    scores = {t["nct_id"]: t["score"] for t in scored}

    # 9. Dossier generation for top-K trials (NEI questions + structured dossier)
    dossier_dir = DOSSIERS_DIR / topic_id
    dossier_dir.mkdir(parents=True, exist_ok=True)

    ranked_trials_json = []
    for rank, t in enumerate(scored, 1):
        _meta = t.get("trial_meta") or {}
        _phases = _meta.get("phases") or []
        entry: dict = {
            "rank": rank,
            "nct_id": t["nct_id"],
            "score": round(t["score"], 6),
            "title": t["title"],
            "phase": _phases[0] if _phases else _meta.get("phase", ""),
            "status": _meta.get("status", ""),
            "eligibility_summary": t["summary"],
        }

        enriched = t.get("enriched_verdicts")
        if rank <= DOSSIER_TOP_K and enriched:
            try:
                # Generate NEI questions for all NEI verdicts in this trial
                nei_questions: dict[str, str] = {}
                for ctype, ctext, v in enriched:
                    if v.verdict == "NEI":
                        nei_questions[ctext] = generate_nei_question(
                            verdict=v,
                            criterion_text=ctext,
                            trial_title=t["title"],
                            profile=profile,
                            use_cache=use_cache,
                        )

                dossier = generate_dossier(
                    profile=profile,
                    verdicts=enriched,
                    trial_metadata=t.get("trial_meta", {}),
                    nei_questions=nei_questions,
                )

                dossier_path = dossier_dir / f"{t['nct_id']}.json"
                dossier_path.write_text(dossier.model_dump_json(indent=2))
                entry["dossier_path"] = str(dossier_path)
            except Exception as exc:
                logger.warning(
                    "Dossier failed for %s (topic %s): %s", t["nct_id"], topic_id, exc
                )

        ranked_trials_json.append(entry)

    predictions_entry = {"patient_id": topic_id, "ranked_trials": ranked_trials_json}
    return scores, predictions_entry, survivors, criteria_hits, criteria_total


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_t2(
    predictions: list[dict],
    qrels: dict[str, dict[str, int]],
) -> tuple[float, dict]:
    """
    Compute T2 Micro-F1 over (topic, nct_id) pairs in both run and qrels.

    Label derivation (pure logic, no threshold):
      exclusion_violations > 0  → NOT_MET
      inclusion_met > 0         → MET
      else                      → NEI
    """
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
            y_pred.append(_derive_label(trial["eligibility_summary"]))

    if not y_true:
        logger.warning("No overlapping (topic, nct_id) pairs for T2 computation.")
        return 0.0, {}

    micro_f1 = f1_score(y_true, y_pred, average="micro",
                        labels=["MET", "NOT_MET", "NEI"], zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=["MET", "NOT_MET", "NEI"], zero_division=0,
        output_dict=True,
    )
    return micro_f1, report


def compute_retrieval_metrics(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
) -> dict[str, float]:
    """Compute T1 Recall@20, T3 NDCG@10, and MAP via pytrec_eval."""
    eval_qrels = {tid: qrels[tid] for tid in run if tid in qrels}
    if not eval_qrels:
        return {"ndcg_cut_10": 0.0, "recall_20": 0.0, "map": 0.0}
    evaluator = pytrec_eval.RelevanceEvaluator(
        eval_qrels, {"ndcg_cut_10", "recall_20", "map"}
    )
    per_topic = evaluator.evaluate(run)
    return {
        metric: statistics.mean(m[metric] for m in per_topic.values())
        for metric in ("ndcg_cut_10", "recall_20", "map")
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_diagnostics(
    predictions: list[dict],
    qrels: dict[str, dict[str, int]],
    topics: dict[str, str],
    avg_survivors: float,
    criteria_hits: int,
    criteria_total: int,
) -> None:
    print("\n" + "=" * 60)
    print("  DIAGNOSTICS")
    print("=" * 60)
    print(f"  Avg trials surviving hard filter per topic : {avg_survivors:.1f} / {CANDIDATE_CAP}")
    hit_pct = 100 * criteria_hits / criteria_total if criteria_total else 0
    print(f"  CriteriaParser cache hit rate              : "
          f"{criteria_hits}/{criteria_total} ({hit_pct:.0f}%)")
    print()

    # 3 topics where top-1 trial is wrong (qrel grade < 2)
    wrong: list[dict] = []
    for entry in predictions:
        tid = str(entry["patient_id"])
        topic_qrels = qrels.get(tid, {})
        ranked = entry.get("ranked_trials", [])
        if not ranked:
            continue
        top1 = ranked[0]
        grade = topic_qrels.get(top1["nct_id"], -1)
        if grade < 2:
            wrong.append({
                "topic_id": tid,
                "patient_snippet": topics.get(tid, "")[:300],
                "top1_nct": top1["nct_id"],
                "top1_title": top1.get("title", "N/A"),
                "top1_grade": grade,
                "top1_score": top1["score"],
                "summary": top1["eligibility_summary"],
            })
        if len(wrong) >= 3:
            break

    if wrong:
        print(f"  3 topics where top-1 is wrong (grade < 2):")
        print("-" * 60)
        for w in wrong:
            print(f"\n  Topic {w['topic_id']} — top-1 grade={w['top1_grade']}, "
                  f"score={w['top1_score']:.4f}")
            print(f"  Patient: {w['patient_snippet'][:200]}...")
            print(f"  Top-1:   {w['top1_nct']} | {w['top1_title'][:70]}")
            s = w["summary"]
            print(f"  Summary: inc_met={s['inclusion_met']} "
                  f"inc_not_met={s['inclusion_not_met']} "
                  f"inc_nei={s['inclusion_nei']} "
                  f"excl_viol={s['exclusion_violations']}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Full pipeline eval — TREC Clinical Trials")
    parser.add_argument("--year", type=int, choices=[2021, 2022], default=2021,
                        help="TREC year to evaluate (default: 2021)")
    parser.add_argument("--topic", type=int, default=None,
                        help="Run a single topic ID (debug mode)")
    parser.add_argument("--topics-limit", type=int, default=None,
                        help="Run only the first N topics")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable all caching (re-fetches everything)")
    parser.add_argument("--budget", type=float, default=1.75,
                        help="Hard stop when estimated LLM cost exceeds this USD amount (default 1.75)")
    parser.add_argument("--use-trec-index", action="store_true",
                        help="(deprecated) Alias for --retrieval-mode trec-index")
    parser.add_argument("--trec-index-path", default="data/cache/bm25_trec2021_index.pkl",
                        help="Path to saved TREC BM25 index")
    parser.add_argument("--retrieval-mode",
                        choices=["api", "trec-index", "semantic", "hybrid", "clinical"],
                        default="hybrid",
                        help="Retrieval strategy (default: hybrid)")
    parser.add_argument("--max-workers", type=int, default=5,
                        help="ThreadPoolExecutor workers for per-trial processing (default: 5)")
    parser.add_argument("--candidate-cap", type=int, default=None,
                        help="Override CANDIDATE_CAP (default: use module-level constant)")
    args = parser.parse_args()

    if args.candidate_cap is not None:
        global CANDIDATE_CAP
        CANDIDATE_CAP = args.candidate_cap

    use_cache = not args.no_cache
    budget_usd = args.budget

    # Year-specific paths
    trec_dir = TREC_2021_DIR if args.year == 2021 else TREC_2022_DIR
    global RUN_PATH, PREDICTIONS_PATH, DOSSIERS_DIR
    RUN_PATH = DATA_DIR / "runs" / f"full_pipeline_{args.year}.txt"
    PREDICTIONS_PATH = DATA_DIR / "predictions" / f"full_pipeline_{args.year}.json"
    DOSSIERS_DIR = DATA_DIR / "dossiers" / str(args.year)

    topics_path = trec_dir / "topics.xml"
    qrels_path = trec_dir / "qrels.txt"
    for p in (topics_path, qrels_path):
        if not p.exists():
            logger.error("Required file not found: %s", p)
            sys.exit(1)

    topics = load_topics(topics_path)
    qrels = load_qrels(qrels_path)

    # Filter to requested topics
    if args.topic is not None:
        key = str(args.topic)
        if key not in topics:
            logger.error("Topic %d not found.", args.topic)
            sys.exit(1)
        topics = {key: topics[key]}
    elif args.topics_limit:
        topics = dict(list(topics.items())[: args.topics_limit])

    # Retriever setup
    mode = args.retrieval_mode
    if args.use_trec_index and mode == "hybrid":
        mode = "trec-index"  # honour legacy flag

    retriever = None
    if mode == "trec-index":
        index_path = Path(args.trec_index_path)
        if not index_path.exists():
            logger.error("TREC index not found at %s", index_path)
            sys.exit(1)
        retriever = TrecIndexRetriever().load_index(index_path, cache_dir=CACHE_DIR / "trial_data")
        logger.info("Retrieval mode: trec-index (%d trials)", len(retriever._nct_ids))
    elif mode in ("semantic", "hybrid"):
        emb_path = CACHE_DIR / "embeddings" / "trial_embeddings.npy"
        ids_path = CACHE_DIR / "embeddings" / "trial_nct_ids.pkl"
        if not emb_path.exists():
            logger.error("Embeddings not found at %s", emb_path)
            sys.exit(1)
        sem = SemanticRetriever(emb_path, ids_path, CACHE_DIR / "trial_data")
        if mode == "semantic":
            retriever = sem
            logger.info("Retrieval mode: semantic (BioBERT, 26k trials)")
        else:
            retriever = HybridRetriever(sem, args.trec_index_path)
            logger.info("Retrieval mode: hybrid (BioBERT + BM25, alpha=%.2f beta=%.2f)",
                        retriever.alpha, retriever.beta)
    elif mode == "clinical":
        logger.info("Retrieval mode: clinical (ClinicalQueryPlanner + MeSH)")
    else:
        logger.info("Retrieval mode: api (CT API + QueryBuilder)")

    use_clinical_planner = (mode == "clinical")

    # Shared caches
    qb_cache = diskcache.Cache(str(CACHE_DIR / "query_builder"))
    criteria_cache = diskcache.Cache(str(CACHE_DIR / "criteria_parser"))
    reasoner_cache = diskcache.Cache(str(CACHE_DIR / "eligibility_reasoner"))

    # Output setup
    RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOSSIERS_DIR.mkdir(parents=True, exist_ok=True)
    run_fh = open(RUN_PATH, "w")

    all_predictions: list[dict] = []
    full_run: dict[str, dict[str, float]] = {}

    total_survivors = 0
    total_criteria_hits = 0
    total_criteria_calls = 0
    failed_topics = 0
    t_start = time.time()

    topic_items = list(topics.items())
    logger.info("Starting full pipeline — %d topics, cap=%d", len(topic_items), CANDIDATE_CAP)

    for i, (topic_id, patient_text) in enumerate(topic_items, 1):
        try:
            scores, pred_entry, survivors, c_hits, c_total = run_topic(
                topic_id=topic_id,
                patient_text=patient_text,
                use_cache=use_cache,
                qb_cache=qb_cache,
                criteria_cache=criteria_cache,
                reasoner_cache=reasoner_cache,
                retriever=retriever,
                max_workers=args.max_workers,
                use_clinical_planner=use_clinical_planner,
            )
        except Exception as exc:
            logger.error("Topic %s FAILED: %s", topic_id, exc)
            scores, pred_entry, survivors, c_hits, c_total = {}, {
                "patient_id": topic_id, "ranked_trials": []
            }, 0, 0, 0
            failed_topics += 1

        full_run[topic_id] = scores
        all_predictions.append(pred_entry)
        total_survivors += survivors
        total_criteria_hits += c_hits
        total_criteria_calls += c_total

        # Write run lines incrementally and flush
        _write_topic_lines(run_fh, topic_id, scores)
        run_fh.flush()

        cost_so_far = _llm_cost_usd()
        if i % 10 == 0 or i == len(topic_items):
            top1_score = (pred_entry["ranked_trials"][0]["score"]
                          if pred_entry.get("ranked_trials") else 0.0)
            elapsed = time.time() - t_start
            logger.info(
                "Topic %d/%d (id=%s) — survivors=%d, top1_score=%.3f, "
                "elapsed=%.0fs, cost=$%.4f",
                i, len(topic_items), topic_id, survivors, top1_score,
                elapsed, cost_so_far,
            )

        if cost_so_far >= budget_usd:
            logger.warning(
                "Budget limit $%.2f reached after topic %s (cost=$%.4f). "
                "Stopping early — %d/%d topics completed.",
                budget_usd, topic_id, cost_so_far, i, len(topic_items),
            )
            break

    run_fh.close()

    # Write predictions JSON
    try:
        PREDICTIONS_PATH.write_text(json.dumps(all_predictions, indent=2))
        logger.info("Predictions written to %s", PREDICTIONS_PATH)
    except Exception as exc:
        logger.error("Failed to write predictions JSON: %s", exc)

    elapsed_total = time.time() - t_start

    # --- Metrics ---
    t2_micro_f1 = 0.0
    t2_report: dict = {}
    retrieval: dict[str, float] = {"ndcg_cut_10": 0.0, "recall_20": 0.0, "map": 0.0}

    try:
        t2_micro_f1, t2_report = compute_t2(all_predictions, qrels)
    except Exception as exc:
        logger.error("T2 computation failed: %s", exc)

    try:
        retrieval = compute_retrieval_metrics(full_run, qrels)
    except Exception as exc:
        logger.error("Retrieval metrics computation failed: %s", exc)

    t1_recall20 = retrieval["recall_20"]
    t3_ndcg10 = retrieval["ndcg_cut_10"]
    map_score = retrieval["map"]
    composite = 0.20 * t1_recall20 + 0.30 * t2_micro_f1 + 0.25 * t3_ndcg10

    # --- Results banner ---
    print("\n" + "=" * 60)
    print(f"  FULL PIPELINE — TREC {args.year}")
    print("=" * 60)
    print(f"  Topics evaluated    : {len(all_predictions)}  (failed: {failed_topics})")
    print(f"  Elapsed             : {elapsed_total:.0f}s")
    print(f"  LLM cost (est.)     : ${_llm_cost_usd():.4f}")
    print("-" * 60)
    print(f"  T1 Recall@20        : {t1_recall20:.4f}")
    print(f"  T2 Micro-F1         : {t2_micro_f1:.4f}")
    print(f"  T3 NDCG@10          : {t3_ndcg10:.4f}")
    print(f"  MAP                 : {map_score:.4f}")
    print(f"  Composite           : {composite:.4f}  (0.20*T1 + 0.30*T2 + 0.25*T3)")
    print("-" * 60)
    if t2_report:
        for label in ("MET", "NOT_MET", "NEI"):
            m = t2_report.get(label, {})
            print(f"  {label:<10}  P={m.get('precision', 0):.3f}  "
                  f"R={m.get('recall', 0):.3f}  F1={m.get('f1-score', 0):.3f}  "
                  f"n={m.get('support', 0)}")
    print("=" * 60)
    print(f"  Run file : {RUN_PATH}")
    print(f"  JSON     : {PREDICTIONS_PATH}")
    print(f"  Dossiers : {DOSSIERS_DIR}/<topic_id>/<nct_id>.json")

    # --- Diagnostics ---
    avg_survivors = total_survivors / max(len(topic_items), 1)
    print_diagnostics(
        predictions=all_predictions,
        qrels=qrels,
        topics=topics,
        avg_survivors=avg_survivors,
        criteria_hits=total_criteria_hits,
        criteria_total=total_criteria_calls,
    )


if __name__ == "__main__":
    main()
