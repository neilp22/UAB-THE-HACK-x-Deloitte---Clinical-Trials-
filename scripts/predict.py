#!/usr/bin/env python3
"""
Batch prediction script — non-interactive, JSON in / JSON out.

INPUT FORMAT (--input FILE):
  JSON array of patient objects:
    [
      {"id": "P001", "text": "Patient is a 45-year-old woman with NSCLC..."},
      {"id": "P002", "text": "67-year-old man with metastatic colorectal cancer..."}
    ]

  Also accepts plain text files (one patient per line, IDs auto-assigned P001, P002...):
    Patient is a 45-year-old woman with NSCLC...
    67-year-old man with metastatic colorectal cancer...

OUTPUT FORMAT (--output FILE, default: data/predictions/batch_predictions.json):
  [
    {
      "patient_id": "P001",
      "ranked_trials": [
        {
          "rank": 1,
          "nct_id": "NCT00123456",
          "score": 0.4823,
          "score_breakdown": {...},
          "score_explanation": "Ranked based on ...",
          "title": "Phase 3 Trial of ...",
          "phase": "PHASE3",
          "status": "RECRUITING",
          "eligibility_summary": {
            "inclusion_met": 4,
            "inclusion_not_met": 1,
            "inclusion_nei": 2,
            "exclusion_violations": 0
          },
          "dossier_path": "data/dossiers/batch/P001/NCT00123456.json"
        },
        ...
      ]
    },
    ...
  ]

USAGE:
  # Run on a JSON list of patients (recommended)
  python scripts/predict.py --input patients.json

  # Run on a plain-text file (one patient note per line)
  python scripts/predict.py --input patients.txt

  # Specify output path
  python scripts/predict.py --input patients.json --output results/predictions.json

  # Use high-recall retrieval (best quality)
  python scripts/predict.py --input patients.json --retrieval-mode high-recall

  # Limit candidates per patient (faster, cheaper)
  python scripts/predict.py --input patients.json --candidate-cap 100

  # Skip cache (force full re-evaluation)
  python scripts/predict.py --input patients.json --no-cache

  # Cap LLM spend
  python scripts/predict.py --input patients.json --budget 5.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import diskcache

from src.config import CACHE_DIR, DATA_DIR
from src.llm_client import get_usage, reset_usage
from src.matching.eligibility_reasoner import evaluate_trial
from src.matching.hard_filter import apply_hard_filter
from src.output.dossier_generator import generate_dossier
from src.output.nei_question_generator import generate_nei_question
from src.parsing.criteria_parser import parse_criteria
from src.parsing.patient_normalizer import normalize_patient
from src.ranking.scorer import explain_score_breakdown, score_trial_breakdown
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.query_builder import build_queries_clinical, build_queries_combined, retrieve_candidates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Pricing (gpt-4o-mini)
_COST_PER_INPUT_TOKEN = 0.15 / 1_000_000
_COST_PER_OUTPUT_TOKEN = 0.60 / 1_000_000

SYSTEM_NAME = "clinical_agent"
DOSSIERS_DIR = DATA_DIR / "dossiers" / "batch"
DOSSIER_TOP_K = 10


def _llm_cost_usd() -> float:
    u = get_usage()
    return u["prompt_tokens"] * _COST_PER_INPUT_TOKEN + u["completion_tokens"] * _COST_PER_OUTPUT_TOKEN


def load_patients(path: Path) -> list[dict]:
    """
    Load patient profiles from a JSON or plain-text file.

    JSON format: [{"id": "P001", "text": "..."}, ...]
    Text format: one patient note per line (IDs auto-assigned P001, P002, ...)
    """
    raw = path.read_text(encoding="utf-8").strip()

    if path.suffix.lower() == ".json" or raw.startswith("[") or raw.startswith("{"):
        data = json.loads(raw)
        if isinstance(data, dict):
            data = [data]
        patients = []
        for i, item in enumerate(data):
            if isinstance(item, str):
                patients.append({"id": f"P{i+1:03d}", "text": item})
            elif isinstance(item, dict):
                pid = str(item.get("id") or item.get("patient_id") or f"P{i+1:03d}")
                text = item.get("text") or item.get("patient_text") or item.get("note") or ""
                if not text:
                    logger.warning("Patient %s has no text field — skipping.", pid)
                    continue
                patients.append({"id": pid, "text": text})
        return patients

    # Plain text: one patient per line
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    return [{"id": f"P{i+1:03d}", "text": line} for i, line in enumerate(lines)]


def _eligibility_summary(typed_verdicts: list, eliminated: bool) -> dict:
    inc_met = sum(1 for t, v in typed_verdicts if t == "inclusion" and v.verdict == "MET")
    inc_not_met = sum(1 for t, v in typed_verdicts if t == "inclusion" and v.verdict == "NOT_MET")
    inc_nei = sum(1 for t, v in typed_verdicts if t == "inclusion" and v.verdict == "NEI")
    excl_viol = sum(1 for t, v in typed_verdicts if t == "exclusion" and v.verdict == "NOT_MET")
    if eliminated:
        excl_viol = max(excl_viol, 1)
    return {
        "inclusion_met": inc_met,
        "inclusion_not_met": inc_not_met,
        "inclusion_nei": inc_nei,
        "exclusion_violations": excl_viol,
    }


def run_patient(
    patient_id: str,
    patient_text: str,
    use_cache: bool,
    qb_cache: diskcache.Cache,
    criteria_cache: diskcache.Cache,
    reasoner_cache: diskcache.Cache,
    retrieval_mode: str,
    candidate_cap: int,
    max_workers: int,
    retriever=None,
) -> dict:
    """Full pipeline for one patient. Returns predictions_entry dict."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    use_clinical_planner = retrieval_mode == "clinical"
    use_combined = retrieval_mode == "combined"

    # Stage 1: Normalize patient
    profile = normalize_patient(patient_text, use_cache=use_cache)

    # Stage 2: Retrieve candidates
    if retriever is not None:
        from src.retrieval.high_recall_retriever import HighRecallRetriever
        if isinstance(retriever, HighRecallRetriever):
            nct_ids = retriever.retrieve(patient_text, profile=profile, top_k=candidate_cap)
        else:
            nct_ids = retriever.query(patient_text, top_k=candidate_cap)
        if nct_ids and isinstance(nct_ids[0], dict):
            nct_ids = [r["nct_id"] for r in nct_ids]
        top_trials = [
            t for nid in nct_ids
            if (t := retriever.get_trial(nid)) and t.get("nct_id")
        ]
        if not top_trials:
            logger.warning("Patient %s: no candidates from retriever.", patient_id)
            return {"patient_id": patient_id, "ranked_trials": []}
    elif use_combined or use_clinical_planner:
        if use_combined:
            nct_ids = build_queries_combined(
                patient_text=patient_text,
                profile=profile,
                max_candidates=candidate_cap,
                use_cache=use_cache,
                cache=qb_cache,
            )
        else:
            nct_ids = build_queries_clinical(
                patient_text=patient_text,
                profile=profile,
                max_candidates=candidate_cap,
                use_cache=use_cache,
                cache=qb_cache,
            )
        _tc = diskcache.Cache(str(CACHE_DIR / "trial_data"))
        top_trials = []
        for nid in nct_ids:
            t = qb_cache.get(f"trial:{nid}") or _tc.get(nid)
            if t and isinstance(t, dict):
                top_trials.append(t)
        top_trials = top_trials[:candidate_cap]
        if not top_trials:
            logger.warning("Patient %s: no candidates from %s.", patient_id, retrieval_mode)
            return {"patient_id": patient_id, "ranked_trials": []}
    else:
        # CT API path
        conditions = profile.conditions or [patient_text.split()[0]]
        trials = retrieve_candidates(
            conditions, max_total=500, use_cache=use_cache, cache=qb_cache
        )
        if not trials:
            logger.warning("Patient %s: no candidates retrieved.", patient_id)
            return {"patient_id": patient_id, "ranked_trials": []}
        try:
            bm25 = BM25Retriever(trials)
            bm25_query = " ".join(profile.conditions) if profile.conditions else patient_text
            top_trials = bm25.query(bm25_query, top_k=candidate_cap)
        except ValueError:
            top_trials = trials[:candidate_cap]

    # Stages 3-6: per-trial pipeline (parallel)
    scored: list[dict] = []

    def _process_trial(trial: dict) -> dict:
        nct_id = trial.get("nct_id", "")
        if not nct_id:
            return {}
        score = 0.0
        score_breakdown: dict = {}
        score_explanation = ""
        try:
            crit_text = trial.get("eligibility_criteria", "")
            ck = hashlib.md5(crit_text.encode()).hexdigest() if crit_text else ""
            parsed = parse_criteria(crit_text, use_cache=use_cache, cache=criteria_cache)

            excl_det = [c for c in parsed.criteria if c.type == "exclusion" and c.deterministic]
            survived = apply_hard_filter(profile, excl_det)

            if not survived:
                return {
                    "nct_id": nct_id,
                    "score": -0.5,
                    "score_breakdown": {"hard_filter_eliminated": True, "final_score": -0.5},
                    "score_explanation": "Eliminated by deterministic exclusion criterion.",
                    "summary": _eligibility_summary([], eliminated=True),
                    "title": trial.get("title", ""),
                    "_survived": False,
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
            score_breakdown = score_trial_breakdown(typed_verdicts, metadata)
            score = score_breakdown["final_score"]
            score_explanation = explain_score_breakdown(score_breakdown)
            summary = _eligibility_summary(typed_verdicts, eliminated=False)
            enriched = [
                (ctype, c.text, v)
                for c, (ctype, v) in zip(parsed.criteria, typed_verdicts)
            ]
            return {
                "nct_id": nct_id,
                "score": score,
                "score_breakdown": score_breakdown,
                "score_explanation": score_explanation,
                "summary": summary,
                "title": trial.get("title", ""),
                "trial_meta": trial,
                "enriched_verdicts": enriched,
                "_survived": True,
            }
        except Exception as exc:
            logger.warning("Trial %s failed (patient %s): %s", nct_id, patient_id, exc)
            return {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_trial, t): t for t in top_trials}
        for future in as_completed(futures):
            result = future.result()
            if result:
                scored.append(result)

    scored.sort(key=lambda x: x["score"], reverse=True)

    # Stage 7: dossiers + NEI questions for top-K
    dossier_dir = DOSSIERS_DIR / patient_id
    dossier_dir.mkdir(parents=True, exist_ok=True)

    ranked_trials_json = []
    for rank, t in enumerate(scored, 1):
        _meta = t.get("trial_meta") or {}
        _phases = _meta.get("phases") or []
        entry: dict = {
            "rank": rank,
            "nct_id": t["nct_id"],
            "score": round(t["score"], 6),
            "score_breakdown": t.get("score_breakdown", {}),
            "score_explanation": t.get("score_explanation", ""),
            "title": t["title"],
            "phase": _phases[0] if _phases else _meta.get("phase", ""),
            "status": _meta.get("status", ""),
            "eligibility_summary": t["summary"],
        }

        enriched = t.get("enriched_verdicts")
        if rank <= DOSSIER_TOP_K and enriched:
            try:
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
                logger.warning("Dossier failed for %s (patient %s): %s", t["nct_id"], patient_id, exc)

        ranked_trials_json.append(entry)

    return {"patient_id": patient_id, "ranked_trials": ranked_trials_json}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch clinical trial matching — JSON in, JSON out.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", required=True, metavar="FILE",
        help="Path to input file: JSON array [{id, text}, ...] or plain text (one patient per line)",
    )
    parser.add_argument(
        "--output", default=None, metavar="FILE",
        help="Path to output JSON (default: data/predictions/batch_predictions.json)",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["api", "trec-index", "semantic", "hybrid", "clinical", "combined", "high-recall"],
        default="high-recall",
        help="Retrieval strategy (default: high-recall)",
    )
    parser.add_argument(
        "--candidate-cap", type=int, default=250,
        help="Max trials retrieved per patient (default: 250)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=5,
        help="ThreadPoolExecutor workers for per-trial LLM calls (default: 5)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable all caching (re-fetches everything, much slower)",
    )
    parser.add_argument(
        "--budget", type=float, default=10.0,
        help="Stop if estimated LLM cost exceeds this USD amount (default: 10.0)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    output_path = Path(args.output) if args.output else DATA_DIR / "predictions" / "batch_predictions.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    use_cache = not args.no_cache
    candidate_cap = args.candidate_cap

    # Load patients
    try:
        patients = load_patients(input_path)
    except Exception as exc:
        logger.error("Failed to load patients from %s: %s", input_path, exc)
        sys.exit(1)

    if not patients:
        logger.error("No patients found in %s", input_path)
        sys.exit(1)

    logger.info("Loaded %d patient(s) from %s", len(patients), input_path)

    # Set up retriever
    retriever = None
    mode = args.retrieval_mode

    if mode in ("trec-index", "semantic", "hybrid", "high-recall"):
        from src.config import CACHE_DIR
        bm25_index_path = CACHE_DIR / "bm25_trec2021_index.pkl"
        emb_path = CACHE_DIR / "embeddings" / "trial_embeddings.npy"
        ids_path = CACHE_DIR / "embeddings" / "trial_nct_ids.pkl"

        if mode == "trec-index":
            if not bm25_index_path.exists():
                logger.error(
                    "TREC BM25 index not found at %s — run: python scripts/build_trec_index.py",
                    bm25_index_path,
                )
                sys.exit(1)
            from src.retrieval.trec_index_retriever import TrecIndexRetriever
            retriever = TrecIndexRetriever().load_index(bm25_index_path, cache_dir=CACHE_DIR / "trial_data")
            logger.info("Retrieval mode: trec-index (%d trials)", len(retriever._nct_ids))

        elif mode == "semantic":
            if not emb_path.exists():
                logger.error("Embeddings not found at %s", emb_path)
                sys.exit(1)
            from src.retrieval.semantic_retriever import SemanticRetriever
            retriever = SemanticRetriever(emb_path, ids_path, CACHE_DIR / "trial_data")
            logger.info("Retrieval mode: semantic")

        elif mode == "hybrid":
            if not emb_path.exists() or not bm25_index_path.exists():
                logger.error("Hybrid mode requires both BM25 index and embeddings.")
                sys.exit(1)
            from src.retrieval.semantic_retriever import SemanticRetriever
            from src.retrieval.hybrid_retriever import HybridRetriever
            sem = SemanticRetriever(emb_path, ids_path, CACHE_DIR / "trial_data")
            retriever = HybridRetriever(sem, str(bm25_index_path))
            logger.info("Retrieval mode: hybrid (BM25 α=0.50 + BioBERT β=0.50)")

        elif mode == "high-recall":
            field_index = CACHE_DIR / "field_bm25_index.pkl"
            from src.retrieval.high_recall_retriever import HighRecallRetriever
            retriever = HighRecallRetriever(
                bm25_index_path=str(bm25_index_path),
                field_index_path=str(field_index),
                embeddings_path=str(emb_path),
                nct_ids_pkl=str(ids_path),
                trial_cache_dir=str(CACHE_DIR / "trial_data"),
            )
            logger.info("Retrieval mode: high-recall (multi-query × field-BM25 × RRF)")

    elif mode == "combined":
        logger.info("Retrieval mode: combined (ClinicalQueryPlanner ∪ MeSH)")
    elif mode == "clinical":
        logger.info("Retrieval mode: clinical (ClinicalQueryPlanner + MeSH)")
    else:
        logger.info("Retrieval mode: api (CT API + BM25 rerank)")

    # Shared caches
    qb_cache = diskcache.Cache(str(CACHE_DIR / "query_builder"))
    criteria_cache = diskcache.Cache(str(CACHE_DIR / "criteria_parser"))
    reasoner_cache = diskcache.Cache(str(CACHE_DIR / "eligibility_reasoner"))

    DOSSIERS_DIR.mkdir(parents=True, exist_ok=True)

    # Process patients
    all_predictions: list[dict] = []
    t_start = time.time()
    reset_usage()

    for i, patient in enumerate(patients, 1):
        pid = patient["id"]
        text = patient["text"]
        logger.info("Processing patient %d/%d: %s", i, len(patients), pid)

        try:
            pred_entry = run_patient(
                patient_id=pid,
                patient_text=text,
                use_cache=use_cache,
                qb_cache=qb_cache,
                criteria_cache=criteria_cache,
                reasoner_cache=reasoner_cache,
                retrieval_mode=mode,
                candidate_cap=candidate_cap,
                max_workers=args.max_workers,
                retriever=retriever,
            )
        except Exception as exc:
            logger.error("Patient %s FAILED: %s", pid, exc)
            pred_entry = {"patient_id": pid, "ranked_trials": []}

        all_predictions.append(pred_entry)

        n_ranked = len(pred_entry.get("ranked_trials", []))
        cost = _llm_cost_usd()
        elapsed = time.time() - t_start
        logger.info(
            "  → %d trials ranked | elapsed %.0fs | cost $%.4f",
            n_ranked, elapsed, cost,
        )

        if cost >= args.budget:
            logger.warning(
                "Budget $%.2f reached after patient %s (cost=$%.4f). Stopping.",
                args.budget, pid, cost,
            )
            break

    # Write output
    output_path.write_text(json.dumps(all_predictions, indent=2))

    elapsed_total = time.time() - t_start
    print("\n" + "=" * 60)
    print("  BATCH PREDICTION COMPLETE")
    print("=" * 60)
    print(f"  Patients processed : {len(all_predictions)}/{len(patients)}")
    print(f"  Retrieval mode     : {mode}")
    print(f"  Candidate cap      : {candidate_cap}")
    print(f"  Elapsed            : {elapsed_total:.0f}s")
    print(f"  LLM cost (est.)    : ${_llm_cost_usd():.4f}")
    print(f"  Output             : {output_path}")
    print(f"  Dossiers           : {DOSSIERS_DIR}/<patient_id>/<nct_id>.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
