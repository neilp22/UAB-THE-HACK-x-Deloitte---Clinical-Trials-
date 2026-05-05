"""
BM25 Baseline — TREC Clinical Trials 2021/2022.

Pipeline per topic:
  1. Load TREC topic text
  2. Query ClinicalTrials.gov API (diskcache result per topic)
  3. BM25-rank retrieved trials against patient text
  4. Write TREC run file
  5. Evaluate with pytrec_eval (Recall@20, NDCG@10, MAP)

Usage:
    python scripts/run_bm25_baseline.py --year 2021
    python scripts/run_bm25_baseline.py --year 2021 --max-results 500 --no-cache
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

# Ensure repo root is on path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import diskcache
import lxml.etree as ET
import pytrec_eval
from tqdm import tqdm

from src.config import CACHE_DIR, OUTPUT_DIR, TREC_2021_DIR, TREC_2022_DIR
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.ct_client import search_trials

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SYSTEM_NAME = "bm25_baseline"


# ---------------------------------------------------------------------------
# TREC helpers
# ---------------------------------------------------------------------------

def load_topics(topics_xml: Path) -> dict[str, str]:
    """Parse TREC topics XML → {topic_id_str: text}."""
    tree = ET.parse(str(topics_xml))
    topics: dict[str, str] = {}
    for topic_el in tree.findall(".//topic"):
        num = topic_el.get("number", "").strip()
        text = (topic_el.text or "").strip()
        if num and text:
            topics[num] = text
    logger.info("Loaded %d topics from %s", len(topics), topics_xml.name)
    return topics


def load_qrels(qrels_path: Path) -> dict[str, dict[str, int]]:
    """
    Parse TREC qrels file → {topic_id_str: {nct_id: grade}}.

    Qrel grade meanings:
        0 = not relevant
        1 = excluded (patient excluded by criteria — still relevant for retrieval)
        2 = eligible
    """
    qrels: dict[str, dict[str, int]] = {}
    with open(qrels_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            topic_id, _, doc_id, grade = parts[0], parts[1], parts[2], int(parts[3])
            qrels.setdefault(topic_id, {})[doc_id] = grade
    logger.info(
        "Loaded qrels: %d topics, %d total judgments",
        len(qrels),
        sum(len(v) for v in qrels.values()),
    )
    return qrels


def write_run_file(run: dict[str, dict[str, float]], output_path: Path) -> None:
    """Write TREC-format run file: topic Q0 docid rank score sysname."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for topic_id in sorted(run, key=lambda x: int(x)):
            scored = sorted(run[topic_id].items(), key=lambda kv: kv[1], reverse=True)
            for rank, (nct_id, score) in enumerate(scored, start=1):
                f.write(f"{topic_id} Q0 {nct_id} {rank} {score:.6f} {SYSTEM_NAME}\n")
    logger.info("Run file written to %s (%d topics)", output_path, len(run))


def evaluate(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
) -> dict[str, float]:
    """
    Evaluate run against qrels with pytrec_eval.

    Relevance threshold: grade >= 1 (both excluded and eligible count as relevant
    for retrieval metrics; NDCG uses graded relevance).
    """
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels,
        {"recall.20", "ndcg_cut.10", "map", "P.10"},
    )
    per_topic = evaluator.evaluate(run)

    # Macro-average over topics
    agg: dict[str, list[float]] = {}
    for metrics in per_topic.values():
        for k, v in metrics.items():
            agg.setdefault(k, []).append(v)

    return {k: sum(vs) / len(vs) for k, vs in agg.items()}


# ---------------------------------------------------------------------------
# Query extraction
# ---------------------------------------------------------------------------

# Words that appear frequently in MIMIC notes but are not useful CT query terms
_CLINICAL_STOP_WORDS = frozenset({
    # Demographic / generic
    "patient", "woman", "female", "male", "gender", "years", "year",
    "month", "months", "weeks", "adult", "elderly", "person",
    # Presentation verbs
    "presented", "presents", "presenting", "comes", "referred", "admitted",
    "showed", "revealed", "noted", "found", "reported", "diagnosed",
    "evaluated", "reviewed", "discharged", "transferred",
    # Generic clinical nouns (not conditions)
    "history", "hospital", "medical", "physical", "clinic", "emergency",
    "exam", "underwent", "received", "treated", "discharge", "follow",
    "given", "taken", "started", "continued", "developed", "denied",
    "prior", "bilateral", "significant", "consistent", "evidence",
    "secondary", "symptoms", "treatment", "therapy", "medications",
    "diagnosis", "clinical", "evaluation", "management", "associated",
    "following", "approximately", "including", "without", "second",
    "after", "before", "during", "since", "while", "acute", "chronic",
    "stable", "severe", "moderate", "initial", "current", "known",
    "right", "left", "lower", "upper", "small", "large", "normal",
    "positive", "negative", "recent", "previous", "multiple", "single",
    "onset", "course", "review", "systems", "family", "social",
    "surgical", "procedure", "imaging", "laboratory", "result", "level",
    "further", "first", "other", "which", "where", "there", "these",
    "those", "their", "workup", "status", "disease", "disorder",
    "condition", "complication", "complaint", "concern", "issues",
    "residual", "deficits", "notable", "abnormal", "elevated", "decreased",
    "assessment", "obesity", "notable", "notable",
})


def extract_query(patient_text: str, n_terms: int = 2) -> str:
    """
    Extract n_terms key medical condition terms for CT API query.cond.

    CT API rules (empirical):
    - query.cond with 2-3 terms → best results (AND logic)
    - query.term with >4 words → HTTP 400 'too complicated query'
    - Best strategy: query.cond with 2 specific medical terms

    Approach:
    1. Look for "history of X" / "diagnosed with X" patterns first
    2. Fall back to longest medical-looking words from first sentences

    Day 2 replaces this with proper MeSH-mapped condition extraction.
    """
    text = patient_text

    # Remove MIMIC de-identification markers [**...**]
    text = re.sub(r"\[\*\*[^\]]*\*\*\]", " ", text)
    # Remove all non-alphanumeric except spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    # Collapse whitespace
    text = " ".join(text.split())

    # --- Strategy 1: extract phrase after "history of" / "diagnosed with" ---
    # These patterns reliably point to the primary condition
    pattern_match = re.search(
        r"(?:history of|diagnosed with|diagnosis of|found to have|known)\s+([a-zA-Z][a-zA-Z0-9\s]{3,50}?)(?:\s+(?:complicated|with|and|who|which|that|the)|$)",
        text,
        re.IGNORECASE,
    )
    if pattern_match:
        phrase = pattern_match.group(1).strip()
        # Take first 2 substantive words (filter short function words)
        words = [w for w in phrase.split() if len(w) >= 4][:2]
        query = " ".join(words)
        if len(query) >= 5:
            return query

    # --- Strategy 2: longest medical-looking words from first 300 chars ---
    snippet = text[:300]
    words = snippet.split()
    key_terms: list[str] = []
    seen: set[str] = set()
    for word in words:
        w = word.lower()
        if (
            len(word) >= 6
            and w not in _CLINICAL_STOP_WORDS
            and not word.isdigit()
            and w not in seen
        ):
            key_terms.append(word)
            seen.add(w)
        if len(key_terms) >= n_terms:
            break

    return " ".join(key_terms) if key_terms else snippet[:50]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_baseline(
    year: int,
    max_results: int = 200,
    use_cache: bool = True,
    topics_limit: int | None = None,
) -> None:
    trec_dir = TREC_2021_DIR if year == 2021 else TREC_2022_DIR

    topics_path = trec_dir / "topics.xml"
    qrels_path = trec_dir / "qrels.txt"

    if not topics_path.exists():
        logger.error("Topics file not found: %s", topics_path)
        sys.exit(1)
    if not qrels_path.exists():
        logger.error("Qrels file not found: %s", qrels_path)
        sys.exit(1)

    topics = load_topics(topics_path)
    qrels = load_qrels(qrels_path)

    if topics_limit:
        topics = dict(list(topics.items())[:topics_limit])

    cache = diskcache.Cache(str(CACHE_DIR / f"ct_api_{year}"))
    run: dict[str, dict[str, float]] = {}

    logger.info("Starting BM25 baseline — %d topics, max_results=%d", len(topics), max_results)
    total_retrieved = 0
    t_start = time.time()

    for topic_id, patient_text in tqdm(topics.items(), desc=f"TREC {year} topics"):
        cache_key = f"topic_{topic_id}_n{max_results}"

        if use_cache and cache_key in cache:
            trials = cache[cache_key]
        else:
            query = extract_query(patient_text)
            logger.debug("Topic %s → query.cond: '%s'", topic_id, query)
            try:
                # Primary: query.cond with 2 terms (AND logic — may miss some trials)
                trials = search_trials(query_cond=query, max_results=max_results)

                # Fallback 1: if too specific, retry with first term only
                if not trials and " " in query:
                    first_term = query.split()[0]
                    logger.debug("Topic %s fallback → query.cond: '%s'", topic_id, first_term)
                    trials = search_trials(query_cond=first_term, max_results=max_results)

                # Fallback 2: if still nothing, use query.term with cleaned 2-word snippet
                if not trials:
                    short_term = " ".join(query.split()[:2])
                    logger.debug("Topic %s fallback2 → query.term: '%s'", topic_id, short_term)
                    trials = search_trials(query_term=short_term, max_results=max_results)

                if use_cache:
                    cache[cache_key] = trials
            except Exception as exc:
                logger.error("CT API failed for topic %s: %s", topic_id, exc)
                trials = []

        total_retrieved += len(trials)

        if not trials:
            logger.warning("Topic %s: no trials retrieved.", topic_id)
            run[topic_id] = {}
            continue

        # BM25-rank all retrieved trials against the full patient text
        retriever = BM25Retriever(trials)
        ranked = retriever.query(patient_text, top_k=len(trials))

        run[topic_id] = {r["nct_id"]: r["bm25_score"] for r in ranked if r["nct_id"]}

    elapsed = time.time() - t_start
    logger.info(
        "Pipeline complete: %.1fs, avg %.1f trials/topic",
        elapsed,
        total_retrieved / max(len(topics), 1),
    )

    # Write run file
    run_path = OUTPUT_DIR / f"bm25_baseline_{year}.run"
    write_run_file(run, run_path)

    # Evaluate
    # Filter qrels to only topics we ran (in case topics_limit was set)
    eval_qrels = {tid: qrels[tid] for tid in run if tid in qrels}
    if not eval_qrels:
        logger.warning("No qrel-covered topics in run — skipping evaluation.")
        return

    metrics = evaluate(eval_qrels, run)

    print("\n" + "=" * 55)
    print(f"  BM25 Baseline — TREC {year}")
    print("=" * 55)
    print(f"  Topics evaluated : {len(eval_qrels)}")
    print(f"  Avg trials/topic : {total_retrieved / max(len(topics), 1):.0f}")
    print("-" * 55)
    for metric, value in sorted(metrics.items()):
        print(f"  {metric:<25} {value:.4f}")
    print("=" * 55)
    print(f"  Run file: {run_path}")

    # Composite score (weighted as per hackathon)
    r20 = metrics.get("recall_20", 0)
    ndcg10 = metrics.get("ndcg_cut_10", 0)
    composite = 0.20 * r20 + 0.25 * ndcg10
    print(f"\n  Composite (T1+T3 only): {composite:.4f}")
    print("  Note: T2/T4/T5 not measured by this baseline.")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BM25 baseline on TREC Clinical Trials.")
    parser.add_argument("--year", type=int, default=2021, choices=[2021, 2022])
    parser.add_argument("--max-results", type=int, default=200,
                        help="Max CT API results per topic (default: 200)")
    parser.add_argument("--topics-limit", type=int, default=None,
                        help="Run only the first N topics (for quick testing)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore CT API cache and re-fetch everything")
    args = parser.parse_args()

    run_baseline(
        year=args.year,
        max_results=args.max_results,
        use_cache=not args.no_cache,
        topics_limit=args.topics_limit,
    )
