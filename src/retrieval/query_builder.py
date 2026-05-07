"""
Multi-query builder for ClinicalTrials.gov retrieval.

Given a list of medical conditions, generates multiple CT API query variants
(original terms + MeSH synonyms) and merges results to maximise recall.

CT API hard limits (empirical, Day 1):
- query.cond: max 2 terms (AND logic) — 1 term is safest / broadest
- query.term: >4 words → HTTP 400 "too complicated query"
- Use requests (not httpx) — Cloudflare WAF blocks httpx via TLS fingerprint
"""
from __future__ import annotations

import logging
import time

import diskcache
import requests

from src.config import CACHE_DIR, CT_POLITE_DELAY
from src.parsing.patient_normalizer import PatientProfile
from src.retrieval.ct_client import search_trials

logger = logging.getLogger(__name__)

_MESH_API = "https://id.nlm.nih.gov/mesh/lookup/descriptor"
_mesh_cache = diskcache.Cache(str(CACHE_DIR / "mesh"))

# Common medical abbreviation → expanded term (supplements MeSH for abbreviations)
_ABBREV_MAP: dict[str, str] = {
    "nsclc": "non-small cell lung carcinoma",
    "sclc": "small cell lung carcinoma",
    "aml": "acute myeloid leukemia",
    "all": "acute lymphoblastic leukemia",
    "cll": "chronic lymphocytic leukemia",
    "cml": "chronic myelogenous leukemia",
    "mds": "myelodysplastic syndromes",
    "mm": "multiple myeloma",
    "dlbcl": "diffuse large b-cell lymphoma",
    "hl": "hodgkin lymphoma",
    "nhl": "non-hodgkin lymphoma",
    "t2dm": "type 2 diabetes mellitus",
    "dm2": "type 2 diabetes mellitus",
    "htn": "hypertension",
    "cad": "coronary artery disease",
    "hf": "heart failure",
    "chf": "congestive heart failure",
    "ckd": "chronic kidney disease",
    "esrd": "end-stage renal disease",
    "copd": "chronic obstructive pulmonary disease",
    "ibd": "inflammatory bowel disease",
    "ms": "multiple sclerosis",
    "ra": "rheumatoid arthritis",
    "sle": "systemic lupus erythematosus",
    "uc": "ulcerative colitis",
    "gerd": "gastroesophageal reflux disease",
    "tia": "transient ischemic attack",
    "mi": "myocardial infarction",
    "acs": "acute coronary syndrome",
    "hiv": "hiv infections",
    "hcc": "hepatocellular carcinoma",
    "rcc": "renal cell carcinoma",
    "tnbc": "triple negative breast neoplasms",
    "gist": "gastrointestinal stromal tumor",
    "pe": "pulmonary embolism",
    "dvt": "deep vein thrombosis",
    "bph": "benign prostatic hyperplasia",
    "avm": "arteriovenous malformations",
    "tbi": "traumatic brain injury",
    "als": "amyotrophic lateral sclerosis",
    "pd": "parkinson disease",
    "ad": "alzheimer disease",
}


def lookup_mesh_term(condition: str, timeout: float = 5.0) -> str | None:
    """
    Look up the NLM MeSH preferred descriptor label for a condition string.

    Returns the first matching MeSH label, or None if not found / API unavailable.
    Cached indefinitely (MeSH terms are stable).
    """
    key = f"mesh:{condition.strip().lower()}"
    if key in _mesh_cache:
        return _mesh_cache[key]

    try:
        resp = requests.get(
            _MESH_API,
            params={"label": condition, "match": "contains", "limit": "3"},
            timeout=timeout,
            headers={"Accept": "application/json"},
        )
        result: str | None = None
        if resp.status_code == 200:
            entries = resp.json()
            if entries:
                result = entries[0].get("label")
        _mesh_cache[key] = result
        return result
    except Exception as exc:
        logger.debug("MeSH lookup failed for '%s': %s", condition, exc)
        _mesh_cache[key] = None
        return None


def _condition_variants(condition: str) -> list[str]:
    """
    Generate up to 3 unique query.cond variants for a single condition.

    Strategy:
    1. Expand known abbreviations (e.g. NSCLC → non-small cell lung carcinoma)
    2. 1-word variant: broadest (query.cond AND logic with 1 term)
    3. 2-word variant: slightly more specific
    4. MeSH first word (synonym from controlled vocabulary)

    All variants are lowercase, min 3 chars.
    """
    raw = condition.strip().lower()
    expanded = _ABBREV_MAP.get(raw, raw)

    variants: list[str] = []
    seen: set[str] = set()

    def add(s: str) -> None:
        w = s.strip()
        if w and len(w) >= 3 and w not in seen:
            variants.append(w)
            seen.add(w)

    words = expanded.split()
    if words:
        add(words[0])
    if len(words) >= 2:
        add(" ".join(words[:2]))

    mesh = lookup_mesh_term(condition)
    if mesh:
        # MeSH labels are often inverted: "Carcinoma, Non-Small-Cell Lung"
        # Take first word before the comma for a clean 1-term query
        mesh_first_part = mesh.lower().split(",")[0]
        mesh_words = mesh_first_part.split()
        if mesh_words:
            add(mesh_words[0])

    return variants[:3]


def retrieve_candidates(
    conditions: list[str],
    max_total: int = 500,
    per_query_limit: int = 200,
    use_cache: bool = True,
    cache: diskcache.Cache | None = None,
) -> list[dict]:
    """
    Retrieve up to max_total unique clinical trial candidates for a list of conditions.

    For each condition, generates query variants and queries CT API.
    Deduplicates results by NCT ID across all queries.

    Args:
        conditions: Medical conditions from patient profile (most important first).
        max_total: Hard ceiling on unique trials returned.
        per_query_limit: Max results per individual CT API query.
        use_cache: Whether to cache CT API query results.
        cache: Optional existing diskcache.Cache instance.

    Returns:
        Deduplicated list of flat trial dicts (NCT ID + metadata).
    """
    if not conditions:
        return []

    if use_cache and cache is None:
        cache = diskcache.Cache(str(CACHE_DIR / "query_builder"))

    # Build ordered variant list across top 4 conditions (deduped)
    all_variants: list[str] = []
    seen_v: set[str] = set()
    for cond in conditions[:4]:
        for v in _condition_variants(cond):
            if v not in seen_v:
                all_variants.append(v)
                seen_v.add(v)

    logger.info(
        "Query builder: %d conditions → %d variants: %s",
        len(conditions), len(all_variants), all_variants,
    )

    seen_nct: set[str] = set()
    merged: list[dict] = []

    for variant in all_variants:
        if len(merged) >= max_total:
            break

        remaining = min(per_query_limit, max_total - len(merged))
        cache_key = f"qb:{variant}:n{remaining}"

        if use_cache and cache is not None and cache_key in cache:
            trials = cache[cache_key]
        else:
            try:
                trials = search_trials(query_cond=variant, max_results=remaining)
                time.sleep(CT_POLITE_DELAY)
                if use_cache and cache is not None:
                    cache[cache_key] = trials
            except Exception as exc:
                logger.warning("Query variant '%s' failed: %s", variant, exc)
                trials = []

        for trial in trials:
            nct_id = trial.get("nct_id", "")
            if nct_id and nct_id not in seen_nct:
                seen_nct.add(nct_id)
                merged.append(trial)
                if len(merged) >= max_total:
                    break

    logger.info(
        "Query builder: %d unique trials from %d variants",
        len(merged), len(all_variants),
    )
    return merged


def build_queries_clinical(
    patient_text: str,
    profile: PatientProfile,
    max_candidates: int = 800,
    per_query_limit: int = 200,
    use_cache: bool = True,
    cache: diskcache.Cache | None = None,
) -> list[str]:
    """
    Retrieve NCT IDs using ClinicalQueryPlanner + MeSH expansion.

    Combines structured clinical queries (condition/term) from the planner
    with existing MeSH-expanded queries.  Deduplicates across both paths.

    Returns up to max_candidates unique NCT IDs (strings, not dicts).
    """
    from src.retrieval.clinical_query_planner import plan_clinical_queries

    if use_cache and cache is None:
        cache = diskcache.Cache(str(CACHE_DIR / "query_builder"))

    seen_nct: set[str] = set()
    results: list[str] = []

    def _fetch(query_cond: str | None, query_term: str | None) -> None:
        remaining = max_candidates - len(results)
        if remaining <= 0:
            return
        n = min(per_query_limit, remaining)
        ck = f"cqb:{'c' if query_cond else 't'}:{query_cond or query_term}:n{n}"
        if use_cache and cache is not None and cache.get(ck) is not None:
            trials = cache[ck]
        else:
            try:
                trials = search_trials(
                    query_cond=query_cond,
                    query_term=query_term,
                    max_results=n,
                )
                time.sleep(CT_POLITE_DELAY)
                if use_cache and cache is not None:
                    cache[ck] = trials
            except Exception as exc:
                logger.warning("Clinical query failed (%s/%s): %s", query_cond, query_term, exc)
                trials = []
        for t in trials:
            nct = t.get("nct_id", "")
            if nct and nct not in seen_nct:
                seen_nct.add(nct)
                results.append(nct)

    # Path 1: structured clinical queries from planner
    specs = plan_clinical_queries(profile, patient_text, max_queries=12)
    for spec in specs:
        if len(results) >= max_candidates:
            break
        if spec.query_type == "condition":
            words = spec.query.split()
            # CT API hard limit: max 2 terms for query.cond
            _fetch(query_cond=" ".join(words[:2]), query_term=None)
        else:
            # CT API hard limit: max 3 words for query.term (enforced by _clean_query)
            _fetch(query_cond=None, query_term=spec.query)

    # Path 2: MeSH expansion via existing retrieve_candidates
    if profile.conditions and len(results) < max_candidates:
        mesh_trials = retrieve_candidates(
            conditions=profile.conditions,
            max_total=max_candidates - len(results),
            per_query_limit=per_query_limit,
            use_cache=use_cache,
            cache=cache,
        )
        for t in mesh_trials:
            nct = t.get("nct_id", "")
            if nct and nct not in seen_nct:
                seen_nct.add(nct)
                results.append(nct)

    logger.info(
        "build_queries_clinical: %d unique NCT IDs (planner+MeSH, cap=%d)",
        len(results), max_candidates,
    )
    return results[:max_candidates]
