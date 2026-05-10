"""
Biomedical query expansion: abbreviations, MeSH synonyms, therapy-class mapping.

Parts 1 & 7 of the high-recall retrieval architecture.

Deterministic-first: no LLM needed unless use_llm=True.
All expansions are cached; zero network cost after first run.
"""
from __future__ import annotations

import hashlib
import logging
import re

import diskcache

from src.config import CACHE_DIR
from src.parsing.patient_normalizer import PatientProfile

logger = logging.getLogger(__name__)

_exp_cache = diskcache.Cache(str(CACHE_DIR / "query_expander"))

# ---------------------------------------------------------------------------
# Expansion vocabularies
# ---------------------------------------------------------------------------

# Abbreviation → preferred term (safe for CT API query.cond / BM25)
ABBREV: dict[str, str] = {
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
    "hcc": "hepatocellular carcinoma",
    "rcc": "renal cell carcinoma",
    "tnbc": "triple negative breast neoplasms",
    "gist": "gastrointestinal stromal tumor",
    "pe": "pulmonary embolism",
    "dvt": "deep vein thrombosis",
    "als": "amyotrophic lateral sclerosis",
    "pd": "parkinson disease",
    "ad": "alzheimer disease",
    "glioblastoma": "glioblastoma multiforme",
    "gbm": "glioblastoma multiforme",
    "tbi": "traumatic brain injury",
    "sah": "subarachnoid hemorrhage",
    "ich": "intracerebral hemorrhage",
    "pcos": "polycystic ovary syndrome",
    "her2": "ERBB2 amplified",
    "tnm": "tumor node metastasis",
}

# Drug/biomarker → drug-class synonyms
THERAPY_CLASS: dict[str, list[str]] = {
    "osimertinib": ["EGFR-TKI", "third generation EGFR inhibitor", "AZD9291"],
    "erlotinib":   ["EGFR-TKI", "first generation EGFR inhibitor"],
    "gefitinib":   ["EGFR-TKI", "first generation EGFR inhibitor"],
    "afatinib":    ["EGFR-TKI", "second generation EGFR inhibitor"],
    "crizotinib":  ["ALK inhibitor", "ALK-TKI", "MET inhibitor"],
    "alectinib":   ["ALK inhibitor", "second generation ALK-TKI"],
    "pembrolizumab": ["PD-1 inhibitor", "immune checkpoint inhibitor", "anti-PD-1"],
    "nivolumab":   ["PD-1 inhibitor", "immune checkpoint inhibitor", "anti-PD-1"],
    "atezolizumab": ["PD-L1 inhibitor", "immune checkpoint inhibitor"],
    "imatinib":    ["BCR-ABL inhibitor", "tyrosine kinase inhibitor"],
    "ibrutinib":   ["BTK inhibitor"],
    "venetoclax":  ["BCL-2 inhibitor"],
    "bevacizumab": ["VEGF inhibitor", "anti-angiogenic"],
    "trastuzumab": ["HER2 inhibitor", "anti-HER2"],
    "carboplatin": ["platinum chemotherapy", "cytotoxic"],
    "cisplatin":   ["platinum chemotherapy", "cytotoxic"],
    "paclitaxel":  ["taxane chemotherapy"],
    "docetaxel":   ["taxane chemotherapy"],
}

# Biomarker → expanded forms
BIOMARKER_EXPAND: dict[str, list[str]] = {
    "egfr": ["EGFR mutation", "EGFR exon 19", "EGFR L858R", "epidermal growth factor receptor"],
    "alk":  ["ALK rearrangement", "ALK translocation", "anaplastic lymphoma kinase"],
    "kras": ["KRAS mutation", "KRAS G12C", "KRAS G12D"],
    "braf": ["BRAF V600E", "BRAF mutation"],
    "her2": ["HER2 amplification", "ERBB2", "HER2 positive"],
    "brca": ["BRCA1", "BRCA2", "BRCA mutation", "homologous recombination"],
    "pdl1": ["PD-L1 expression", "immune checkpoint", "TMB"],
    "msi":  ["microsatellite instability", "mismatch repair deficient", "MMR deficient"],
    "tp53": ["TP53 mutation", "p53 mutation"],
    "met":  ["MET amplification", "MET exon 14", "c-MET"],
    "ret":  ["RET rearrangement", "RET fusion"],
    "ros1": ["ROS1 rearrangement", "ROS1 fusion"],
    "ntrk": ["NTRK fusion", "TRK inhibitor"],
    "idh1": ["IDH1 mutation", "isocitrate dehydrogenase"],
    "idh2": ["IDH2 mutation", "isocitrate dehydrogenase"],
}


def expand_term(term: str) -> list[str]:
    """
    Return a list of expanded forms for a single term.
    Always includes the original. May include abbreviation expansion,
    therapy-class synonyms, and biomarker expansions.
    """
    lower = term.lower().strip()
    results: list[str] = [term]

    # Abbreviation expansion
    if lower in ABBREV:
        results.append(ABBREV[lower])

    # Therapy class
    if lower in THERAPY_CLASS:
        results.extend(THERAPY_CLASS[lower])

    # Biomarker expansion (match on first token)
    first_token = re.split(r"[\s\-]", lower)[0]
    if first_token in BIOMARKER_EXPAND:
        results.extend(BIOMARKER_EXPAND[first_token])

    return list(dict.fromkeys(results))  # deduplicate, preserve order


def expand_query(query: str) -> list[str]:
    """
    Expand a full query string by expanding each content token.
    Returns the original query plus all synonym-expanded variants (short forms only).
    """
    tokens = query.lower().split()
    variants: list[str] = [query]

    for token in tokens:
        clean = re.sub(r"[^\w]", "", token)
        if clean in ABBREV:
            variants.append(query.lower().replace(token, ABBREV[clean]))
        if clean in THERAPY_CLASS:
            for syn in THERAPY_CLASS[clean][:2]:
                variants.append(query.lower().replace(token, syn))

    return list(dict.fromkeys(variants))


# ---------------------------------------------------------------------------
# Multi-query generation from PatientProfile (Part 1)
# ---------------------------------------------------------------------------

def generate_queries(profile: PatientProfile, raw_text: str = "") -> list[str]:
    """
    Generate diverse retrieval queries from a PatientProfile.

    Strategy: decompose the profile into independent facets, each yielding
    a focused query. More facets → higher union recall.

    Returns a deduplicated list of 5–12 short query strings.
    """
    queries: list[str] = []

    # ---- Facet 1: each condition individually (expanded) ----
    for cond in profile.conditions[:4]:
        for variant in expand_term(cond)[:2]:
            queries.append(variant)

    # ---- Facet 2: first condition + first biomarker/medication ----
    all_meds = list(profile.medications) + list(profile.prior_treatments)
    if profile.conditions and all_meds:
        primary = profile.conditions[0]
        for med in all_meds[:3]:
            for c_exp in expand_term(primary)[:1]:
                for m_exp in expand_term(med)[:1]:
                    queries.append(f"{c_exp} {m_exp}")

    # ---- Facet 3: biomarker-only queries ----
    biomarker_hints: list[str] = []
    for med in all_meds[:4]:
        lower = med.lower()
        first = re.split(r"[\s\-]", lower)[0]
        if first in BIOMARKER_EXPAND:
            biomarker_hints.extend(BIOMARKER_EXPAND[first][:2])
        elif first in THERAPY_CLASS:
            biomarker_hints.extend(THERAPY_CLASS[first][:1])
    for bm in biomarker_hints[:4]:
        queries.append(bm)

    # ---- Facet 4: prior therapy resistance / progression context ----
    resistance_words = {"resistant", "refractory", "progressed", "progressive",
                        "relapsed", "recurrent", "failed"}
    text_lower = raw_text.lower()
    for med in all_meds[:3]:
        for rw in resistance_words:
            if rw in text_lower:
                for exp in expand_term(med)[:1]:
                    queries.append(f"{exp} {rw}")
                break

    # ---- Facet 5: demographics + primary condition ----
    if profile.conditions:
        primary_expanded = expand_term(profile.conditions[0])[0]
        if profile.age:
            age_bucket = "pediatric" if profile.age < 18 else "adult"
            queries.append(f"{age_bucket} {primary_expanded}")
        if profile.gender in ("male", "female"):
            queries.append(f"{profile.gender} {primary_expanded}")

    # ---- Facet 6: raw text first 15 tokens (catch anything missed) ----
    if raw_text:
        first_tokens = " ".join(raw_text.split()[:15])
        queries.append(first_tokens)

    # Deduplicate and filter very short queries
    seen: set[str] = set()
    result: list[str] = []
    for q in queries:
        q = q.strip()
        if len(q) >= 3 and q not in seen:
            seen.add(q)
            result.append(q)

    logger.debug("Generated %d queries from profile: %s", len(result), result[:3])
    return result[:12]  # cap at 12 to control latency


def generate_queries_llm(profile: PatientProfile, raw_text: str, use_cache: bool = True) -> list[str]:
    """
    Optional LLM-augmented query generation.
    Adds 3–5 additional queries on top of deterministic ones.
    Cached by raw_text hash.
    """
    base = generate_queries(profile, raw_text)

    key = f"llm_queries:{hashlib.md5(raw_text.encode()).hexdigest()}"
    if use_cache and _exp_cache.get(key) is not None:
        llm_extras = _exp_cache[key]
        return list(dict.fromkeys(base + llm_extras))

    try:
        from pydantic import BaseModel, Field
        from src.llm_client import complete_structured

        class _Queries(BaseModel):
            queries: list[str] = Field(
                description="5 short BM25-optimised retrieval queries for clinical trial search",
                max_length=5,
            )

        prompt = (
            "You are a biomedical information retrieval expert. "
            "Given the patient description below, generate exactly 5 short retrieval queries "
            "optimised for searching a clinical trial database. "
            "Each query should be 2–6 words, using MeSH-compatible terminology. "
            "Vary the facets: disease subtype, biomarker, therapy class, stage, mechanism.\n\n"
            f"Patient: {raw_text[:500]}\n\n"
            "Return JSON: {\"queries\": [\"...\", ...]}"
        )
        result = complete_structured(prompt, _Queries)
        llm_extras = [q.strip() for q in result.queries if len(q.strip()) >= 3]
    except Exception as exc:
        logger.warning("LLM query generation failed (%s) — using deterministic only", exc)
        llm_extras = []

    if use_cache:
        _exp_cache[key] = llm_extras

    return list(dict.fromkeys(base + llm_extras))
