from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal

from src.parsing.patient_normalizer import PatientProfile


QueryType = Literal["condition", "term"]


@dataclass(frozen=True)
class QuerySpec:
    query_type: QueryType
    query: str
    reason: str
    priority: int


CONDITION_SYNONYMS = {
    "nsclc": ["non-small cell lung cancer", "non-small cell lung carcinoma"],
    "sclc": ["small cell lung cancer"],
    "tnbc": ["triple negative breast cancer"],
    "hcc": ["hepatocellular carcinoma"],
    "rcc": ["renal cell carcinoma"],
    "aml": ["acute myeloid leukemia"],
    "cll": ["chronic lymphocytic leukemia"],
    "cml": ["chronic myeloid leukemia"],
}

BIOMARKERS = {
    "egfr", "alk", "ros1", "braf", "kras", "her2", "brca1", "brca2",
    "pd-l1", "pdl1", "msi", "mmr", "ntrk", "ret", "met", "flt3",
    "idh1", "idh2", "cd19", "cd20", "bcma",
}

STAGE_WORDS = {
    "metastatic", "advanced", "unresectable", "relapsed", "refractory",
    "recurrent", "stage iii", "stage iv",
}

TREATMENT_WORDS = {
    "chemotherapy", "radiotherapy", "immunotherapy", "platinum",
    "cisplatin", "carboplatin", "pembrolizumab", "nivolumab",
    "surgery", "osimertinib", "trastuzumab",
}

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "with", "without", "for",
    "in", "on", "to", "by", "patient", "patients", "male", "female",
    "year", "old", "yo", "prior", "previous", "current",
}


def plan_clinical_queries(
    patient: PatientProfile,
    patient_text: str,
    max_queries: int = 12,
) -> list[QuerySpec]:
    queries: list[QuerySpec] = []

    queries.extend(_condition_queries(patient.conditions))
    queries.extend(_biomarker_queries(patient, patient_text))
    queries.extend(_stage_queries(patient_text))
    queries.extend(_treatment_queries(patient, patient_text))

    if not queries:
        fallback_query = _clean_query(patient_text)
        if fallback_query:
            queries.append(
                QuerySpec(
                    query_type="term",
                    query=fallback_query,
                    reason="Fallback query from patient text",
                    priority=10,
                )
            )

    queries = _remove_duplicates(queries)
    queries.sort(key=lambda item: item.priority, reverse=True)

    return queries[:max_queries]


def _condition_queries(conditions: list[str]) -> list[QuerySpec]:
    queries: list[QuerySpec] = []

    for index, condition in enumerate(conditions[:4]):
        aliases = _expand_condition(condition)

        for alias_index, alias in enumerate(aliases[:3]):
            query = _clean_query(alias)

            if query:
                queries.append(
                    QuerySpec(
                        query_type="condition",
                        query=query,
                        reason=f"Condition search based on: {condition}",
                        priority=100 - index * 10 - alias_index,
                    )
                )

    return queries


def _biomarker_queries(patient: PatientProfile, patient_text: str) -> list[QuerySpec]:
    # Only search conditions + patient_text; biomarkers appearing in
    # medications/treatments are treatments received, not condition biomarkers.
    condition_text = patient_text + " " + " ".join(patient.conditions)
    found_biomarkers = _find_terms_exact(condition_text, BIOMARKERS)

    return [
        QuerySpec(
            query_type="term",
            query=biomarker,
            reason=f"Biomarker detected in condition context: {biomarker}",
            priority=85,
        )
        for biomarker in found_biomarkers[:5]
    ]


def _stage_queries(patient_text: str) -> list[QuerySpec]:
    found_stages = _find_terms(patient_text, STAGE_WORDS)

    return [
        QuerySpec(
            query_type="term",
            query=stage,
            reason=f"Disease stage search: {stage}",
            priority=75,
        )
        for stage in found_stages[:4]
    ]


def _treatment_queries(patient: PatientProfile, patient_text: str) -> list[QuerySpec]:
    text = _combined_patient_text(patient, patient_text)
    found_treatments = _find_terms(text, TREATMENT_WORDS)

    return [
        QuerySpec(
            query_type="term",
            query=treatment,
            reason=f"Treatment history search: {treatment}",
            priority=65,
        )
        for treatment in found_treatments[:4]
    ]


def _expand_condition(condition: str) -> list[str]:
    condition = _normalise(condition)
    aliases = [condition]

    for abbreviation, synonyms in CONDITION_SYNONYMS.items():
        if abbreviation in condition.split():
            aliases.extend(synonyms)

    if "cancer" in condition:
        aliases.append(condition.replace("cancer", "carcinoma"))

    if "carcinoma" in condition:
        aliases.append(condition.replace("carcinoma", "cancer"))

    return _deduplicate(aliases)


def _combined_patient_text(patient: PatientProfile, patient_text: str) -> str:
    return " ".join(
        [
            patient_text,
            " ".join(patient.prior_treatments),
            " ".join(patient.medications),
            " ".join(patient.relevant_history),
        ]
    )


def _find_terms(text: str, terms: Iterable[str]) -> list[str]:
    text = _normalise(text)
    found = []

    for term in terms:
        if re.search(r'\b' + re.escape(term) + r'\b', text):
            found.append(term)

    return _deduplicate(found)


def _find_terms_exact(text: str, terms: Iterable[str]) -> list[str]:
    """Word-boundary–safe term search; prevents substring false positives."""
    text = _normalise(text)
    found = []

    for term in terms:
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, text):
            found.append(term)

    return _deduplicate(found)


def _clean_query(text: str, max_words: int = 3) -> str:
    text = re.sub(r"[^a-zA-Z0-9+\-\s]", " ", text.lower())
    words = [word for word in text.split() if word not in STOPWORDS]

    return " ".join(words[:max_words])


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _deduplicate(items: Iterable[str]) -> list[str]:
    result = []
    seen = set()

    for item in items:
        item = _normalise(item)

        if item and item not in seen:
            seen.add(item)
            result.append(item)

    return result


def _remove_duplicates(queries: list[QuerySpec]) -> list[QuerySpec]:
    best_queries: dict[tuple[str, str], QuerySpec] = {}

    for query in queries:
        key = (query.query_type, query.query)

        if key not in best_queries:
            best_queries[key] = query
        elif query.priority > best_queries[key].priority:
            best_queries[key] = query

    return list(best_queries.values())