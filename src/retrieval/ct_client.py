"""
Client for ClinicalTrials.gov REST API v2.

Note: httpx is blocked by the CT API's Cloudflare protection (TLS fingerprint).
We use `requests` which passes the fingerprint check.

Usage:
    trials = search_trials(query_term="lung cancer EGFR")
    trials = search_trials(query_cond="anaplastic astrocytoma", max_results=500)
"""
from __future__ import annotations

import logging
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import (
    CT_API_BASE,
    CT_MAX_RESULTS,
    CT_MAX_RETRIES,
    CT_PAGE_SIZE,
    CT_POLITE_DELAY,
    CT_RETRY_BASE_DELAY,
)

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
}


def _make_session() -> requests.Session:
    """Session with automatic retry on transient HTTP errors."""
    session = requests.Session()
    session.headers.update(_HEADERS)
    retry = Retry(
        total=CT_MAX_RETRIES,
        backoff_factor=CT_RETRY_BASE_DELAY,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _parse_study(raw: dict) -> dict:
    """Flatten a raw v2 API study into a canonical flat dict."""
    ps = raw.get("protocolSection", {})
    id_mod = ps.get("identificationModule", {})
    status_mod = ps.get("statusModule", {})
    desc_mod = ps.get("descriptionModule", {})
    elig_mod = ps.get("eligibilityModule", {})
    design_mod = ps.get("designModule", {})
    cond_mod = ps.get("conditionsModule", {})

    return {
        "nct_id": id_mod.get("nctId", ""),
        "title": id_mod.get("briefTitle", ""),
        "status": status_mod.get("overallStatus", ""),
        "brief_summary": desc_mod.get("briefSummary", ""),
        "eligibility_criteria": elig_mod.get("eligibilityCriteria", ""),
        "sex": elig_mod.get("sex", "ALL"),
        "min_age": elig_mod.get("minimumAge", ""),
        "max_age": elig_mod.get("maximumAge", ""),
        "std_ages": elig_mod.get("stdAges", []),
        "phases": design_mod.get("phases", []),
        "conditions": cond_mod.get("conditions", []),
    }


def search_trials(
    query_term: str | None = None,
    query_cond: str | None = None,
    overall_status: list[str] | None = None,
    page_size: int = CT_PAGE_SIZE,
    max_results: int = CT_MAX_RESULTS,
) -> list[dict]:
    """
    Query ClinicalTrials.gov v2 API and return up to max_results parsed trials.

    Args:
        query_term: Free-text search (title, description, criteria).
        query_cond: Condition/disease filter.
        overall_status: e.g. ["RECRUITING", "ACTIVE_NOT_RECRUITING"].
        page_size: Results per API page (max 1000).
        max_results: Hard ceiling on returned studies.

    Returns:
        List of flat study dicts with canonical field names.
    """
    if not query_term and not query_cond:
        raise ValueError("Provide at least one of query_term or query_cond.")

    params: dict[str, Any] = {
        "format": "json",
        "pageSize": min(page_size, 1000),
    }
    if query_term:
        params["query.term"] = query_term[:500]  # URL-length safety
    if query_cond:
        params["query.cond"] = query_cond
    if overall_status:
        params["filter.overallStatus"] = ",".join(overall_status)

    studies: list[dict] = []
    page_token: str | None = None
    session = _make_session()

    while len(studies) < max_results:
        if page_token:
            params["pageToken"] = page_token

        try:
            resp = session.get(
                f"{CT_API_BASE}/studies", params=params, timeout=60.0
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.error("CT API request failed: %s", exc)
            break

        raw_studies = data.get("studies", [])
        if not raw_studies:
            break

        for s in raw_studies:
            parsed = _parse_study(s)
            if parsed["nct_id"]:
                studies.append(parsed)
                if len(studies) >= max_results:
                    break

        page_token = data.get("nextPageToken")
        if not page_token:
            break

        time.sleep(CT_POLITE_DELAY)

    logger.info("Retrieved %d studies from CT API.", len(studies))
    return studies
