#!/usr/bin/env python3
"""
Demo: match a patient description to clinical trials.

Usage:
  python scripts/demo.py "58yo male with NSCLC EGFR mutation"
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _label(enriched: list, eliminated: bool) -> str:
    if eliminated:
        return "NOT_ELIGIBLE"
    excl_viol = sum(1 for ct, _, v in enriched if ct == "exclusion" and v.verdict == "NOT_MET")
    inc_met = sum(1 for ct, _, v in enriched if ct == "inclusion" and v.verdict == "MET")
    inc_nei = sum(1 for ct, _, v in enriched if ct == "inclusion" and v.verdict == "NEI")
    if excl_viol > 0:
        return "NOT_ELIGIBLE"
    if inc_met > 0 and inc_nei == 0:
        return "ELIGIBLE"
    if inc_met > 0 and inc_nei > 0:
        return "ELIGIBLE_PENDING_CLARIFICATION"
    if inc_nei > 0:
        return "ELIGIBLE_PENDING_CLARIFICATION"
    return "NEI"


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python scripts/demo.py "<patient description>"')
        sys.exit(1)

    patient_text = " ".join(sys.argv[1:])

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT.parent / ".env")  # fallback for dev setups
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.")
        print("  Create a .env file in the project root:")
        print("    OPENAI_API_KEY=sk-...")
        sys.exit(1)

    import diskcache
    from src.config import CACHE_DIR
    from src.matching.eligibility_reasoner import evaluate_trial
    from src.matching.hard_filter import apply_hard_filter
    from src.output.nei_question_generator import generate_nei_question
    from src.parsing.criteria_parser import parse_criteria
    from src.parsing.patient_normalizer import normalize_patient
    from src.ranking.scorer import score_trial
    from src.retrieval.query_builder import retrieve_candidates

    # 1. Normalize patient
    profile = normalize_patient(patient_text, use_cache=True)
    age_str = f"{profile.age}yo" if profile.age else "?"
    conds = ", ".join(profile.conditions[:3]) if profile.conditions else "unknown"
    print(f"\nPatient: {age_str} {profile.gender} | {conds}")
    print("─" * 50)

    # 2. Retrieve top-10 candidates via MeSH queries
    conditions = profile.conditions or patient_text.split()[:2]
    trials = retrieve_candidates(conditions, max_total=10, use_cache=True)
    if not trials:
        print("No candidates found. Try a more specific patient description.")
        sys.exit(0)

    # 3. Run eligibility on top-5
    criteria_cache = diskcache.Cache(str(CACHE_DIR / "criteria_parser"))
    reasoner_cache = diskcache.Cache(str(CACHE_DIR / "eligibility_reasoner"))

    scored: list[dict] = []
    for trial in trials[:5]:
        nct_id = trial.get("nct_id", "")
        crit_text = trial.get("eligibility_criteria", "")
        try:
            parsed = parse_criteria(crit_text, use_cache=True, cache=criteria_cache)
            excl_det = [c for c in parsed.criteria if c.type == "exclusion" and c.deterministic]
            survived = apply_hard_filter(profile, excl_det)
            if not survived:
                scored.append({"trial": trial, "score": 0.0, "enriched": [], "eliminated": True})
                continue
            typed_verdicts = evaluate_trial(
                profile, parsed.criteria, nct_id=nct_id,
                use_cache=True, cache=reasoner_cache,
            )
            enriched = [
                (ctype, c.text, v)
                for c, (ctype, v) in zip(parsed.criteria, typed_verdicts)
            ]
            phases = trial.get("phases") or []
            meta = {"phase": phases[0] if phases else "", "status": trial.get("status", "")}
            score = score_trial(typed_verdicts, meta)
            scored.append({"trial": trial, "score": score, "enriched": enriched, "eliminated": False})
        except Exception as exc:
            scored.append({"trial": trial, "score": 0.0, "enriched": [], "error": str(exc)})

    for trial in trials[5:]:
        scored.append({"trial": trial, "score": 0.0, "enriched": [], "skipped": True})

    scored.sort(key=lambda x: x["score"], reverse=True)

    # 4. Print results
    print()
    for rank, item in enumerate(scored, 1):
        trial = item["trial"]
        nct_id = trial.get("nct_id", "?")
        title = trial.get("title", "Unknown")[:72]
        score = item["score"]
        enriched = item.get("enriched", [])
        eliminated = item.get("eliminated", False)

        if item.get("skipped") or item.get("error"):
            verdict_label = "NOT_EVALUATED"
        else:
            verdict_label = _label(enriched, eliminated)

        print(f"#{rank} {nct_id} [{score:.2f}] {verdict_label}")
        print(f"   {title}")

        if enriched and not item.get("skipped") and not item.get("error"):
            met = [(ct, txt, v) for ct, txt, v in enriched if v.verdict == "MET"]
            for _, txt, _ in met[:2]:
                print(f"   ✓ MET:  {txt[:80]}")

            nei_items = [(ct, txt, v) for ct, txt, v in enriched if v.verdict == "NEI"]
            for _, txt, v in nei_items[:2]:
                print(f"   ? NEI:  {txt[:80]}")
                try:
                    q = generate_nei_question(
                        verdict=v,
                        criterion_text=txt,
                        trial_title=trial.get("title", ""),
                        profile=profile,
                        use_cache=True,
                    )
                    print(f"     → {q}")
                except Exception:
                    pass
        print()


if __name__ == "__main__":
    main()
