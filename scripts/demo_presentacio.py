#!/usr/bin/env python3
"""
Demo de presentació — UAB THE HACK × Deloitte, Clinical Trials AI Agent.

Mostra els 8 passos del pipeline de manera interactiva i pedagògica:
  1. PACIENT D'ENTRADA       — text clínic lliure
  2. NORMALITZACIÓ           — PatientProfile estructurat
  3. QUERIES GENERADES       — QuerySpec llista (plan_clinical_queries)
  4. TOP 5 CANDIDATS         — retrieve_candidates (CT API)
  5. FILTRE DUR              — apply_hard_filter per cada candidat
  6. RAONAMENT (top 3)       — parse_criteria + evaluate_trial
  7. PREGUNTES NEI           — generate_nei_question per criteria NEI
  8. RANKING FINAL + DOSSIER — score_trial_breakdown + generate_dossier

Ús:
  python scripts/demo_presentacio.py
  python scripts/demo_presentacio.py --patient "68yo male with NSCLC..."
  python scripts/demo_presentacio.py --topic 5          # pacient TREC topic 5
  python scripts/demo_presentacio.py --save             # desa dossier a data/dossiers/demo/
  python scripts/demo_presentacio.py --no-cache         # força re-avaluació
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# PATH SETUP — allow running from repo root or scripts/
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.parsing.patient_normalizer import normalize_patient
from src.retrieval.clinical_query_planner import plan_clinical_queries
from src.retrieval.query_builder import retrieve_candidates
from src.retrieval.ct_client import search_trials
from src.parsing.criteria_parser import parse_criteria
from src.matching.hard_filter import apply_hard_filter
from src.matching.eligibility_reasoner import evaluate_trial
from src.matching.label_deriver import derive_label
from src.ranking.scorer import score_trial_breakdown, explain_score_breakdown
from src.output.nei_question_generator import generate_nei_question
from src.output.dossier_generator import generate_dossier

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
DEFAULT_PATIENT = (
    "52-year-old woman with HER2-positive metastatic breast cancer. "
    "Previously treated with trastuzumab and pertuzumab combined with docetaxel "
    "with disease progression after 10 months. BRCA2 germline mutation detected. "
    "ECOG performance status 1. Brain metastases absent on recent MRI. "
    "Creatinine 0.8 mg/dL, ALT 22 U/L, hemoglobin 11.8 g/dL. "
    "Current medications: trastuzumab (stopped), capecitabine."
)

TREC_TOPICS: dict[int, str] = {
    1: (
        "Patient is a 45-year-old man with anaplastic astrocytoma of the spine "
        "complicated by severe lower extremity weakness and urinary retention, "
        "status post Foley catheter. Currently on levetiracetam, dexamethasone, "
        "oxycodone, terazosin. ECOG 2. Creatinine 1.1 mg/dL."
    ),
    2: (
        "58-year-old woman with stage IV non-small cell lung cancer (adenocarcinoma), "
        "EGFR exon 19 deletion positive. ECOG PS 1. No brain metastases. "
        "Previously treated with erlotinib, now with disease progression."
    ),
    3: (
        "67-year-old male with metastatic colorectal cancer, MSS, KRAS wild-type. "
        "Previously treated with FOLFOX and FOLFIRI plus bevacizumab, progression. "
        "ECOG 1. No prior anti-EGFR therapy."
    ),
    5: (
        "71-year-old male with metastatic prostate cancer, castration-resistant. "
        "Prior docetaxel chemotherapy. PSA 48 ng/mL. ECOG 1. "
        "Bone metastases present. Creatinine 1.2 mg/dL."
    ),
}

_SEP = "─" * 72
_THIN = "·" * 72
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

# ---------------------------------------------------------------------------
# DISPLAY HELPERS
# ---------------------------------------------------------------------------

def _header(step: int, title: str) -> None:
    print(f"\n{_SEP}")
    print(f"{_BOLD}{_CYAN}  PAS {step}: {title}{_RESET}")
    print(_SEP)


def _ok(msg: str) -> None:
    print(f"  {_GREEN}✓{_RESET}  {msg}")


def _warn(msg: str) -> None:
    print(f"  {_YELLOW}⚠{_RESET}  {msg}")


def _err(msg: str) -> None:
    print(f"  {_RED}✗{_RESET}  {msg}")


def _label_color(label: str) -> str:
    if label == "MET":
        return f"{_GREEN}{label}{_RESET}"
    if label == "NOT_MET":
        return f"{_RED}{label}{_RESET}"
    return f"{_YELLOW}{label}{_RESET}"


def _verdict_color(verdict: str) -> str:
    return _label_color(verdict)


def _timing(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds*1000:.0f} ms"
    return f"{seconds:.1f} s"


def _pause(msg: str = "Prem ENTER per continuar...") -> None:
    try:
        input(f"\n  {_YELLOW}▶{_RESET}  {msg} ")
    except (KeyboardInterrupt, EOFError):
        print()


# ---------------------------------------------------------------------------
# PIPELINE STEPS
# ---------------------------------------------------------------------------

def step1_pacient(patient_text: str) -> None:
    _header(1, "PACIENT D'ENTRADA")
    print()
    for line in patient_text.split(". "):
        if line.strip():
            print(f"  {line.strip()}.")
    print()
    _ok(f"Text clínic rebut: {len(patient_text)} caràcters")


def step2_normalitzacio(patient_text: str, use_cache: bool) -> object:
    _header(2, "NORMALITZACIÓ — PatientNormalizer (GPT-4o-mini + regex)")
    t0 = time.time()
    profile = normalize_patient(patient_text, use_cache=use_cache)
    elapsed = time.time() - t0

    print()
    print(f"  {'Edat:':<22} {profile.age or '—'}")
    print(f"  {'Sexe:':<22} {profile.gender or '—'}")
    print(f"  {'ECOG:':<22} {profile.ecog_score if profile.ecog_score is not None else '—'}")
    if profile.conditions:
        print(f"  {'Condicions:':<22} {', '.join(profile.conditions[:5])}")
    if profile.medications:
        print(f"  {'Medicaments:':<22} {', '.join(profile.medications[:5])}")
    if profile.prior_treatments:
        print(f"  {'Tractaments previs:':<22} {', '.join(profile.prior_treatments[:4])}")
    if profile.lab_values:
        labs_str = ", ".join(f"{k}={v}" for k, v in list(profile.lab_values.items())[:4])
        print(f"  {'Lab values:':<22} {labs_str}")
    if profile.relevant_history:
        print(f"  {'Historial clínic:':<22} {', '.join(profile.relevant_history[:3])}")

    print()
    _ok(f"PatientProfile extret en {_timing(elapsed)}")
    return profile


def step3_queries(profile: object, patient_text: str) -> list:
    _header(3, "QUERIES GENERADES — ClinicalQueryPlanner (heurística + LLM)")
    t0 = time.time()
    queries = plan_clinical_queries(profile, patient_text, max_queries=10)
    elapsed = time.time() - t0

    print()
    print(f"  {'#':<4} {'Tipus':<12} {'Query':<40} {'Prioritat'}")
    print(f"  {_THIN}")
    for i, q in enumerate(queries, 1):
        print(f"  {i:<4} {q.query_type:<12} {q.query[:40]:<40} {q.priority}")

    print()
    _ok(f"{len(queries)} queries generades en {_timing(elapsed)}")
    return queries


_STAGE_PREFIX_RE = __import__("re").compile(
    r"^(stage\s+[ivxIVX\d]+\s+|metastatic\s+|advanced\s+|recurrent\s+|refractory\s+|relapsed\s+)+",
    __import__("re").IGNORECASE,
)


def _clean_condition(cond: str) -> str:
    """Strip staging/grade prefixes so CT API query.cond gets a clean disease name."""
    return _STAGE_PREFIX_RE.sub("", cond).strip()


def step4_candidats(profile: object, use_cache: bool, candidate_cap: int) -> list[dict]:
    _header(4, f"TOP {candidate_cap} CANDIDATS — CT API (ClinicalTrials.gov)")
    t0 = time.time()

    conditions = getattr(profile, "conditions", [])
    if not conditions:
        _warn("No s'han detectat condicions — recuperació limitada")
        conditions = ["cancer"]

    # Strip staging prefixes so CT API gets clean condition names
    clean_conditions = [_clean_condition(c) for c in conditions]
    clean_conditions = [c for c in clean_conditions if c]

    # Build supplementary term query from biomarkers / medications
    history = getattr(profile, "relevant_history", [])
    meds = getattr(profile, "medications", [])
    term_parts = [h for h in history[:3] if any(k in h.lower() for k in
                  ["mutation", "positive", "negative", "deletion", "amplification",
                   "egfr", "her2", "brca", "kras", "braf", "alk", "ros"])]
    term_parts += [m for m in meds[:2] if m]
    term_query = " ".join(term_parts)[:200] if term_parts else None

    print(f"\n  Condicions (netejades): {', '.join(clean_conditions)}")
    if term_query:
        print(f"  Termes addicionals:     {term_query[:80]}")

    seen_nct: set[str] = set()
    candidates: list[dict] = []

    def _fetch(cond: str, term: str | None, label: str) -> None:
        if len(candidates) >= candidate_cap:
            return
        remaining = candidate_cap - len(candidates)
        print(f"\n  Cercant ({label}): query.cond='{cond}'"
              + (f" + term='{term[:35]}'" if term else ""))
        try:
            results = search_trials(query_cond=cond, query_term=term, max_results=remaining)
            added = 0
            for trial in results:
                nct = trial.get("nct_id", "")
                if nct and nct not in seen_nct:
                    seen_nct.add(nct)
                    candidates.append(trial)
                    added += 1
            print(f"           → {added} nous assajos (total: {len(candidates)})")
        except Exception as exc:
            _warn(f"CT API error: {exc}")

    # Pass 1: specific — condition + biomarker terms (most relevant first)
    for cond in clean_conditions[:2]:
        _fetch(cond, term_query, "específic")

    # Pass 2: broader — condition only, no term filter (fills up to cap)
    for cond in clean_conditions[:2]:
        if len(candidates) >= candidate_cap:
            break
        _fetch(cond, None, "ampli")

    elapsed = time.time() - t0

    print()
    print(f"  {'#':<4} {'NCT ID':<14} {'Fase':<10} {'Estat':<16} {'Títol'}")
    print(f"  {_THIN}")
    for i, trial in enumerate(candidates[:5], 1):
        nct = trial.get("nct_id", "—")
        phase = (trial.get("phase") or "—")[:8]
        status = (trial.get("status") or "—")[:14]
        title = (trial.get("title") or trial.get("brief_title") or "—")[:45]
        print(f"  {i:<4} {nct:<14} {phase:<10} {status:<16} {title}")

    if len(candidates) > 5:
        print(f"  ... i {len(candidates) - 5} candidats més")

    print()
    _ok(f"{len(candidates)} assajos candidats recuperats en {_timing(elapsed)}")
    return candidates


def step5_filtre_dur(candidates: list[dict], profile: object) -> list[dict]:
    _header(5, "FILTRE DUR — HardFilter (edat / ECOG / labs, sense LLM)")
    t0 = time.time()

    # Demo: parse criteria only for the first DEMO_CHECK trials to avoid LLM bottleneck.
    # The rest pass through — hard filter is a pre-screen, not the main evaluation.
    DEMO_CHECK = 5

    survivors: list[dict] = []
    eliminated_count = 0

    print(f"\n  Avaluant {min(DEMO_CHECK, len(candidates))} candidats en detall "
          f"(+{max(0, len(candidates)-DEMO_CHECK)} passen directament al pas 6)...")
    print()
    print(f"  {'NCT ID':<14} {'Resultat':<14} Criteri violat")
    print(f"  {_THIN}")

    for i, trial in enumerate(candidates):
        nct = trial.get("nct_id", "—")

        if i >= DEMO_CHECK:
            # Pass remaining trials without LLM parse — shown in summary only
            survivors.append(trial)
            continue

        criteria_text = trial.get("eligibility_criteria") or trial.get("criteria") or ""
        if not criteria_text:
            survivors.append(trial)
            print(f"  {nct:<14} {_GREEN}PASSA{_RESET}          (sense criteris publicats)")
            continue

        # parse_criteria uses cache → fast on 2nd run
        parsed = parse_criteria(criteria_text, use_cache=True)
        det_excl = [c for c in parsed.criteria if c.type == "exclusion" and c.deterministic]
        passes = apply_hard_filter(profile, det_excl)

        if passes:
            survivors.append(trial)
            print(f"  {nct:<14} {_GREEN}PASSA{_RESET}          ({len(det_excl)} criteris det. OK)")
        else:
            eliminated_count += 1
            violated = next(
                (c.text[:45] for c in det_excl),
                "criteri desconegut",
            )
            print(f"  {nct:<14} {_RED}ELIMINAT{_RESET}       {violated}")

    elapsed = time.time() - t0
    print()
    _ok(f"{len(survivors)}/{len(candidates)} passen el filtre dur ({elapsed:.1f} s)")
    if eliminated_count:
        _warn(f"{eliminated_count} eliminats per criteris deterministes (edat/ECOG/labs)")
    return survivors


def step6_raonament(survivors: list[dict], profile: object, use_cache: bool, reasoning_n: int = 10) -> list[dict]:
    top3 = survivors[:reasoning_n]
    _header(6, f"RAONAMENT LLM — EligibilityReasoner (top {len(top3)} assajos, GPT-4o-mini)")

    results = []

    for idx, trial in enumerate(top3, 1):
        nct = trial.get("nct_id", f"TRIAL{idx}")
        title = (trial.get("title") or trial.get("brief_title") or nct)[:60]
        print(f"\n  [{idx}] {_BOLD}{nct}{_RESET} — {title}")
        print(f"  {_THIN}")

        criteria_text = trial.get("eligibility_criteria") or trial.get("criteria") or ""
        parsed = parse_criteria(criteria_text, use_cache=use_cache)

        inc = [c for c in parsed.criteria if c.type == "inclusion"]
        exc = [c for c in parsed.criteria if c.type == "exclusion"]
        print(f"       Criteris parseats: {len(inc)} inclusió, {len(exc)} exclusió")

        t0 = time.time()
        verdicts_raw = evaluate_trial(
            profile=profile,
            criteria=parsed.criteria,
            nct_id=nct,
            use_cache=use_cache,
        )
        elapsed = time.time() - t0

        # Build (type, text, verdict) triples
        verdicts_full = [
            (ctype, crit.text, verd)
            for (ctype, verd), crit in zip(verdicts_raw, parsed.criteria)
        ]

        met = sum(1 for (t, _, v) in verdicts_full if t == "inclusion" and v.verdict == "MET")
        not_met = sum(1 for (t, _, v) in verdicts_full if v.verdict == "NOT_MET")
        nei = sum(1 for (_, _, v) in verdicts_full if v.verdict == "NEI")
        excl_viol = sum(1 for (t, _, v) in verdicts_full if t == "exclusion" and v.verdict == "NOT_MET")

        eligibility_summary = {
            "inclusion_met": met,
            "inclusion_not_met": not_met - excl_viol,
            "inclusion_nei": nei,
            "exclusion_violations": excl_viol,
        }
        label = derive_label(eligibility_summary)

        print(f"       Veredicte global: {_label_color(label)}  "
              f"(met={met} not_met={not_met} nei={nei} excl_viol={excl_viol})  "
              f"[{_timing(elapsed)}]")

        # Show top 4 individual verdicts
        for ctype, ctext, verd in verdicts_full[:4]:
            prefix = "INC" if ctype == "inclusion" else "EXC"
            print(f"       [{prefix}] {_verdict_color(verd.verdict):<8}  {ctext[:58]}")

        results.append({
            "trial": trial,
            "verdicts_raw": verdicts_raw,
            "verdicts_full": verdicts_full,
            "parsed": parsed,
            "label": label,
            "eligibility_summary": eligibility_summary,
        })

    print()
    _ok(f"Raonament completat per {len(top3)} assajos")
    return results


def step7_nei(results: list[dict], profile: object, use_cache: bool) -> list[dict]:
    _header(7, "PREGUNTES NEI — NEIQuestionGenerator (criteris incerts)")
    print()

    for r in results:
        trial = r["trial"]
        nct = trial.get("nct_id", "?")
        nei_verdicts = [(ctext, verd) for (_, ctext, verd) in r["verdicts_full"]
                        if verd.verdict == "NEI"]

        if not nei_verdicts:
            _ok(f"{nct}: cap criteri NEI — elegibilitat completament resolta")
            r["nei_questions"] = {}
            continue

        title = trial.get("title") or trial.get("brief_title") or nct
        print(f"  {_BOLD}{nct}{_RESET} — {len(nei_verdicts)} criteris NEI:")
        nei_questions: dict[str, str] = {}

        for ctext, verd in nei_verdicts[:3]:  # max 3 per demo speed
            question = generate_nei_question(
                verdict=verd,
                criterion_text=ctext,
                trial_title=title,
                profile=profile,
                use_cache=use_cache,
            )
            nei_questions[ctext] = question
            short_crit = ctext[:50] + ("..." if len(ctext) > 50 else "")
            print(f"    {_YELLOW}?{_RESET} Criteri: {short_crit}")
            print(f"      Pregunta: {_CYAN}{question}{_RESET}")

        r["nei_questions"] = nei_questions

    print()
    _ok("Preguntes NEI generades")
    return results


def step8_ranking(results: list[dict], profile: object, use_cache: bool, save: bool) -> None:
    _header(8, "RANKING FINAL + DOSSIER — Scorer + DossierGenerator")

    scored: list[tuple[float, dict]] = []
    for r in results:
        breakdown = score_trial_breakdown(
            verdicts=r["verdicts_raw"],
            metadata=r["trial"],
        )
        explanation = explain_score_breakdown(breakdown)
        scored.append((breakdown["final_score"], {**r, "breakdown": breakdown, "explanation": explanation}))

    scored.sort(key=lambda x: x[0], reverse=True)

    print()
    print(f"  {'Rank':<5} {'NCT ID':<14} {'Score':<8} {'Label':<10} {'Fase':<10} Títol")
    print(f"  {_THIN}")
    for rank, (score, r) in enumerate(scored, 1):
        trial = r["trial"]
        nct = trial.get("nct_id", "—")
        label = r["label"]
        phase = (trial.get("phase") or "—")[:8]
        title = (trial.get("title") or trial.get("brief_title") or "—")[:38]
        print(f"  {rank:<5} {nct:<14} {score:<8.4f} {_label_color(label):<19} {phase:<10} {title}")

    # Show breakdown for rank #1
    if scored:
        best_score, best_r = scored[0]
        print(f"\n  {_BOLD}Desglossament puntuació #{1} ({best_r['trial'].get('nct_id', '?')}):{_RESET}")
        bd = best_r["breakdown"]
        print(f"    Inclusió met ratio : {bd.get('inclusion_met_ratio', 0):.2f} × 0.55 = "
              f"{bd.get('weighted_inclusion', 0):.4f}")
        print(f"    Penalització exclusió: {bd.get('exclusion_penalty', 0):.2f} × 1.00 = "
              f"-{bd.get('weighted_exclusion', abs(bd.get('weighted_exclusion', 0))):.4f}")
        print(f"    Bonus fase         : {bd.get('phase_bonus', 0):.2f} × 0.15 = "
              f"{bd.get('weighted_phase', 0):.4f}")
        print(f"    Bonus reclutament  : {bd.get('recruiting_bonus', 0):.2f} × 0.10 = "
              f"{bd.get('weighted_recruiting', 0):.4f}")
        print(f"    Penalització NEI   : {bd.get('nei_ratio', 0):.2f} × 0.10 = "
              f"-{bd.get('weighted_nei_penalty', abs(bd.get('weighted_nei_penalty', 0))):.4f}")
        print(f"    {_BOLD}Score final: {best_score:.4f}{_RESET}")

    # Generate dossier for top trial
    if scored:
        best_score, best_r = scored[0]
        trial = best_r["trial"]
        nct = trial.get("nct_id", "DEMO")
        print(f"\n  Generant dossier clínic per {_BOLD}{nct}{_RESET}...")
        t0 = time.time()
        dossier = generate_dossier(
            profile=profile,
            verdicts=best_r["verdicts_full"],
            trial_metadata=trial,
            nei_questions=best_r["nei_questions"],
            score_explanation=best_r["explanation"],
            score_breakdown=best_r["breakdown"],
        )
        elapsed = time.time() - t0

        print(f"\n  {_BOLD}DOSSIER CLÍNIC:{_RESET}")
        print(f"    NCT ID             : {dossier.nct_id}")
        print(f"    Títol              : {dossier.trial_title[:65]}")
        print(f"    Fase / Estat       : {dossier.phase} / {dossier.status}")
        print(f"    Recomanació final  : {_BOLD}{dossier.final_recommendation}{_RESET}")
        print(f"    Confiança          : {dossier.confidence_score:.2f}")
        if dossier.score_explanation:
            print(f"    Explicació         : {dossier.score_explanation[:80]}")
        if dossier.attention_flags:
            for flag in dossier.attention_flags[:3]:
                print(f"    {_YELLOW}⚠{_RESET} Flag: {flag}")
        print(f"    Taula elegibilitat : {len(dossier.eligibility_table)} criteris avaluats")
        if dossier.missing_information_questions:
            print(f"    Preguntes clíniques: {len(dossier.missing_information_questions)} obertes")
        _ok(f"Dossier generat en {_timing(elapsed)}")

        if save:
            demo_dir = _REPO / "data" / "dossiers" / "demo"
            demo_dir.mkdir(parents=True, exist_ok=True)
            out_path = demo_dir / f"{nct}.json"
            out_path.write_text(dossier.model_dump_json(indent=2))
            _ok(f"Dossier desat a {out_path.relative_to(_REPO)}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Demo de presentació del pipeline clínic d'assajos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--patient", type=str, default=None,
        help="Text clínic lliure del pacient (entre cometes)",
    )
    p.add_argument(
        "--topic", type=int, default=None, choices=list(TREC_TOPICS.keys()),
        help=f"Número de tòpic TREC predefinit ({', '.join(str(k) for k in TREC_TOPICS)})",
    )
    p.add_argument(
        "--candidate-cap", type=int, default=30,
        help="Màxim d'assajos candidats a recuperar (default: 30)",
    )
    p.add_argument(
        "--reasoning-trials", type=int, default=10,
        help="Quants trials passen pel raonament LLM (default: 10)",
    )
    p.add_argument(
        "--no-cache", action="store_true",
        help="Desactiva la caché (força re-avaluació completa)",
    )
    p.add_argument(
        "--save", action="store_true",
        help="Desa el dossier del millor assaig a data/dossiers/demo/",
    )
    p.add_argument(
        "--non-interactive", action="store_true",
        help="No espera ENTER entre passos (útil per a tests)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    use_cache = not args.no_cache
    interactive = not args.non_interactive

    # Resolve patient text
    if args.topic is not None:
        patient_text = TREC_TOPICS[args.topic]
    elif args.patient:
        patient_text = args.patient.strip()
    else:
        patient_text = DEFAULT_PATIENT

    print(f"\n{_SEP}")
    print(f"{_BOLD}{_CYAN}  UAB THE HACK × Deloitte — Clinical Trials AI Agent{_RESET}")
    print(f"{_BOLD}{_CYAN}  Demo interactiva del pipeline de 8 passos{_RESET}")
    print(_SEP)
    print(f"  Caché: {'activada' if use_cache else 'desactivada'} | "
          f"Candidats: {args.candidate_cap} | "
          f"Mode: {'interactiu' if interactive else 'no-interactiu'}")

    # PAS 1
    step1_pacient(patient_text)
    if interactive:
        _pause()

    # PAS 2
    profile = step2_normalitzacio(patient_text, use_cache)
    if interactive:
        _pause()

    # PAS 3
    step3_queries(profile, patient_text)
    if interactive:
        _pause()

    # PAS 4
    candidates = step4_candidats(profile, use_cache, args.candidate_cap)
    if not candidates:
        print(f"\n  {_RED}No s'han trobat candidats. Comprova la connexió a CT API.{_RESET}\n")
        sys.exit(1)
    if interactive:
        _pause()

    # PAS 5
    survivors = step5_filtre_dur(candidates, profile)
    if not survivors:
        _warn("Tots els candidats han estat eliminats pel filtre dur.")
        survivors = candidates[:3]  # fallback per demo
    if interactive:
        _pause()

    # PAS 6
    results = step6_raonament(survivors, profile, use_cache, reasoning_n=args.reasoning_trials)
    if not results:
        _err("No s'ha pogut completar el raonament.")
        sys.exit(1)
    if interactive:
        _pause()

    # PAS 7
    results = step7_nei(results, profile, use_cache)
    if interactive:
        _pause()

    # PAS 8
    step8_ranking(results, profile, use_cache, args.save)

    print(f"\n{_SEP}")
    print(f"{_BOLD}{_GREEN}  Demo completada!{_RESET}")
    print(_SEP)
    print()


if __name__ == "__main__":
    main()
