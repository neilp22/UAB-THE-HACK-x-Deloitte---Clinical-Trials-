#!/usr/bin/env python3
"""
Demo interactiva del Clinical Trial Matching Agent.
Mostra tot el pipeline pas a pas amb 5 pacients ficticis.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Afegeix src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsing.patient_normalizer import normalize_patient as _normalize_patient
from src.parsing.criteria_parser import parse_criteria
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.clinical_query_planner import extract_clinical_terms_llm
from src.matching.hard_filter import apply_hard_filter
from src.matching.eligibility_reasoner import evaluate_trial
from src.matching.label_deriver import derive_label
from src.ranking.scorer import score_trial
from src.output.nei_question_generator import generate_nei_question


class Colors:
    """Colors per terminal"""
    HEADER    = '\033[95m'
    OKBLUE    = '\033[94m'
    OKCYAN    = '\033[96m'
    OKGREEN   = '\033[92m'
    WARNING   = '\033[93m'
    FAIL      = '\033[91m'
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'


# ---------------------------------------------------------------------------
# Helpers de format
# ---------------------------------------------------------------------------

def print_header(text: str):
    print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}\n")

def print_phase(phase_num: int, phase_name: str):
    print(f"\n{Colors.OKBLUE}{'─'*80}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{Colors.BOLD}FASE {phase_num}: {phase_name}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{'─'*80}{Colors.ENDC}\n")

def print_success(text: str):
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_info(text: str):
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")

def print_warning(text: str):
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


# ---------------------------------------------------------------------------
# Càrrega de pacients
# ---------------------------------------------------------------------------

def load_demo_patients() -> List[Dict]:
    """Carrega els pacients de demo des de data/demo/demo_patients.json"""
    demo_file = Path("data/demo/demo_patients.json")

    if not demo_file.exists():
        print_warning(f"Demo patients file not found: {demo_file}")
        print_info("Creating demo patients file...")
        demo_file.parent.mkdir(parents=True, exist_ok=True)

        patients_data = {
            "patients": [
                {
                    "id": "DEMO-001",
                    "name": "Maria García",
                    "description": (
                        "67-year-old female with stage IIIB non-small cell lung carcinoma (NSCLC). "
                        "EGFR mutation positive (exon 19 deletion). Previously treated with platinum-based "
                        "chemotherapy and osimertinib, but disease progressed after 14 months. "
                        "ECOG performance status 1. No brain metastases. Adequate organ function. "
                        "Currently seeking second-line treatment options."
                    ),
                    "expected_trials_keywords": ["NSCLC", "EGFR", "lung cancer", "second-line", "platinum-resistant"]
                },
                {
                    "id": "DEMO-002",
                    "name": "Joan Martínez",
                    "description": (
                        "52-year-old male with newly diagnosed stage IV metastatic colorectal adenocarcinoma. "
                        "KRAS wild-type, BRAF wild-type. Liver and lung metastases present. No prior chemotherapy. "
                        "CEA elevated at 125 ng/mL. Good performance status (ECOG 0). Considering first-line "
                        "systemic therapy with biological agents."
                    ),
                    "expected_trials_keywords": ["colorectal cancer", "metastatic", "KRAS wild-type", "first-line", "biological therapy"]
                },
                {
                    "id": "DEMO-003",
                    "name": "Anna Rodríguez",
                    "description": (
                        "45-year-old female with triple-negative breast cancer (ER-, PR-, HER2-). "
                        "Stage IIA, T2N0M0. Tumor size 3.2 cm. BRCA1 mutation carrier. Completed neoadjuvant "
                        "chemotherapy with residual disease. Planning adjuvant therapy. High risk of recurrence. "
                        "Family history of breast and ovarian cancer."
                    ),
                    "expected_trials_keywords": ["triple-negative breast cancer", "BRCA1", "adjuvant", "high-risk", "TNBC"]
                },
                {
                    "id": "DEMO-004",
                    "name": "Pau López",
                    "description": (
                        "61-year-old male with advanced melanoma, BRAF V600E mutation positive. "
                        "Stage IV with multiple skin and lymph node metastases. Previously treated with "
                        "BRAF/MEK inhibitor combination (dabrafenib + trametinib) for 8 months, now progressed. "
                        "LDH elevated. No prior immunotherapy. ECOG performance status 1. "
                        "Seeking alternative treatment options."
                    ),
                    "expected_trials_keywords": ["melanoma", "BRAF V600E", "stage IV", "immunotherapy", "second-line"]
                },
                {
                    "id": "DEMO-005",
                    "name": "Laura Fernández",
                    "description": (
                        "38-year-old female with relapsed/refractory acute lymphoblastic leukemia (ALL). "
                        "Ph+ (Philadelphia chromosome positive), BCR-ABL1 fusion present. Failed two prior lines "
                        "of tyrosine kinase inhibitors (imatinib and dasatinib). Minimal residual disease detected. "
                        "Considered for allogeneic stem cell transplant but seeking bridging therapy. "
                        "T315I mutation detected. ECOG 2."
                    ),
                    "expected_trials_keywords": ["acute lymphoblastic leukemia", "Philadelphia positive", "BCR-ABL", "T315I", "relapsed refractory"]
                }
            ]
        }

        with open(demo_file, 'w') as f:
            json.dump(patients_data, f, indent=2)

        print_success(f"Created demo patients file: {demo_file}")

    with open(demo_file) as f:
        data = json.load(f)

    return data['patients']


# ---------------------------------------------------------------------------
# Pipeline principal (un pacient)
# ---------------------------------------------------------------------------

def demo_patient_pipeline(
    patient: Dict,
    retrieval_mode: str = 'hybrid',
    candidate_cap: int = 50,
    interactive: bool = True
) -> Dict:
    """
    Executa tot el pipeline per un pacient i mostra cada fase.

    Args:
        patient:        Dict amb info del pacient.
        retrieval_mode: 'hybrid' o 'ct_api_expanded'.
        candidate_cap:  Nombre màxim de candidates.
        interactive:    Si True, fa pauses amb input(). Si False, s'executa sol.

    Returns:
        Dict amb resultats de cada fase.
    """

    def pause(msg: str = "Press ENTER to continue..."):
        if interactive:
            input(f"\n{Colors.OKCYAN}{msg}{Colors.ENDC}")

    results = {
        'patient_id':   patient['id'],
        'patient_name': patient['name'],
        'phases':       {},
        'total_time':   0,
        'total_cost':   0
    }

    start_time = time.time()

    # ── FASE 0: Presentació ──────────────────────────────────────────────────
    print_header(f"DEMO: {patient['name']} ({patient['id']})")
    print(f"{Colors.BOLD}Patient Description:{Colors.ENDC}")
    print(f"{patient['description']}\n")
    print_info(f"Expected trial keywords: {', '.join(patient['expected_trials_keywords'])}")
    pause("Press ENTER to start pipeline...")

    # ── FASE 1: Patient Normalization ────────────────────────────────────────
    print_phase(1, "PATIENT NORMALIZATION (LLM)")
    print_info("Converting free text to structured PatientProfile...")

    t0 = time.time()
    patient_profile = _normalize_patient(patient['description'])
    phase1_time    = time.time() - t0

    print_success("Patient profile extracted!\n")
    print(f"{Colors.BOLD}Extracted Profile:{Colors.ENDC}")
    print(f"  Age:              {Colors.OKGREEN}{patient_profile.age}{Colors.ENDC}")
    print(f"  Gender:           {Colors.OKGREEN}{patient_profile.gender}{Colors.ENDC}")
    print(f"  Conditions:       {Colors.OKGREEN}{patient_profile.conditions}{Colors.ENDC}")
    print(f"  Prior treatments: {Colors.OKGREEN}{patient_profile.prior_treatments}{Colors.ENDC}")
    print(f"  History:          {Colors.OKGREEN}{patient_profile.relevant_history[:2]}{Colors.ENDC}")
    print(f"\n⏱  Time: {phase1_time:.2f}s")

    results['phases']['normalization'] = {
        'time': phase1_time,
        'profile': {
            'age':              patient_profile.age,
            'gender':           patient_profile.gender,
            'conditions':       patient_profile.conditions,
            'prior_treatments': patient_profile.prior_treatments,
        }
    }
    pause("Press ENTER to continue to query expansion...")

    # ── FASE 2: Query Expansion ──────────────────────────────────────────────
    print_phase(2, "QUERY EXPANSION (LLM)")
    print_info("Extracting precise clinical terms for search...")

    t0 = time.time()
    llm_terms   = extract_clinical_terms_llm(patient['description'])
    phase2_time = time.time() - t0

    if llm_terms:
        print_success(f"Extracted {len(llm_terms)} clinical terms!\n")
        print(f"{Colors.BOLD}LLM Expanded Terms:{Colors.ENDC}")
        for i, term in enumerate(llm_terms, 1):
            print(f"  {i}. {Colors.OKGREEN}{term}{Colors.ENDC}")
    else:
        print_warning("No LLM expansion terms generated")

    print(f"\n⏱  Time: {phase2_time:.2f}s")

    results['phases']['query_expansion'] = {
        'time':  phase2_time,
        'terms': llm_terms
    }
    pause("Press ENTER to continue to retrieval...")

    # ── FASE 3: Candidate Retrieval ──────────────────────────────────────────
    print_phase(3, f"CANDIDATE RETRIEVAL (mode={retrieval_mode}, cap={candidate_cap})")

    t0 = time.time()

    if retrieval_mode == 'hybrid':
        print_info("Using Hybrid Retriever (BM25 + BioBERT over TREC index)...")
        from src.retrieval.semantic_retriever import SemanticRetriever
        import src.config as _cfg
        _emb_path = _cfg.CACHE_DIR / "embeddings" / "trial_embeddings.npy"
        _ids_path = _cfg.CACHE_DIR / "embeddings" / "trial_nct_ids.pkl"
        _idx_path = str(_cfg.CACHE_DIR / "bm25_trec2021_index.pkl")
        _sem      = SemanticRetriever(_emb_path, _ids_path, _cfg.CACHE_DIR / "trial_data")
        retriever  = HybridRetriever(_sem, _idx_path)
        nct_ids    = retriever.query(patient['description'], top_k=candidate_cap)
        candidates = [t for nid in nct_ids if (t := retriever.get_trial(nid))]

    elif retrieval_mode == 'ct_api_expanded':
        print_info("Using CT API with expanded queries...")
        from src.retrieval.query_builder import build_queries_combined
        import diskcache, src.config as _cfg
        _cache    = diskcache.Cache(str(_cfg.CACHE_DIR / "query_builder"))
        nct_ids   = build_queries_combined(
            patient_text=patient['description'],
            profile=patient_profile,
            max_candidates=candidate_cap,
            use_cache=True,
            cache=_cache,
        )
        from src.retrieval.ct_client import search_trials as _st
        candidates = [t for nid in nct_ids
                      if (t := _cache.get(f"trial:{nid}")) and isinstance(t, dict)]
        if not candidates:
            candidates = []

    else:
        raise ValueError(f"Unknown retrieval_mode: {retrieval_mode}")

    phase3_time = time.time() - t0

    print_success(f"Retrieved {len(candidates)} candidate trials!\n")
    print(f"{Colors.BOLD}Top 5 Retrieved Trials:{Colors.ENDC}")
    for i, trial in enumerate(candidates[:5], 1):
        nct_id = trial.get('nct_id') or trial.get('id', 'N/A')
        title  = trial.get('title', 'N/A')[:80]
        print(f"  {i}. {Colors.OKGREEN}{nct_id}{Colors.ENDC}: {title}...")

    print(f"\n⏱  Time: {phase3_time:.2f}s")

    results['phases']['retrieval'] = {
        'time':                phase3_time,
        'mode':                retrieval_mode,
        'candidates_retrieved': len(candidates),
        'top_5': [
            {'nct_id': c.get('nct_id') or c.get('id'), 'title': c.get('title', 'N/A')}
            for c in candidates[:5]
        ]
    }
    print_info(f"Pre-parsing eligibility criteria for {len(candidates)} trials (cached after first run)...")
    for i, trial in enumerate(candidates, 1):
        trial['_parsed_criteria'] = parse_criteria(trial.get('eligibility_criteria', ''))
        print(f"  Parsed {i}/{len(candidates)}...", end='\r')
    print()

    pause("Press ENTER to continue to hard filtering...")

    # ── FASE 4: Hard Filtering ───────────────────────────────────────────────
    print_phase(4, "HARD FILTERING (Deterministic)")
    print_info("Applying deterministic age & gender filters...")

    t0 = time.time()
    filtered_candidates = []
    for trial in candidates:
        parsed_tmp = trial['_parsed_criteria']
        excl_det = [c for c in parsed_tmp.criteria if c.type == 'exclusion' and c.deterministic]
        if apply_hard_filter(patient_profile, excl_det):
            filtered_candidates.append(trial)
    phase4_time = time.time() - t0

    eliminated     = len(candidates) - len(filtered_candidates)
    eliminated_pct = (eliminated / len(candidates) * 100) if candidates else 0

    if eliminated > 0:
        print_warning(f"Eliminated {eliminated} trials ({eliminated_pct:.1f}%)")
    else:
        print_success("All candidates passed hard filters")

    print_success(f"Remaining candidates: {len(filtered_candidates)}")
    print(f"\n⏱  Time: {phase4_time:.2f}s")

    results['phases']['hard_filter'] = {
        'time':          phase4_time,
        'eliminated':    eliminated,
        'eliminated_pct': eliminated_pct,
        'remaining':     len(filtered_candidates)
    }
    pause("Press ENTER to continue to eligibility matching...")

    # ── FASE 5: Eligibility Matching ─────────────────────────────────────────
    print_phase(5, "ELIGIBILITY MATCHING (LLM)")
    print_info(f"Evaluating eligibility for {len(filtered_candidates)} trials...")
    print_info("This may take a few minutes...\n")

    t0 = time.time()

    matched_trials = []
    eval_cap       = min(len(filtered_candidates), 10)  # Limita a 10 per demo

    for idx, trial in enumerate(filtered_candidates[:eval_cap], 1):
        nct_id = trial.get('nct_id') or trial.get('id')
        print(f"  [{idx}/{eval_cap}] Processing {nct_id}...", end='\r')

        parsed         = trial['_parsed_criteria']
        typed_verdicts = evaluate_trial(patient_profile, parsed.criteria, nct_id=nct_id)

        inc_met     = sum(1 for t, v in typed_verdicts if t == 'inclusion' and v.verdict == 'MET')
        inc_not_met = sum(1 for t, v in typed_verdicts if t == 'inclusion' and v.verdict == 'NOT_MET')
        inc_nei     = sum(1 for t, v in typed_verdicts if t == 'inclusion' and v.verdict == 'NEI')
        excl_viol   = sum(1 for t, v in typed_verdicts if t == 'exclusion' and v.verdict == 'NOT_MET')
        summary     = {'inclusion_met': inc_met, 'inclusion_not_met': inc_not_met,
                       'inclusion_nei': inc_nei, 'exclusion_violations': excl_viol}
        label = derive_label(summary)

        matched_trials.append({
            'trial':          trial,
            'typed_verdicts': typed_verdicts,
            'label':          label,
        })

    print()  # Nova línia després del progress
    phase5_time = time.time() - t0

    print_success(f"Evaluated {len(matched_trials)} trials!\n")

    label_counts = {}
    for mt in matched_trials:
        label_counts[mt['label']] = label_counts.get(mt['label'], 0) + 1

    print(f"{Colors.BOLD}Label Distribution:{Colors.ENDC}")
    for label, count in label_counts.items():
        pct   = count / len(matched_trials) * 100
        color = Colors.OKGREEN if label == 'MET' else Colors.WARNING if label == 'NEI' else Colors.FAIL
        print(f"  {color}{label}: {count} ({pct:.1f}%){Colors.ENDC}")

    print(f"\n⏱  Time: {phase5_time:.2f}s")

    results['phases']['eligibility'] = {
        'time':               phase5_time,
        'evaluated':          len(matched_trials),
        'label_distribution': label_counts
    }
    pause("Press ENTER to continue to scoring...")

    # ── FASE 6: Scoring & Ranking ────────────────────────────────────────────
    print_phase(6, "SCORING & RANKING")
    print_info("Computing weighted scores for ranking...")

    t0 = time.time()

    for mt in matched_trials:
        phase_list = mt['trial'].get('phases') or []
        metadata = {
            'phase':  phase_list[0] if phase_list else mt['trial'].get('phase', ''),
            'status': mt['trial'].get('status', ''),
        }
        mt['score'] = score_trial(mt['typed_verdicts'], metadata)

    matched_trials.sort(key=lambda x: x['score'], reverse=True)
    phase6_time = time.time() - t0

    print_success("Trials ranked!\n")
    print(f"{Colors.BOLD}Top 5 Ranked Trials:{Colors.ENDC}")
    for i, mt in enumerate(matched_trials[:5], 1):
        trial  = mt['trial']
        nct_id = trial.get('nct_id') or trial.get('id')
        title  = trial.get('title', 'N/A')[:60]
        score  = mt['score']
        label  = mt['label']
        lcolor = Colors.OKGREEN if label == 'MET' else Colors.WARNING if label == 'NEI' else Colors.FAIL
        print(f"  {i}. {Colors.OKGREEN}{nct_id}{Colors.ENDC} | Score: {score:.3f} | {lcolor}{label}{Colors.ENDC}")
        print(f"     {title}...")

    print(f"\n⏱  Time: {phase6_time:.2f}s")

    results['phases']['scoring'] = {
        'time':  phase6_time,
        'top_5': [
            {
                'nct_id': mt['trial'].get('nct_id') or mt['trial'].get('id'),
                'score':  mt['score'],
                'label':  mt['label'],
                'title':  mt['trial'].get('title', 'N/A')
            }
            for mt in matched_trials[:5]
        ]
    }
    pause("Press ENTER to continue to NEI questions...")

    # ── FASE 7: NEI Questions Generation ─────────────────────────────────────
    print_phase(7, "NEI QUESTIONS GENERATION (LLM)")
    print_info("Generating clarifying questions for NEI criteria...")

    t0         = time.time()
    nei_trials = [mt for mt in matched_trials[:5] if mt['label'] == 'NEI']

    if nei_trials:
        print_success(f"Found {len(nei_trials)} trials with NEI verdicts\n")

        for mt in nei_trials[:2]:  # Primers 2 per demo
            trial  = mt['trial']
            nct_id = trial.get('nct_id') or trial.get('id')
            print(f"{Colors.BOLD}Trial: {nct_id}{Colors.ENDC}")

            nei_pairs = [(crit_type, v) for crit_type, v in mt['typed_verdicts'] if v.verdict == 'NEI']

            for crit_type, verdict in nei_pairs[:2]:
                question = generate_nei_question(
                    verdict=verdict,
                    criterion_text=verdict.criterion_text if hasattr(verdict, 'criterion_text') else '',
                    trial_title=trial.get('title', ''),
                    profile=patient_profile,
                )
                print(f"  {Colors.WARNING}?{Colors.ENDC} {question}")

            print()
    else:
        print_info("No trials with NEI verdicts in top 5")

    phase7_time = time.time() - t0
    print(f"⏱  Time: {phase7_time:.2f}s")

    results['phases']['nei_questions'] = {
        'time':            phase7_time,
        'nei_trials_count': len(nei_trials)
    }

    # ── Resum final del pacient ───────────────────────────────────────────────
    total_time            = time.time() - start_time
    results['total_time'] = total_time

    print_header("PIPELINE COMPLETED")
    print(f"{Colors.BOLD}Summary for {patient['name']}:{Colors.ENDC}\n")
    print(f"  Initial candidates:  {len(candidates)}")
    print(f"  After hard filter:   {len(filtered_candidates)}")
    print(f"  Evaluated:           {len(matched_trials)}")

    if matched_trials:
        top = matched_trials[0]
        print(f"  Top recommendation:  {top['trial'].get('nct_id') or top['trial'].get('id')}")
        print(f"  Top score:           {top['score']:.3f}")
        print(f"  Top label:           {top['label']}")

    print(f"\n  {Colors.OKGREEN}Total time: {total_time:.2f}s{Colors.ENDC}")

    return results


# ---------------------------------------------------------------------------
# Orquestrador multi-pacient
# ---------------------------------------------------------------------------

def run_demo(
    retrieval_mode: str = 'hybrid',
    candidate_cap:  int = 50,
    num_patients:   int = 5,
    interactive:    bool = True
):
    """Executa demo completa amb múltiples pacients."""

    print_header("CLINICAL TRIAL MATCHING AGENT - INTERACTIVE DEMO")

    print(f"{Colors.BOLD}Configuration:{Colors.ENDC}")
    print(f"  Retrieval mode:    {Colors.OKGREEN}{retrieval_mode}{Colors.ENDC}")
    print(f"  Candidate cap:     {Colors.OKGREEN}{candidate_cap}{Colors.ENDC}")
    print(f"  Number of patients:{Colors.OKGREEN}{num_patients}{Colors.ENDC}")
    print(f"  Interactive:       {Colors.OKGREEN}{interactive}{Colors.ENDC}")

    patients = load_demo_patients()[:num_patients]

    print(f"\n{Colors.BOLD}Demo patients loaded:{Colors.ENDC}")
    for i, p in enumerate(patients, 1):
        print(f"  {i}. {p['name']} ({p['id']})")

    if interactive:
        input(f"\n{Colors.OKCYAN}Press ENTER to start demo...{Colors.ENDC}")

    all_results = []

    for i, patient in enumerate(patients, 1):
        print(f"\n\n{Colors.HEADER}{'#'*80}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}PATIENT {i}/{len(patients)}{Colors.ENDC}")
        print(f"{Colors.HEADER}{'#'*80}{Colors.ENDC}")

        results = demo_patient_pipeline(patient, retrieval_mode, candidate_cap, interactive)
        all_results.append(results)

        if interactive and i < len(patients):
            input(f"\n{Colors.OKCYAN}Press ENTER to continue to next patient...{Colors.ENDC}")

    # ── Resum global ──────────────────────────────────────────────────────────
    print_header("DEMO COMPLETED - GLOBAL SUMMARY")

    total_time = sum(r['total_time'] for r in all_results)
    avg_time   = total_time / len(all_results)

    print(f"{Colors.BOLD}Statistics:{Colors.ENDC}")
    print(f"  Patients processed:         {len(all_results)}")
    print(f"  Total time:                 {total_time:.2f}s ({total_time/60:.1f} min)")
    print(f"  Average time per patient:   {avg_time:.2f}s")

    output_file = Path(f"demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': {
                'retrieval_mode': retrieval_mode,
                'candidate_cap':  candidate_cap
            },
            'results': all_results
        }, f, indent=2)

    print(f"\n{Colors.OKGREEN}Results saved to: {output_file}{Colors.ENDC}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Interactive demo of Clinical Trial Matching Agent')
    parser.add_argument('--mode', type=str, default='hybrid',
                        choices=['hybrid', 'ct_api_expanded'],
                        help='Retrieval mode')
    parser.add_argument('--cap', type=int, default=50,
                        help='Candidate cap')
    parser.add_argument('--patients', type=int, default=5,
                        help='Number of demo patients (max 5)')
    args = parser.parse_args()

    run_demo(
        retrieval_mode=args.mode,
        candidate_cap=args.cap,
        num_patients=min(args.patients, 5),
        interactive=True      # versió interactiva amb pauses
    )


if __name__ == '__main__':
    main()
