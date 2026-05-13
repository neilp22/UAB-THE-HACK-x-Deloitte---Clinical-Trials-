Changes Made

  src/retrieval/bm25_retriever.py

  _trial_text() — removed eligibility_criteria from the BM25 document corpus. The index now only uses title, brief_summary, and conditions. Eligibility text was dominating BM25 scores
   with noisy boolean lists.

  src/matching/label_deriver.py

  derive_label() — added if inclusion_not_met > inclusion_met: return "NOT_MET" before the nei_ratio check. Previously a patient with 3 inclusion misses and 1 inclusion hit could
  still reach MET.

  src/matching/eligibility_reasoner.py

  _SYSTEM + _TRIAL_SYSTEM — replaced "Default to NEI." with "Return NEI only when required information is explicitly missing." in both single-criterion and batched-trial system
  prompts.

  src/retrieval/query_builder.py

  retrieve_candidates(), _fetch() inside build_queries_clinical(), build_queries_combined() LLM expansion loop — each now writes cache.set(f"trial:{nct_id}", trial) when a new NCT ID
  is seen, so downstream lookup via qb_cache.get(f"trial:{nid}") hits without a second API call.

  scripts/run_full_pipeline.py

  ┌──────────────────────────────────────────────┬─────────────────────────────────────┐
  │                    Change                    │              Location               │
  ├──────────────────────────────────────────────┼─────────────────────────────────────┤
  │ import pytrec_eval moved lazy                │ inside compute_retrieval_metrics()  │
  ├──────────────────────────────────────────────┼─────────────────────────────────────┤
  │ from sklearn.metrics import … moved lazy     │ inside compute_t2()                 │
  ├──────────────────────────────────────────────┼─────────────────────────────────────┤
  │ Hard-filter score 0.0 → -0.5                 │ _process_trial() return dict        │
  ├──────────────────────────────────────────────┼─────────────────────────────────────┤
  │ BM25 query uses " ".join(profile.conditions) │ CT API fallback path in run_topic() │
  ├──────────────────────────────────────────────┼─────────────────────────────────────┤
  │ --mode {benchmark,demo} added                │ main() parser                       │
  ├──────────────────────────────────────────────┼─────────────────────────────────────┤
  │ --patient-text TEXT added                    │ main() parser                       │
  ├──────────────────────────────────────────────┼─────────────────────────────────────┤
  │ Demo early-return if args.mode == "demo"     │ main()                              │
  ├──────────────────────────────────────────────┼─────────────────────────────────────┤
  │ _run_demo() function                         │ new, before main()                  │
  └──────────────────────────────────────────────┴─────────────────────────────────────┘

  CLI usage:
  # benchmark (unchanged — default)
  python scripts/run_full_pipeline.py --year 2021 --retrieval-mode hybrid

  # demo — live CT API, no TREC files required
  python scripts/run_full_pipeline.py --mode demo \
    --patient-text "67-year-old female with NSCLC, EGFR exon 19 deletion, ECOG 1"