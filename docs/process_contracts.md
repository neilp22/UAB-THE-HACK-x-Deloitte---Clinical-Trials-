\# Process Contracts



This document defines the contracts between the main modules of the clinical trial matching agent.



The goal is to make the pipeline auditable, reproducible and easier to debug. Each module should have a clear responsibility, a well-defined input, a well-defined output, known failure modes and a direct relationship with the evaluation metrics.



\---



\## Global Pipeline



```text

Patient profile

→ patient normalization

→ query planning / retrieval

→ candidate trials

→ criteria parsing

→ hard filtering

→ eligibility reasoning

→ NEI question generation

→ scoring and ranking

→ dossier generation

→ predictions JSON

→ metrics

