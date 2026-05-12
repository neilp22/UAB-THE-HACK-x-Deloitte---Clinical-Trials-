\# Score Breakdown



The ranking score is now decomposed into explicit components.



\## Components



\- inclusion\_met\_ratio

\- exclusion\_penalty

\- nei\_ratio

\- phase\_bonus

\- recruiting\_bonus

\- weighted\_inclusion

\- weighted\_exclusion

\- weighted\_phase

\- weighted\_recruiting

\- weighted\_nei\_penalty

\- raw\_score

\- final\_score



\## Purpose



This improves interpretability of the ranking without changing the original scoring philosophy.



The legacy `score\_trial()` function still returns a float for backwards compatibility.



The new `score\_trial\_breakdown()` function returns the full explanation structure.



The new `explain\_score\_breakdown()` function generates a deterministic explanation without using an LLM.



\## Pipeline output



The full pipeline can now include the following fields for each ranked trial:



\- `score`

\- `score\_breakdown`

\- `score\_explanation`



This makes the ranking easier to audit, explain and include in the final dossier.

