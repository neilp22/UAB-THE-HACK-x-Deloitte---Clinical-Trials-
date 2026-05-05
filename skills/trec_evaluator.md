# Skill: trec_evaluator

## Purpose
Run TREC Clinical Trials evaluation metrics against system predictions.

## Metrics
| Metric     | Tool          | Notes                          |
|------------|--------------|-------------------------------|
| Recall@20  | pytrec_eval  | T1 — retrieval coverage        |
| Micro-F1   | custom       | T2 — per-criterion MET/NOT/NEI |
| NDCG@10    | pytrec_eval  | T3 — ranking quality           |

## Data
- TREC 2021 qrels: data/trec/2021/qrels.txt (dev — tune here)
- TREC 2022 qrels: data/trec/2022/qrels.txt (validation — Day 7 only)

## Prediction Format
TREC run file: `<topic_id> Q0 <nct_id> <rank> <score> <system_name>`

## Usage
```bash
python scripts/evaluate.py --year 2021 --run output/predictions.run
```

## Status
[ ] Not implemented
