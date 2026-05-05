# TREC Clinical Trials Data

## Download Instructions

Run from repo root:
```bash
# TREC 2021 (development set — tune here)
curl -L --user-agent "Mozilla/5.0" https://trec.nist.gov/data/trials/topics2021.xml -o data/trec/2021/topics.xml
curl -L --user-agent "Mozilla/5.0" https://trec.nist.gov/data/trials/qrels2021.txt   -o data/trec/2021/qrels.txt

# TREC 2022 (validation — DO NOT USE until Day 7)
curl -L --user-agent "Mozilla/5.0" https://trec.nist.gov/data/trials/topics2022.xml -o data/trec/2022/topics.xml
curl -L --user-agent "Mozilla/5.0" https://trec.nist.gov/data/trials/qrels2022.txt   -o data/trec/2022/qrels.txt
```

## File Structure
```
data/trec/
├── 2021/
│   ├── topics.xml     — 75 patient profiles (XML)
│   └── qrels.txt      — 35,832 relevance judgments
└── 2022/
    ├── topics.xml     — 50 patient profiles (XML)
    └── qrels.txt      — ~50k relevance judgments
```

## Relevance Grades (qrels)
| Grade | Meaning |
|-------|---------|
| 0 | Not relevant |
| 1 | Excluded (patient is excluded by trial criteria but trial is relevant) |
| 2 | Eligible (patient meets all criteria) |

## IMPORTANT
- **2021 = development set**: tune all parameters here; inspect freely.
- **2022 = validation set**: touch ONLY on Day 7. Never tune on 2022 data.
