#!/usr/bin/env python3
"""
Evaluate the clinical trial matching pipeline on TREC 2021 and/or 2022.

Usage:
  python scripts/evaluate.py                    # 10 topics, both years
  python scripts/evaluate.py --topics 5         # 5 topics each year
  python scripts/evaluate.py --year 2021        # one year only
  python scripts/evaluate.py --topics 1,3,5     # specific topic IDs
  python scripts/evaluate.py --quick            # 3 topics, fast test
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _check_env() -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT.parent / ".env")  # fallback for dev setups
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.")
        print("  Create a .env file in the project root:")
        print("    OPENAI_API_KEY=sk-...")
        sys.exit(1)


def _check_data(year: int) -> None:
    d = ROOT / "data" / "trec" / str(year)
    for fname in ("topics.xml", "qrels.txt"):
        p = d / fname
        if not p.exists():
            print(f"ERROR: TREC {year} data not found at {p}")
            print(f"  Place topics.xml and qrels.txt in data/trec/{year}/")
            sys.exit(1)


def _stream_pipeline(cmd: list[str]) -> tuple[int, float]:
    """Run cmd, stream output to terminal in real-time, return (returncode, cost_usd)."""
    cost = 0.0
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        m = re.search(r"LLM cost \(est\.\)\s*:\s*\$([0-9.]+)", line)
        if m:
            cost = float(m.group(1))
    proc.wait()
    return proc.returncode, cost


def _compute_metrics(pred_path: Path, qrels_path: Path) -> dict:
    """Load predictions JSON, compute T1/T2/T3 from qrels."""
    import run_full_pipeline as rfp
    predictions = json.loads(pred_path.read_text())
    qrels = rfp.load_qrels(qrels_path)
    t2, _ = rfp.compute_t2(predictions, qrels)
    run = {
        str(e["patient_id"]): {t["nct_id"]: t["score"] for t in e.get("ranked_trials", [])}
        for e in predictions
    }
    ret = rfp.compute_retrieval_metrics(run, qrels)
    t1, t3 = ret["recall_20"], ret["ndcg_cut_10"]
    return {"t1": t1, "t2": t2, "t3": t3, "composite": 0.20 * t1 + 0.30 * t2 + 0.25 * t3}


def _print_table(rows: list[dict]) -> None:
    print()
    print("┌──────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐")
    print("│ Dataset      │ Topics   │ Recall@20│ Micro-F1 │ NDCG@10  │ Composite│")
    print("├──────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤")
    for r in rows:
        print(
            f"│ {r['label']:<12} │ {r['n']:<8} │ {r['t1']:<8.4f} │"
            f" {r['t2']:<8.4f} │ {r['t3']:<8.4f} │ {r['composite']:<8.4f}│"
        )
    print("└──────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘")


def _save_report(rows: list[dict], cost: float, elapsed: float) -> None:
    path = ROOT / "docs" / "evaluation_report.md"
    lines = [
        "# Evaluation Report\n",
        "| Dataset | Topics | Recall@20 | Micro-F1 | NDCG@10 | Composite |",
        "|---------|--------|-----------|----------|---------|-----------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['n']} | {r['t1']:.4f}"
            f" | {r['t2']:.4f} | {r['t3']:.4f} | {r['composite']:.4f} |"
        )
    lines.append(f"\nTotal cost: ${cost:.2f} | Time: {elapsed / 60:.1f} min")
    path.write_text("\n".join(lines) + "\n")
    print(f"Report saved → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate clinical trial pipeline on TREC data")
    parser.add_argument("--year", choices=["2021", "2022", "both"], default="both")
    parser.add_argument("--topics", default="10",
                        help="Number of topics or comma-separated IDs (default: 10)")
    parser.add_argument("--quick", action="store_true", help="Alias for --topics 3")
    parser.add_argument("--no-dual", action="store_true",
                        help="Use gpt-4o-mini only (already the default)")
    parser.add_argument("--mode", choices=["combined", "hybrid", "api"], default="hybrid")
    args = parser.parse_args()

    _check_env()

    topics_s = "3" if args.quick else args.topics
    if "," in topics_s:
        topic_ids: list[int] | None = [int(x.strip()) for x in topics_s.split(",")]
        n_display = len(topic_ids)
    else:
        topic_ids = None
        n_display = int(topics_s)

    years = [2021, 2022] if args.year == "both" else [int(args.year)]
    for year in years:
        _check_data(year)

    t_start = time.time()
    rows: list[dict] = []
    total_cost = 0.0

    pipeline_script = str(ROOT / "scripts" / "run_full_pipeline.py")

    for year in years:
        print(f"\n{'=' * 60}")
        print(f"  TREC {year} — {n_display} topic(s), mode={args.mode}")
        print(f"{'=' * 60}")

        base_cmd = [
            sys.executable, pipeline_script,
            "--year", str(year),
            "--retrieval-mode", args.mode,
        ]

        pred_path = ROOT / "data" / "predictions" / f"full_pipeline_{year}.json"

        if topic_ids is not None:
            all_preds: list[dict] = []
            for tid in topic_ids:
                cmd = base_cmd + ["--topic", str(tid)]
                rc, cost = _stream_pipeline(cmd)
                total_cost += cost
                if rc != 0:
                    print(f"WARNING: pipeline exited {rc} for topic {tid}")
                    continue
                if pred_path.exists():
                    data = json.loads(pred_path.read_text())
                    all_preds.extend(data if isinstance(data, list) else [data])
            pred_path.write_text(json.dumps(all_preds, indent=2))
        else:
            cmd = base_cmd + ["--topics-limit", str(n_display)]
            rc, cost = _stream_pipeline(cmd)
            total_cost += cost
            if rc != 0:
                print(f"WARNING: pipeline exited {rc} for TREC {year}")
                continue

        if not pred_path.exists() or pred_path.stat().st_size == 0:
            print(f"WARNING: predictions file missing for TREC {year}")
            continue

        qrels_path = ROOT / "data" / "trec" / str(year) / "qrels.txt"
        try:
            metrics = _compute_metrics(pred_path, qrels_path)
            n_actual = len(json.loads(pred_path.read_text()))
            rows.append({"label": f"TREC {year}", "n": n_actual, **metrics})
        except Exception as exc:
            print(f"WARNING: metric computation failed for TREC {year}: {exc}")

    elapsed = time.time() - t_start

    if rows:
        _print_table(rows)
        print(f"Total cost: ${total_cost:.2f} | Time: {elapsed / 60:.1f} min")
        _save_report(rows, total_cost, elapsed)
    else:
        print("\nNo results to display.")


if __name__ == "__main__":
    main()
