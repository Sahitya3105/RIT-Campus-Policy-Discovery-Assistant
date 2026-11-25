#!/usr/bin/env python3
"""
Benchmark: Raw Query vs Multi-HyDE Hybrid Search
Computes Recall@1, Recall@3, MRR, and latency for both approaches.

Offline Mode (no LLM key): Tests embedding-only retrieval (HyDE uses empty hypotheses).
Online Mode (LLM_API_KEY set): Full Multi-HyDE pipeline.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add parent/src to path so we can import the retriever
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def load_test_queries(path: str) -> list[dict]:
    """Load test_queries.json and filter out trap queries."""
    with open(path, "r", encoding="utf-8") as fh:
        queries = json.load(fh)
    # Only keep non-trap queries for metric calculation
    return [q for q in queries if not q.get("is_trap", False)]


import re

def _is_clause_match(retrieved_id: str, gt: str) -> bool:
    """Matches clause IDs even when represented as ranges like 'Clause 11.1 – 11.5'."""
    r_clean = retrieved_id.lower().strip()
    g_clean = gt.lower().strip()
    if r_clean == g_clean or g_clean in r_clean or r_clean in g_clean:
        return True
    gt_nums = re.findall(r"\d+(?:\.\d+)?", g_clean)
    ret_nums = re.findall(r"\d+(?:\.\d+)?", r_clean)
    return any(n in ret_nums for n in gt_nums)


def recall_at_k(retrieved_clause_ids: list[str], ground_truth: list[str], k: int) -> float:
    """Return 1.0 if any ground-truth clause appears in top-k results, else 0.0."""
    top_k = retrieved_clause_ids[:k]
    for gt in ground_truth:
        for rid in top_k:
            if _is_clause_match(rid, gt):
                return 1.0
    return 0.0


def mean_reciprocal_rank(retrieved_clause_ids: list[str], ground_truth: list[str]) -> float:
    """Return 1/rank of first correct hit in top-3, or 0.0 if none found."""
    top_3 = retrieved_clause_ids[:3]
    for rank, clause_id in enumerate(top_3, start=1):
        for gt in ground_truth:
            if _is_clause_match(clause_id, gt):
                return 1.0 / rank
    return 0.0


def extract_clause_ids_from_results(results: list) -> list[str]:
    """Extract clause_id strings from a list of retrieval result objects/dicts."""
    clause_ids = []
    for item in results:
        # Support both dict and object with .payload attribute
        if isinstance(item, dict):
            payload = item.get("payload", item)
            cid = payload.get("clause_id") or payload.get("clause_id", None)
        else:
            payload = getattr(item, "payload", {}) or {}
            cid = payload.get("clause_id", None)
        if cid:
            clause_ids.append(str(cid))
    return clause_ids


def extract_clause_ids_from_hyde(hyde_result) -> list[str]:
    """
    Extract clause_ids from the hyde_hybrid_search return value.
    Expected shape: {"real_chunks": [...], "hypothetical_docs": [...]}
    Each chunk is a dict or ScoredPoint with payload containing clause_id.
    """
    if isinstance(hyde_result, dict):
        chunks = hyde_result.get("real_chunks", [])
    else:
        chunks = list(hyde_result) if hyde_result else []
    return extract_clause_ids_from_results(chunks)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def print_results_table(
    baseline: dict,
    hyde: dict,
) -> None:
    """Print a formatted Unicode box table of benchmark results."""
    b = baseline
    h = hyde

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║          MSRIT Policy RAG Benchmark Results          ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║ Metric          │ Raw Baseline  │ Multi-HyDE Hybrid  ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║ Recall@1        │ {b['recall_at_1']:<13.3f} │ {h['recall_at_1']:<18.3f} ║")
    print(f"║ Recall@3        │ {b['recall_at_3']:<13.3f} │ {h['recall_at_3']:<18.3f} ║")
    print(f"║ MRR             │ {b['mrr']:<13.3f} │ {h['mrr']:<18.3f} ║")
    print(f"║ Avg Latency(ms) │ {b['avg_latency_ms']:<13.0f} │ {h['avg_latency_ms']:<18.0f} ║")
    print("╚══════════════════════════════════════════════════════╝\n")


def save_results_json(
    benchmark_dir: Path,
    mode: str,
    baseline: dict,
    hyde: dict,
    per_query: list,
) -> None:
    """Persist results to benchmark/results.json."""
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "num_queries": 20,
        "baseline": baseline,
        "hyde": hyde,
        "improvement": {
            "recall_at_1": round(hyde["recall_at_1"] - baseline["recall_at_1"], 4),
            "recall_at_3": round(hyde["recall_at_3"] - baseline["recall_at_3"], 4),
            "mrr": round(hyde["mrr"] - baseline["mrr"], 4),
        },
        "per_query": per_query,
    }
    out_path = benchmark_dir / "results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"Results saved → {out_path}")


def generate_chart(benchmark_dir: Path, baseline: dict, hyde: dict) -> None:
    """Generate side-by-side bar chart and save as retrieval_comparison.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping chart generation.")
        return

    metrics = ["Recall@1", "Recall@3", "MRR"]
    baseline_vals = [baseline["recall_at_1"], baseline["recall_at_3"], baseline["mrr"]]
    hyde_vals = [hyde["recall_at_1"], hyde["recall_at_3"], hyde["mrr"]]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.set_style = None  # will use default

    bars1 = ax.bar(x - width / 2, baseline_vals, width, label="Raw Baseline", color="#4472C4")
    bars2 = ax.bar(x + width / 2, hyde_vals, width, label="Multi-HyDE Hybrid", color="#ED7D31")

    # Value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("MSRIT Policy RAG: Raw Baseline vs Multi-HyDE Hybrid", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out_path = benchmark_dir / "retrieval_comparison.png"
    fig.savefig(str(out_path))
    plt.close(fig)
    print(f"Chart saved → {out_path}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation() -> None:
    benchmark_dir = Path(__file__).parent
    queries_path = benchmark_dir / "test_queries.json"

    # 1. Load queries (non-trap only)
    print(f"Loading test queries from {queries_path} …")
    queries = load_test_queries(str(queries_path))
    print(f"  {len(queries)} non-trap queries loaded for evaluation.")

    # 2. Determine mode
    mode = "online" if os.environ.get("LLM_API_KEY") or os.environ.get("GOOGLE_API_KEY") else "offline"
    print(f"  Mode: {mode.upper()}")

    # 3. Import and initialise retriever
    try:
        from hyde_retriever import HyDERetriever  # type: ignore
    except ImportError as exc:
        print(
            f"\n[ERROR] Could not import HyDERetriever: {exc}\n"
            "Make sure src/hyde_retriever.py exists and dependencies are installed."
        )
        sys.exit(1)

    try:
        retriever = HyDERetriever()
    except Exception as exc:
        print(
            f"\n[ERROR] Failed to initialise HyDERetriever: {exc}\n"
            "Have you run the ingestion pipeline yet? "
            "(Qdrant collection and BM25 index must be built first.)"
        )
        sys.exit(1)

    # 4. Run evaluation loop
    baseline_records: list[dict] = []
    hyde_records: list[dict] = []
    per_query: list[dict] = []

    print(f"\nRunning evaluation on {len(queries)} queries …\n")

    for q in queries:
        qid = q["id"]
        query_text = q["query"]
        ground_truth = q["ground_truth_clauses"]

        # --- Baseline ---
        t0 = time.perf_counter()
        try:
            baseline_results = retriever.raw_baseline_search(query_text)
        except Exception as exc:
            print(f"[ERROR] raw_baseline_search failed for {qid}: {exc}")
            sys.exit(1)
        baseline_latency_ms = (time.perf_counter() - t0) * 1000

        baseline_ids = extract_clause_ids_from_results(baseline_results)

        b_r1 = recall_at_k(baseline_ids, ground_truth, 1)
        b_r3 = recall_at_k(baseline_ids, ground_truth, 3)
        b_mrr = mean_reciprocal_rank(baseline_ids, ground_truth)

        baseline_records.append({
            "recall_at_1": b_r1,
            "recall_at_3": b_r3,
            "mrr": b_mrr,
            "latency_ms": baseline_latency_ms,
        })

        # --- HyDE ---
        t0 = time.perf_counter()
        try:
            hyde_result = retriever.hyde_hybrid_search(query_text)
        except Exception as exc:
            print(f"[ERROR] hyde_hybrid_search failed for {qid}: {exc}")
            sys.exit(1)
        hyde_latency_ms = (time.perf_counter() - t0) * 1000

        hyde_ids = extract_clause_ids_from_hyde(hyde_result)

        h_r1 = recall_at_k(hyde_ids, ground_truth, 1)
        h_r3 = recall_at_k(hyde_ids, ground_truth, 3)
        h_mrr = mean_reciprocal_rank(hyde_ids, ground_truth)

        hyde_records.append({
            "recall_at_1": h_r1,
            "recall_at_3": h_r3,
            "mrr": h_mrr,
            "latency_ms": hyde_latency_ms,
        })

        per_query.append({
            "id": qid,
            "query": query_text,
            "ground_truth": ground_truth,
            "baseline": {
                "retrieved_ids": baseline_ids[:5],
                "recall_at_1": b_r1,
                "recall_at_3": b_r3,
                "mrr": b_mrr,
                "latency_ms": round(baseline_latency_ms, 2),
            },
            "hyde": {
                "retrieved_ids": hyde_ids[:5],
                "recall_at_1": h_r1,
                "recall_at_3": h_r3,
                "mrr": h_mrr,
                "latency_ms": round(hyde_latency_ms, 2),
            },
        })

        status = "✓" if h_r1 > 0 else "✗"
        print(f"  [{status}] {qid}: baseline R@1={b_r1:.1f} MRR={b_mrr:.3f} | hyde R@1={h_r1:.1f} MRR={h_mrr:.3f}")

    # 5. Aggregate metrics
    n = len(queries)

    def avg(records, key):
        return round(sum(r[key] for r in records) / n, 4)

    baseline_agg = {
        "recall_at_1": avg(baseline_records, "recall_at_1"),
        "recall_at_3": avg(baseline_records, "recall_at_3"),
        "mrr": avg(baseline_records, "mrr"),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in baseline_records) / n, 2),
    }
    hyde_agg = {
        "recall_at_1": avg(hyde_records, "recall_at_1"),
        "recall_at_3": avg(hyde_records, "recall_at_3"),
        "mrr": avg(hyde_records, "mrr"),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in hyde_records) / n, 2),
    }

    # 6. Print table
    print_results_table(baseline_agg, hyde_agg)

    # 7. Save results.json
    save_results_json(benchmark_dir, mode, baseline_agg, hyde_agg, per_query)

    # 8. Generate chart
    generate_chart(benchmark_dir, baseline_agg, hyde_agg)


if __name__ == "__main__":
    run_evaluation()
