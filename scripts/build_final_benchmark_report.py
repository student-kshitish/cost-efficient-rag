import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evaluation" / "results"

RETRIEVAL_FILE = RESULTS / "retrieval_benchmark.json"
ANSWER_FILE = RESULTS / "answer_quality_summary.json"
STORAGE_FILE = RESULTS / "scalability_estimates.json"

REPORT_JSON = RESULTS / "final_benchmark_report.json"
REPORT_MD = RESULTS / "final_benchmark_report.md"
COST_CSV = RESULTS / "managed_database_comparison.csv"


def read_json(path):
    if not path.exists():
        print(f"WARNING: Missing artifact: {path}")
        return None

    with path.open(encoding="utf-8") as f:
        return json.load(f)


def find_values(obj, wanted_keys, path=""):
    """Find metric values recursively without assuming a particular schema."""
    matches = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            current = f"{path}.{key}" if path else key

            if key.lower() in wanted_keys and isinstance(
                value, (int, float)
            ):
                matches.append((current, value))

            matches.extend(find_values(value, wanted_keys, current))

    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            matches.extend(
                find_values(value, wanted_keys, f"{path}[{index}]")
            )

    return matches


def fmt(value):
    if isinstance(value, (int, float)):
        return f"{value:,.2f}"
    return str(value)


def main():
    retrieval = read_json(RETRIEVAL_FILE)
    answers = read_json(ANSWER_FILE)
    storage = read_json(STORAGE_FILE)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": "Cost-Efficient RAG",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dimensions": 384,
        "vector_database": "LanceDB",
        "llm_provider": "Local Ollama",
        "llm_model": "llama3.2:3b",
        "retrieval_benchmark": retrieval,
        "answer_quality_collection": answers,
        "storage_estimates": storage,
        "managed_database_pricing_snapshot": {
            "currency": "USD",
            "pinecone": {
                "Builder_plan_minimum_monthly_usd": 20,
                "Standard_plan_minimum_monthly_usd": 50,
                "Enterprise_plan_minimum_monthly_usd": 500,
                "pricing_url": "https://www.pinecone.io/pricing/",
                "interpretation": (
                    "Published plan minimums; actual usage charges can "
                    "be higher. Not a quote for a given vector count."
                ),
            },
            "qdrant_cloud": {
                "free_tier": {
                    "monthly_price_usd": 0,
                    "vcpus": 0.5,
                    "ram_gb": 1,
                    "disk_gb": 4,
                },
                "paid_plan": "Usage-based resource billing",
                "pricing_url": "https://qdrant.tech/pricing/",
                "interpretation": (
                    "Paid cost depends on allocated compute, memory, "
                    "storage, backups, and usage. Obtain a deployment "
                    "estimate from the official calculator."
                ),
            },
        },
        "limitations": [
            "The current real dataset has only 18 indexed chunks.",
            "Large-scale storage values are analytical estimates.",
            "Local latency measurements do not represent 100K, 1M, or 10M vectors.",
            "Successful generation does not establish answer correctness.",
            "Managed plan minimums are not exact deployment cost estimates.",
            "A valid retrieval-quality report requires reviewed relevance labels.",
        ],
    }

    REPORT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# Cost-Efficient RAG — Benchmark and Cost Report",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## 1. System configuration",
        "",
        "| Component | Configuration |",
        "|---|---|",
        "| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |",
        "| Vector dimensions | 384 |",
        "| Vector database | LanceDB |",
        "| LLM runtime | Ollama |",
        "| Language model | llama3.2:3b |",
        "",
        "## 2. Retrieval benchmark",
        "",
    ]

    if retrieval is None:
        lines.append("Retrieval benchmark file is missing.")
    else:
        lines.append("Values extracted from the saved retrieval benchmark:")
        lines.append("")
        metric_keys = {
            "mean_latency_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "mean_latency",
            "p50_latency",
            "p95_latency",
            "queries",
            "questions",
            "total_queries",
            "mean_results",
            "indexed_chunks",
        }
        found = find_values(retrieval, metric_keys)

        if found:
            lines.extend([
                "| Metric | Value |",
                "|---|---:|",
            ])
            for name, value in found:
                lines.append(f"| `{name}` | {fmt(value)} |")
        else:
            lines.append(
                "No recognized summary fields found. See the raw JSON "
                "artifact for the benchmark schema."
            )

    lines.extend([
        "",
        "## 3. Answer generation benchmark",
        "",
    ])

    if answers is None:
        lines.append("Answer-quality summary file is missing.")
    else:
        lines.append(
            "The existing run collected answer-generation samples. "
            "Successful execution is not a correctness score."
        )
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")

        keys = {
            "questions_total",
            "successful_answers",
            "failed_answers",
            "mean_latency_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "total_prompt_tokens",
            "total_completion_tokens",
        }

        found = find_values(answers, keys)
        for name, value in found:
            lines.append(f"| `{name}` | {fmt(value)} |")

    lines.extend([
        "",
        "## 4. Storage estimates",
        "",
        "These are analytical estimates, not measured database footprints.",
        "",
    ])

    if storage and isinstance(storage.get("results"), list):
        lines.extend([
            "| Vectors | Raw vectors (GB) | Estimated total (GB) |",
            "|---:|---:|---:|",
        ])

        for row in storage["results"]:
            lines.append(
                f"| {row['vectors']:,} "
                f"| {row['raw_vector_storage_GB']:.3f} "
                f"| {row['estimated_total_storage_GB']:.3f} |"
            )
    else:
        lines.append("Storage estimate file missing or has an unexpected schema.")

    lines.extend([
        "",
        "## 5. Managed vector database comparison",
        "",
        "| Provider / plan | Published minimum or pricing model |",
        "|---|---|",
        "| Pinecone Builder | $20/month flat plan price |",
        "| Pinecone Standard | $50/month minimum usage |",
        "| Pinecone Enterprise | $500/month minimum usage |",
        "| Qdrant Cloud Free | 0.5 vCPU, 1 GB RAM, 4 GB disk |",
        "| Qdrant Cloud Standard | Usage-based compute, memory, and storage |",
        "",
        "These are plan-level reference points, not directly comparable "
        "quotes for 100K, 1M, or 10M vectors. Workload, index configuration, "
        "query rate, replicas, region, backups, and SLA requirements affect cost.",
        "",
        "## 6. Limitations and interpretation",
        "",
        "- The actual local corpus contains only 18 chunks.",
        "- Large-scale storage figures are estimates based on stated assumptions.",
        "- Do not present estimated large-scale latency as measured performance.",
        "- Do not treat successful LLM calls as evidence of groundedness.",
        "- Final Recall@K, MRR, and nDCG require reviewed relevance judgments.",
        "- Local self-hosting avoids a managed database subscription but still "
        "incurs hardware, electricity, maintenance, backup, and operational costs.",
        "",
        "## 7. Official pricing references",
        "",
        "- Pinecone: https://www.pinecone.io/pricing/",
        "- Qdrant Cloud: https://qdrant.tech/pricing/",
        "- Qdrant billing: "
        "https://qdrant.tech/documentation/cloud-pricing-payments/",
        "",
        "Verify current prices before submission; published plans can change.",
        "",
    ])

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    cost_rows = [
        {
            "provider": "Local LanceDB",
            "plan": "Self-hosted",
            "published_monthly_minimum_usd": "",
            "cost_model": "Local infrastructure and operating costs",
            "exact_cost_for_vector_count": "Not calculated",
        },
        {
            "provider": "Pinecone",
            "plan": "Builder",
            "published_monthly_minimum_usd": 20,
            "cost_model": "Plan price; usage limits apply",
            "exact_cost_for_vector_count": "Not calculated",
        },
        {
            "provider": "Pinecone",
            "plan": "Standard",
            "published_monthly_minimum_usd": 50,
            "cost_model": "Usage-based; $50 monthly minimum",
            "exact_cost_for_vector_count": "Not calculated",
        },
        {
            "provider": "Pinecone",
            "plan": "Enterprise",
            "published_monthly_minimum_usd": 500,
            "cost_model": "Usage-based; $500 monthly minimum",
            "exact_cost_for_vector_count": "Not calculated",
        },
        {
            "provider": "Qdrant Cloud",
            "plan": "Free",
            "published_monthly_minimum_usd": 0,
            "cost_model": "0.5 vCPU, 1 GB RAM, 4 GB disk",
            "exact_cost_for_vector_count": "Capacity must be validated",
        },
        {
            "provider": "Qdrant Cloud",
            "plan": "Standard",
            "published_monthly_minimum_usd": "",
            "cost_model": "Resource-based billing",
            "exact_cost_for_vector_count": "Use official calculator",
        },
    ]

    with COST_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cost_rows[0].keys())
        writer.writeheader()
        writer.writerows(cost_rows)

    print("Created:")
    print(f"  {REPORT_MD}")
    print(f"  {REPORT_JSON}")
    print(f"  {COST_CSV}")
    print("\nReview the Markdown report before submitting it.")


if __name__ == "__main__":
    main()
