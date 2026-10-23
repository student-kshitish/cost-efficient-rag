import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evaluation" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Current project configuration
DIMENSIONS = 384
BYTES_PER_COMPONENT = 4  # float32
VECTOR_COUNTS = [100_000, 1_000_000, 10_000_000]

# Explicit planning assumptions, not measured values.
AVG_TEXT_BYTES = 3000
AVG_METADATA_BYTES = 500

# Extra capacity for indexing, database structures, and overhead.
# This is a planning multiplier, not a measured LanceDB overhead.
INDEX_OVERHEAD_FACTOR = 1.5

BYTES_PER_GB = 1_000_000_000
BYTES_PER_GIB = 1024**3


def gb(value):
    return value / BYTES_PER_GB


def gib(value):
    return value / BYTES_PER_GIB


def main():
    rows = []

    print("COST-EFFICIENT RAG: SCALABILITY ESTIMATES")
    print("=" * 76)
    print(f"Embedding dimensions: {DIMENSIONS}")
    print(f"Vector type: float32")
    print(f"Average text assumption: {AVG_TEXT_BYTES:,} bytes/chunk")
    print(f"Metadata assumption: {AVG_METADATA_BYTES:,} bytes/chunk")
    print(f"Index overhead multiplier: {INDEX_OVERHEAD_FACTOR}")
    print("\nNOTE: These are estimates, not benchmark measurements.\n")

    for count in VECTOR_COUNTS:
        raw_vectors = count * DIMENSIONS * BYTES_PER_COMPONENT
        text_storage = count * AVG_TEXT_BYTES
        metadata_storage = count * AVG_METADATA_BYTES

        base_storage = raw_vectors + text_storage + metadata_storage
        estimated_total = base_storage * INDEX_OVERHEAD_FACTOR

        row = {
            "vectors": count,
            "dimensions": DIMENSIONS,
            "vector_dtype": "float32",
            "raw_vector_storage_GB": round(gb(raw_vectors), 3),
            "raw_vector_storage_GiB": round(gib(raw_vectors), 3),
            "assumed_text_storage_GB": round(gb(text_storage), 3),
            "assumed_metadata_storage_GB": round(gb(metadata_storage), 3),
            "estimated_total_storage_GB": round(gb(estimated_total), 3),
            "estimated_total_storage_GiB": round(gib(estimated_total), 3),
            "status": "analytical_estimate_not_measured",
        }
        rows.append(row)

        print(f"Vector count: {count:,}")
        print(f"  Raw vectors:       {row['raw_vector_storage_GB']:.3f} GB")
        print(f"  Assumed text:      {row['assumed_text_storage_GB']:.3f} GB")
        print(f"  Assumed metadata:  {row['assumed_metadata_storage_GB']:.3f} GB")
        print(f"  Estimated total:   {row['estimated_total_storage_GB']:.3f} GB")
        print()

    json_path = RESULTS / "scalability_estimates.json"
    csv_path = RESULTS / "scalability_estimates.csv"

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "analytical_storage_estimate",
        "assumptions": {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "dimensions": DIMENSIONS,
            "vector_dtype": "float32",
            "average_text_bytes_per_chunk": AVG_TEXT_BYTES,
            "average_metadata_bytes_per_chunk": AVG_METADATA_BYTES,
            "index_overhead_multiplier": INDEX_OVERHEAD_FACTOR,
            "warning": (
                "Estimates only. Actual storage depends on chunk length, "
                "metadata, LanceDB/Arrow layout, index configuration, "
                "compression, deletion history, and filesystem overhead."
            ),
        },
        "results": rows,
    }

    json_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("Files created:")
    print(f"  {json_path}")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
