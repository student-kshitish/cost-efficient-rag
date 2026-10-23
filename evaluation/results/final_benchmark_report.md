# Cost-Efficient RAG — Benchmark and Cost Report

Generated: 2026-10-23T07:21:45.377418+00:00

## 1. System configuration

| Component | Configuration |
|---|---|
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector dimensions | 384 |
| Vector database | LanceDB |
| LLM runtime | Ollama |
| Language model | llama3.2:3b |

## 2. Retrieval benchmark

Values extracted from the saved retrieval benchmark:

| Metric | Value |
|---|---:|
| `questions` | 15.00 |
| `mean_latency_ms` | 23.49 |
| `p50_latency_ms` | 6.78 |
| `p95_latency_ms` | 105.30 |

## 3. Answer generation benchmark

The existing run collected answer-generation samples. Successful execution is not a correctness score.

| Metric | Value |
|---|---:|
| `questions_total` | 15.00 |
| `successful_answers` | 15.00 |
| `failed_answers` | 0.00 |
| `mean_latency_ms` | 4,190.92 |
| `p50_latency_ms` | 3,616.84 |
| `p95_latency_ms` | 8,052.34 |
| `total_prompt_tokens` | 53,136.00 |
| `total_completion_tokens` | 3,293.00 |

## 4. Storage estimates

These are analytical estimates, not measured database footprints.

| Vectors | Raw vectors (GB) | Estimated total (GB) |
|---:|---:|---:|
| 100,000 | 0.154 | 0.755 |
| 1,000,000 | 1.536 | 7.554 |
| 10,000,000 | 15.360 | 75.540 |

## 5. Managed vector database comparison

| Provider / plan | Published minimum or pricing model |
|---|---|
| Pinecone Builder | $20/month flat plan price |
| Pinecone Standard | $50/month minimum usage |
| Pinecone Enterprise | $500/month minimum usage |
| Qdrant Cloud Free | 0.5 vCPU, 1 GB RAM, 4 GB disk |
| Qdrant Cloud Standard | Usage-based compute, memory, and storage |

These are plan-level reference points, not directly comparable quotes for 100K, 1M, or 10M vectors. Workload, index configuration, query rate, replicas, region, backups, and SLA requirements affect cost.

## 6. Limitations and interpretation

- The actual local corpus contains only 18 chunks.
- Large-scale storage figures are estimates based on stated assumptions.
- Do not present estimated large-scale latency as measured performance.
- Do not treat successful LLM calls as evidence of groundedness.
- Final Recall@K, MRR, and nDCG require reviewed relevance judgments.
- Local self-hosting avoids a managed database subscription but still incurs hardware, electricity, maintenance, backup, and operational costs.

## 7. Official pricing references

- Pinecone: https://www.pinecone.io/pricing/
- Qdrant Cloud: https://qdrant.tech/pricing/
- Qdrant billing: https://qdrant.tech/documentation/cloud-pricing-payments/

Verify current prices before submission; published plans can change.
