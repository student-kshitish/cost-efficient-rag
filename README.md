# Cost-Efficient RAG Application

A Retrieval-Augmented Generation (RAG) application that retrieves relevant document passages and generates grounded answers using a local language model.

## Features

- PDF, HTML, and Markdown ingestion
- Configurable text chunking
- Sentence Transformers embeddings
- LanceDB vector storage
- Semantic retrieval
- Local LLM generation using Ollama
- FastAPI endpoints
- Source citations and abstention support
- Retrieval evaluation and latency benchmarking
- Automated tests
- Storage scalability estimates

## Technology Stack

- Python 3.12
- FastAPI
- Sentence Transformers
- LanceDB
- Ollama
- Llama 3.2 3B
- Pytest

## Installation

Clone the repository:

    git clone git@github.com:student-kshitish/cost-efficient-rag.git
    cd cost-efficient-rag

Create a virtual environment:

    python3.12 -m venv .venv
    source .venv/bin/activate

Install dependencies:

    pip install -r requirements-lock.txt

Download the language model:

    ollama pull llama3.2:3b

Configure your environment using `.env.example` and the options supported by `app/config.py`. Never commit secrets or your `.env` file.

## Run the API

    uvicorn app.api.main:app --host 127.0.0.1 --port 8000

Health check:

    curl http://127.0.0.1:8000/health

Application statistics:

    curl http://127.0.0.1:8000/stats

API documentation:

    http://127.0.0.1:8000/docs

## Run Tests

    PYTHONPATH=. pytest -q

## Evaluation

Run retrieval evaluation:

    PYTHONPATH=. python scripts/evaluate_retrieval.py

Run answer-generation evaluation:

    PYTHONPATH=. python scripts/evaluate_answers.py

Run the retrieval benchmark:

    PYTHONPATH=. python scripts/benchmark_retrieval.py

Run scalability estimates:

    PYTHONPATH=. python scripts/scalability_estimate.py

Results are stored in `evaluation/results/`.

## Initial Retrieval Results

The initial evaluation used 15 questions.

| Metric | Result |
|---|---:|
| Recall@5 | 1.0000 |
| Hit Rate@5 | 1.0000 |
| MRR | 0.9667 |
| nDCG@5 | 0.9004 |
| Precision@5 | 0.5600 |

**Caveat:** Relevance labels were generated using a local LLM and have not all been independently verified. Treat these results as preliminary, not independently validated.

Storage scalability figures are estimates rather than measurements from databases containing millions of vectors.

## Project Structure

- `app/` — API, embeddings, ingestion, retrieval, generation, and storage
- `data/documents/` — sample documents
- `evaluation/` — questions, metrics, and results
- `scripts/` — evaluation and benchmarking tools
- `tests/` — automated tests

## Future Improvements

- Independently validate relevance labels and gold answers
- Evaluate answer faithfulness and relevance
- Benchmark larger real-world datasets
- Compare local and managed vector databases under equivalent conditions

## License

No license has been specified yet.
