import time

import httpx

from app.retrieval.search import RetrievalService


class AnswerService:
    """Generate answers grounded in retrieved document chunks."""

    def __init__(
        self,
        retriever: RetrievalService,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        self.retriever = retriever
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        document_type: str | None = None,
    ) -> dict:
        """Retrieve evidence and generate an answer."""

        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        started = time.perf_counter()

        # Step 1: Retrieve relevant document chunks.
        results = self.retriever.retrieve(
            question=question,
            top_k=top_k,
            document_type=document_type,
        )

        # Abstain when retrieval returns no evidence.
        if not results:
            return {
                "question": question,
                "answer": (
                    "I cannot answer this question because "
                    "no relevant evidence was retrieved."
                ),
                "citations": [],
                "retrieved_chunks": 0,
                "latency_ms": round(
                    (time.perf_counter() - started) * 1000, 2
                ),
                "model": self.model,
                "usage": {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                },
            }

        # Step 2: Prepare evidence with citation identifiers.
        evidence_blocks = []
        citations = []

        for index, result in enumerate(results, start=1):
            citation_id = f"S{index}"

            evidence_blocks.append(
                f"[{citation_id}]\n"
                f"Source: {result['source']}\n"
                f"Chunk index: {result['chunk_index']}\n"
                f"Text:\n{result['text']}"
            )

            citations.append({
                "id": citation_id,
                "source": result["source"],
                "chunk_index": result["chunk_index"],
                "distance": result["distance"],
            })

        context = "\n\n---\n\n".join(evidence_blocks)

        # Step 3: Instruct the model to use only retrieved evidence.
        system_prompt = """
You are a retrieval-augmented question-answering assistant.

Rules:
1. Answer using only the supplied evidence.
2. Do not invent facts or fill gaps using outside knowledge.
3. If the evidence is insufficient, say so explicitly.
4. Cite factual claims using identifiers such as [S1] or [S2].
5. Use only citation identifiers supplied in the evidence.
6. Distinguish direct evidence from conclusions.
7. Be concise and technically accurate.
"""

        user_prompt = f"""
Question:
{question}

Retrieved evidence:
{context}

Answer the question using the evidence above.
Include citations for supported claims.
If the evidence is insufficient, abstain rather than guess.
"""

        # Step 4: Generate an answer with the local Ollama model.
        response = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                },
            },
            timeout=self.timeout,
        )

        response.raise_for_status()
        payload = response.json()

        answer_text = payload["message"]["content"].strip()

        return {
            "question": question,
            "answer": answer_text,
            "citations": citations,
            "retrieved_chunks": len(results),
            "latency_ms": round(
                (time.perf_counter() - started) * 1000, 2
            ),
            "model": self.model,
            "usage": {
                "prompt_tokens": payload.get("prompt_eval_count"),
                "completion_tokens": payload.get("eval_count"),
            },
        }
