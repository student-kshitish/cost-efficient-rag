import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.embeddings.embedder import Embedder
from app.ingestion.pipeline import ingest_file
from app.storage.vector_store import VectorStore

document_root = Path(settings.document_dir).resolve()
document_path = document_root / "Criticality_as_a_Resource.pdf"

if not document_path.is_file():
    raise FileNotFoundError(f"PDF not found: {document_path}")

embedder = Embedder(model_name=settings.embedding_model)
store = VectorStore(
    db_path=settings.vector_db_path,
    dimension=embedder.dimension,
)

before = store.count()
print(f"Chunks before: {before}")

first = ingest_file(
    path=document_path,
    document_root=document_root,
    embedder=embedder,
    store=store,
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
)
after_first = store.count()

second = ingest_file(
    path=document_path,
    document_root=document_root,
    embedder=embedder,
    store=store,
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
)
after_second = store.count()

print(f"\nFirst ingestion: {first}")
print(f"Chunks after first ingestion: {after_first}")
print(f"\nSecond ingestion: {second}")
print(f"Chunks after second ingestion: {after_second}")

if second["status"] == "skipped" and after_second == after_first:
    print("\nPASS: unchanged document was skipped without adding duplicate chunks.")
else:
    print("\nFAIL: idempotency verification failed.")
    raise SystemExit(1)
