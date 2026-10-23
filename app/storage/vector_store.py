from pathlib import Path

import lancedb
import pyarrow as pa


class VectorStore:
    """Local vector storage using LanceDB."""

    def __init__(
        self,
        db_path: str = "./data/lancedb",
        dimension: int = 384,
        table_name: str = "chunks",
    ):
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")

        self.dimension = dimension
        self.db_path = str(Path(db_path))

        Path(self.db_path).mkdir(parents=True, exist_ok=True)

        self.db = lancedb.connect(self.db_path)
        self.table_name = table_name

        existing_tables = self.db.list_tables().tables

        if table_name in existing_tables:
            self.table = self.db.open_table(table_name)
        else:
            schema = pa.schema([
                pa.field("chunk_id", pa.string()),
                pa.field("document_id", pa.string()),
                pa.field("source", pa.string()),
                pa.field("document_type", pa.string()),
                pa.field("content_hash", pa.string()),
                pa.field("chunk_index", pa.int64()),
                pa.field("text", pa.string()),
                pa.field(
                    "vector",
                    pa.list_(pa.float32(), dimension),
                ),
            ])

            self.table = self.db.create_table(
                table_name,
                schema=schema,
            )

    def add_chunks(self, chunks: list[dict]) -> int:
        """Insert chunks with vectors and metadata."""
        if not chunks:
            return 0

        required_fields = {
            "chunk_id",
            "document_id",
            "source",
            "document_type",
            "content_hash",
            "chunk_index",
            "text",
            "vector",
        }

        rows = []

        for chunk in chunks:
            missing = required_fields - chunk.keys()

            if missing:
                raise ValueError(
                    f"Missing chunk fields: {sorted(missing)}"
                )

            vector = [float(value) for value in chunk["vector"]]

            if len(vector) != self.dimension:
                raise ValueError(
                    f"Expected vector dimension {self.dimension}, "
                    f"received {len(vector)}"
                )

            if not chunk["text"].strip():
                raise ValueError("Chunk text cannot be empty")

            rows.append({
                "chunk_id": str(chunk["chunk_id"]),
                "document_id": str(chunk["document_id"]),
                "source": str(chunk["source"]),
                "document_type": str(chunk["document_type"]),
                "content_hash": str(chunk["content_hash"]),
                "chunk_index": int(chunk["chunk_index"]),
                "text": str(chunk["text"]),
                "vector": vector,
            })

        self.table.add(rows)

        return len(rows)

    def search(
        self,
        query_vector,
        top_k: int = 5,
        document_type: str | None = None,
    ) -> list[dict]:
        """Search for similar chunks with optional file-type filtering."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        vector = [float(value) for value in query_vector]

        if len(vector) != self.dimension:
            raise ValueError(
                f"Expected query dimension {self.dimension}, "
                f"received {len(vector)}"
            )

        query = self.table.search(vector).metric("cosine")

        if document_type is not None:
            safe_type = document_type.replace("'", "''")
            query = query.where(
                f"document_type = '{safe_type}'",
                prefilter=True,
            )

        return query.limit(top_k).to_list()

    def delete_document(self, document_id: str) -> None:
        """Delete all chunks belonging to one document."""
        safe_id = document_id.replace("'", "''")
        self.table.delete(f"document_id = '{safe_id}'")

    def get_document_hash(self, document_id: str) -> str | None:
        """Return a stored content hash, or None if the document is absent."""
        safe_id = document_id.replace("'", "''")

        rows = (
            self.table.search()
            .where(f"document_id = '{safe_id}'")
            .select(["content_hash"])
            .limit(1)
            .to_list()
        )

        if not rows:
            return None

        return rows[0]["content_hash"]

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self.table.count_rows()
