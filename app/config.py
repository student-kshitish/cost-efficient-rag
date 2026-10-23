import os
from pathlib import Path

from dotenv import load_dotenv

# Find the project root, regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables from the project's .env file.
load_dotenv(PROJECT_ROOT / ".env")


class Settings:
    """Application configuration loaded from environment variables."""

    def __init__(self):
        self.app_name = os.getenv("APP_NAME", "Cost-Efficient RAG")
        self.app_env = os.getenv("APP_ENV", "development")
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

        self.vector_db_path = os.getenv(
            "VECTOR_DB_PATH",
            str(PROJECT_ROOT / "data" / "lancedb"),
        )
        self.document_dir = os.getenv(
            "DOCUMENT_DIR",
            str(PROJECT_ROOT / "data" / "documents"),
        )

        self.embedding_model = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )

        self.chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))
        self.top_k = int(os.getenv("TOP_K", "5"))

        self.llm_provider = os.getenv("LLM_PROVIDER", "ollama")
        self.llm_model = os.getenv("LLM_MODEL", "llama3.2:3b")
        self.llm_base_url = os.getenv(
            "LLM_BASE_URL",
            "http://localhost:11434",
        ).rstrip("/")

        self.relevance_threshold = float(
            os.getenv("RELEVANCE_THRESHOLD", "0.0")
        )


settings = Settings()
