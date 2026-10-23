from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup


SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".md", ".markdown"}


def load_document(path: str | Path) -> dict:
    """Extract text from PDF, HTML, and Markdown documents."""
    path = Path(path).resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension}")

    if extension == ".pdf":
        with pymupdf.open(path) as pdf:
            pages = []

            for page_number, page in enumerate(pdf, start=1):
                page_text = page.get_text("text").strip()

                if page_text:
                    pages.append(
                        f"[Page {page_number}]\n{page_text}"
                    )

        text = "\n\n".join(pages)

    elif extension in {".html", ".htm"}:
        html = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")

        for element in soup(["script", "style", "noscript"]):
            element.decompose()

        text = soup.get_text(separator="\n", strip=True)

    else:
        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()

    text = text.replace("\x00", "").strip()

    if not text:
        raise ValueError(
            f"No text extracted from {path.name}. "
            "The PDF may be scanned and require OCR."
        )

    return {
        "source": path.name,
        "document_type": extension.lstrip("."),
        "text": text,
    }
