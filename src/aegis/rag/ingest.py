"""Document ingestion pipeline: load → chunk → embed → store in Qdrant."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from aegis.config import settings

# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------


def load_text_file(path: Path) -> str:
    """Read a plain-text file and return its contents."""
    return path.read_text(encoding="utf-8")


def load_pdf_file(path: Path) -> str:
    """Read a PDF file and return all pages concatenated as text."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def load_document(path: Path) -> str:
    """Load a document (txt or pdf) and return its text content."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf_file(path)
    if suffix in {".txt", ".md"}:
        return load_text_file(path)
    raise ValueError(f"Unsupported file type: {suffix} ({path})")


def load_all_documents(directory: Path | None = None) -> list[dict[str, str]]:
    """Load all supported documents from a directory.

    Returns a list of ``{"source": filename, "text": content}`` dicts.
    """
    directory = directory or settings.guidelines_dir
    docs: list[dict[str, str]] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in {".txt", ".md", ".pdf"}:
            text = load_document(path)
            docs.append({"source": path.name, "text": text})
    return docs


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Split text into overlapping chunks with metadata.

    Returns a list of ``{"text": ..., "source": ..., "chunk_index": ...}``.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    splits = splitter.split_text(text)
    return [
        {"text": chunk, "source": source, "chunk_index": i}
        for i, chunk in enumerate(splits)
    ]


def chunk_documents(
    docs: list[dict[str, str]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Chunk a list of loaded documents, preserving source metadata."""
    all_chunks: list[dict[str, Any]] = []
    for doc in docs:
        chunks = chunk_text(
            doc["text"], doc["source"],
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(chunks)
    return all_chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def embed_text(text: str) -> list[float]:
    """Generate an embedding vector for a single text using Ollama."""
    response = ollama.embed(model=settings.ollama_embed_model, input=text)
    return response["embeddings"][0]


def embed_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add an ``"embedding"`` field to each chunk dict."""
    for chunk in chunks:
        chunk["embedding"] = embed_text(chunk["text"])
    return chunks


# ---------------------------------------------------------------------------
# Qdrant storage
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 768  # nomic-embed-text output dimension


def get_qdrant_client() -> QdrantClient:
    """Create a Qdrant client from settings."""
    if settings.qdrant_url == ":memory:":
        return QdrantClient(":memory:")
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection(
    client: QdrantClient,
    collection: str | None = None,
    vector_size: int = EMBEDDING_DIM,
) -> None:
    """Create the Qdrant collection if it doesn't exist."""
    collection = collection or settings.qdrant_collection
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )


def store_chunks(
    client: QdrantClient,
    chunks: list[dict[str, Any]],
    collection: str | None = None,
) -> int:
    """Store embedded chunks in Qdrant. Returns the number of points stored."""
    collection = collection or settings.qdrant_collection
    ensure_collection(client, collection)

    points = [
        PointStruct(
            id=i,
            vector=chunk["embedding"],
            payload={"text": chunk["text"], "source": chunk["source"], "chunk_index": chunk["chunk_index"]},
        )
        for i, chunk in enumerate(chunks)
    ]
    client.upsert(collection_name=collection, points=points)
    return len(points)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def ingest_guidelines(
    directory: Path | None = None,
    client: QdrantClient | None = None,
) -> int:
    """Run the full ingestion pipeline: load → chunk → embed → store.

    Returns the total number of chunks stored.
    """
    docs = load_all_documents(directory)
    chunks = chunk_documents(docs)
    chunks = embed_chunks(chunks)

    client = client or get_qdrant_client()
    return store_chunks(client, chunks)
