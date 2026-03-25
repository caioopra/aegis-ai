"""BM25 sparse vectorizer for hybrid search in Qdrant."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

# Portuguese stop words (common words that add noise to BM25)
STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "ao",
        "aos",
        "as",
        "com",
        "como",
        "da",
        "das",
        "de",
        "do",
        "dos",
        "e",
        "em",
        "é",
        "era",
        "esse",
        "essa",
        "este",
        "esta",
        "eu",
        "foi",
        "já",
        "lhe",
        "mais",
        "mas",
        "me",
        "meu",
        "na",
        "nas",
        "no",
        "nos",
        "não",
        "nem",
        "nós",
        "o",
        "os",
        "ou",
        "para",
        "pela",
        "pelas",
        "pelo",
        "pelos",
        "por",
        "que",
        "qual",
        "quando",
        "se",
        "sem",
        "ser",
        "seu",
        "sua",
        "são",
        "também",
        "te",
        "tem",
        "um",
        "uma",
        "uns",
        "você",
    }
)

# Default vocabulary size for hashing trick
VOCAB_SIZE = 30_000


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25: lowercase, split on non-word chars, remove stop words."""
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    return [t for t in tokens if len(t) > 1 and t not in STOP_WORDS]


def _term_hash(term: str, vocab_size: int = VOCAB_SIZE) -> int:
    """Deterministic hash of a term to a sparse vector index."""
    h = hashlib.md5(term.encode("utf-8")).hexdigest()
    return int(h, 16) % vocab_size


class BM25Vectorizer:
    """Compute BM25 sparse vectors for documents and queries.

    Uses the hashing trick to map terms to fixed-size sparse vector indices.
    Fitted statistics (IDF) can be saved/loaded for use at query time.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        vocab_size: int = VOCAB_SIZE,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.vocab_size = vocab_size
        self.doc_count: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_freq: dict[int, int] = {}  # term_hash -> number of docs containing term

    def fit(self, texts: list[str]) -> BM25Vectorizer:
        """Compute IDF statistics from a corpus of texts."""
        self.doc_count = len(texts)
        total_len = 0
        self.doc_freq = {}

        for text in texts:
            tokens = tokenize(text)
            total_len += len(tokens)
            seen: set[int] = set()
            for token in tokens:
                h = _term_hash(token, self.vocab_size)
                if h not in seen:
                    self.doc_freq[h] = self.doc_freq.get(h, 0) + 1
                    seen.add(h)

        self.avg_doc_len = total_len / max(self.doc_count, 1)
        return self

    def encode_document(self, text: str) -> tuple[list[int], list[float]]:
        """Compute a BM25 sparse vector for a document.

        Returns ``(indices, values)`` suitable for Qdrant's ``SparseVector``.
        """
        tokens = tokenize(text)
        tf = Counter(_term_hash(t, self.vocab_size) for t in tokens)
        doc_len = len(tokens)

        indices: list[int] = []
        values: list[float] = []

        for term_hash in sorted(tf):
            freq = tf[term_hash]
            df = self.doc_freq.get(term_hash, 0)
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)
            tf_norm = (freq * (self.k1 + 1)) / (
                freq + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_len, 1))
            )
            score = idf * tf_norm
            if score > 0:
                indices.append(term_hash)
                values.append(round(score, 6))

        return indices, values

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        """Compute a BM25 sparse vector for a query.

        Uses binary term frequency (each term counted once) weighted by IDF.
        """
        tokens = tokenize(text)
        seen: dict[int, bool] = {}
        for t in tokens:
            h = _term_hash(t, self.vocab_size)
            seen[h] = True

        indices: list[int] = []
        values: list[float] = []

        for term_hash in sorted(seen):
            df = self.doc_freq.get(term_hash, 0)
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)
            if idf > 0:
                indices.append(term_hash)
                values.append(round(idf, 6))

        return indices, values

    def save(self, path: Path) -> None:
        """Save fitted statistics to a JSON file."""
        data: dict[str, Any] = {
            "k1": self.k1,
            "b": self.b,
            "vocab_size": self.vocab_size,
            "doc_count": self.doc_count,
            "avg_doc_len": self.avg_doc_len,
            # JSON keys must be strings
            "doc_freq": {str(k): v for k, v in self.doc_freq.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BM25Vectorizer:
        """Load a previously fitted vectorizer from a JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        v = cls(k1=data["k1"], b=data["b"], vocab_size=data["vocab_size"])
        v.doc_count = data["doc_count"]
        v.avg_doc_len = data["avg_doc_len"]
        v.doc_freq = {int(k): v_ for k, v_ in data["doc_freq"].items()}
        return v
