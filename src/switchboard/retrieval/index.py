"""Help-centre retrieval: BM25 and dense embeddings in a Chroma collection.

Both retrievers share one interface (``search(query, k) -> list[Hit]``), so the
``search_help`` tool, the retrieval eval and the API do not care which is active.

- **BM25** (``rank_bm25``): lexical, no model download. The CI baseline.
- **Dense**: ``all-MiniLM-L6-v2`` embeddings stored in an in-process Chroma
  collection with cosine distance. At 22 articles a numpy matrix product would be
  just as exact; Chroma is used because it is the same code path that would hold a
  real help centre (persistence, metadata filters, approximate search at scale).
  An ``official_only`` metadata filter is implemented but ``search_help`` does not
  use it: community posts stay searchable on purpose, so the poisoned article can
  reach the model and the indirect-injection guardrail is actually tested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from switchboard import config
from switchboard.kb import Article, load_articles

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenise(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class Hit:
    article: Article
    score: float


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 3, official_only: bool = False) -> list[Hit]: ...


class BM25Retriever:
    name = "bm25"

    def __init__(self, articles: tuple[Article, ...] | None = None) -> None:
        from rank_bm25 import BM25Okapi

        self.articles = articles or load_articles()
        self._bm25 = BM25Okapi([_tokenise(a.indexed_text) for a in self.articles])

    def search(self, query: str, k: int = 3, official_only: bool = False) -> list[Hit]:
        scores = self._bm25.get_scores(_tokenise(query))
        order = sorted(range(len(self.articles)), key=lambda i: -scores[i])
        hits = [Hit(self.articles[i], float(scores[i])) for i in order]
        if official_only:
            hits = [h for h in hits if h.article.source == "official"]
        return hits[:k]


class DenseRetriever:
    name = "dense"

    def __init__(self, articles: tuple[Article, ...] | None = None) -> None:
        import chromadb
        from chromadb.config import Settings
        from sentence_transformers import SentenceTransformer

        self.articles = articles or load_articles()
        self._by_id = {a.id: a for a in self.articles}
        self._model = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)
        client = chromadb.EphemeralClient(Settings(anonymized_telemetry=False))
        self._col = client.get_or_create_collection(
            "help_centre", metadata={"hnsw:space": "cosine"}
        )
        vecs = self._model.encode(
            [a.indexed_text for a in self.articles], normalize_embeddings=True
        )
        self._col.upsert(
            ids=[a.id for a in self.articles],
            embeddings=vecs.tolist(),
            documents=[a.indexed_text for a in self.articles],
            metadatas=[{"source": a.source} for a in self.articles],
        )

    def search(self, query: str, k: int = 3, official_only: bool = False) -> list[Hit]:
        q = self._model.encode([query], normalize_embeddings=True)[0]
        res = self._col.query(
            query_embeddings=[q.tolist()],
            n_results=min(k, len(self.articles)),
            where={"source": "official"} if official_only else None,
        )
        ids = res["ids"][0]
        dists = (res.get("distances") or [[0.0] * len(ids)])[0]
        return [Hit(self._by_id[i], 1.0 - float(d)) for i, d in zip(ids, dists, strict=True)]


def build_retriever(name: str | None = None) -> Retriever:
    name = (name or config.RETRIEVER).lower()
    if name == "dense":
        return DenseRetriever()
    if name == "bm25":
        return BM25Retriever()
    raise ValueError(f"Unknown retriever {name!r} (expected 'bm25' or 'dense').")
