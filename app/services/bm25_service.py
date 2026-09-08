import re

from rank_bm25 import BM25Okapi

from app.services.vector_store import get_vectorstore, SimpleDoc


def _tokenize(text: str):
    """Splits text into lowercase word tokens for BM25 matching."""
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_search(question: str, document: str = None, k: int = 10):
    """Keyword-based search over a document's chunks using BM25.

    Unlike semantic search, this rewards exact and rare word matches
    rather than overall meaning -- useful for specific terms, codes,
    and names that embeddings sometimes underweight.
    """
    vectorstore = get_vectorstore()
    data = vectorstore.get(where={"source": document}) if document else vectorstore.get()

    docs = data.get("documents") or []
    metas = data.get("metadatas") or []

    if not docs:
        return []

    tokenized_corpus = [_tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized_corpus)

    scores = bm25.get_scores(_tokenize(question))
    ranked = sorted(zip(docs, metas, scores), key=lambda item: item[2], reverse=True)

    return [SimpleDoc(d, m or {}) for d, m, score in ranked[:k] if score > 0]
