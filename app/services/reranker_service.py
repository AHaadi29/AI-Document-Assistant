from sentence_transformers import CrossEncoder

_cross_encoder = None
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_cross_encoder():
    """Loads the cross-encoder reranking model once and reuses it."""
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def rerank(question: str, candidates: list):
    """Re-scores candidate chunks by looking at the question and each
    chunk together directly, rather than relying on precomputed embedding
    similarity. Returns (chunk, score) pairs sorted best-first.
    """
    if not candidates:
        return []

    model = get_cross_encoder()
    pairs = [[question, c.page_content] for c in candidates]
    scores = model.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored
