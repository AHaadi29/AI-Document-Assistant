import json
import re

from groq import Groq

from app.config import GROQ_API_KEY, GROQ_MODEL_NAME
from app.services.vector_store import get_vectorstore, get_first_chunk, get_page_chunks
from app.services.bm25_service import bm25_search
from app.services.reranker_service import rerank
from app.services.web_search_service import search_web

_groq_client = None

DOC_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about a specific "
    "document. Use ONLY the context provided with each question to answer. "
    "If the answer is not in the context, respond with exactly: "
    "\"I don't know based on the document.\" and nothing else."
)

PAGE_SYSTEM_PROMPT = (
    "You are a helpful assistant describing the content of a specific page "
    "from a document. Summarize or answer using ONLY the page content "
    "given below. If the page appears to contain mostly a diagram, image, "
    "or table with little extractable text, say so plainly."
)

WEB_SYSTEM_PROMPT = (
    "You are a helpful assistant. The user's document could not answer "
    "this question, so you have been given live web search results "
    "instead. Answer clearly and concisely using these results."
)

SOURCE_DELIMITER = "\n@@SOURCES@@\n"
DONT_KNOW_PHRASE = "i don't know based on the document"
PAGE_REFERENCE_PATTERN = re.compile(r"\bpage\b[^\d]{0,20}(\d+)", re.IGNORECASE)


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def _is_dont_know(answer: str) -> bool:
    return DONT_KNOW_PHRASE in (answer or "").strip().lower()


def hybrid_retrieve(question: str, document: str, k: int = 10):
    """Combines semantic search and BM25 keyword search, then reranks the
    merged pool with a cross-encoder for a more precise final ordering."""
    vectorstore = get_vectorstore()
    filter_dict = {"source": document} if document else None

    try:
        if filter_dict:
            semantic_results = vectorstore.similarity_search(question, k=k, filter=filter_dict)
        else:
            semantic_results = vectorstore.similarity_search(question, k=k)
    except Exception:
        semantic_results = []

    keyword_results = bm25_search(question, document=document, k=k)

    seen = set()
    candidates = []
    for doc in semantic_results + keyword_results:
        key = (doc.metadata.get("page"), doc.page_content[:80])
        if key not in seen:
            seen.add(key)
            candidates.append(doc)

    if not candidates:
        return []

    return rerank(question, candidates)


def get_document_chunks(question: str, document: str):
    """Best available chunks: top reranked matches plus the document's
    opening chunk as a fallback for broad, whole-document questions."""
    reranked = hybrid_retrieve(question, document)
    top_chunks = [doc for doc, score in reranked[:3]]

    intro_chunk = get_first_chunk(document)
    if intro_chunk is not None:
        already_included = any(
            c.metadata.get("page") == intro_chunk.metadata.get("page") for c in top_chunks
        )
        if not already_included:
            top_chunks.append(intro_chunk)

    return top_chunks[:4]


def rewrite_query(question: str, history: list) -> str:
    """Turns a vague question into a clear, standalone query using recent
    conversation context. Used only when falling back to web search."""
    history_text = "\n".join(
        f"Q: {t.get('question','')}\nA: {t.get('answer','')}" for t in (history or [])[-2:]
    )
    prompt = (
        f"Conversation so far:\n{history_text}\n\n"
        "Rewrite the question below into a clear, specific, standalone "
        "search query. Return ONLY the rewritten query, nothing else.\n\n"
        f"Question: {question}"
    )
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else question
    except Exception:
        return question


def verify_citations(answer: str, sources: list) -> list:
    """Keeps only sources that genuinely support a claim in the answer."""
    if not sources:
        return sources

    numbered = "\n\n".join(f"[{i}] {s['snippet']}" for i, s in enumerate(sources))
    prompt = (
        "Below is an AI-generated answer, followed by numbered source "
        "excerpts. Return ONLY a JSON array of the numbers whose excerpt "
        "genuinely supports a claim made in the answer. If none do, "
        "return [].\n\n"
        f"Answer:\n{answer}\n\nSource excerpts:\n{numbered}"
    )
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = response.choices[0].message.content.strip()
        match = re.search(r"\[[\d,\s]*\]", text)
        if match:
            indices = json.loads(match.group(0))
            verified = [s for i, s in enumerate(sources) if i in indices]
            return verified if verified else sources
    except Exception:
        pass

    return sources


def build_doc_messages(question, context, history, system_prompt=DOC_SYSTEM_PROMPT):
    messages = [{"role": "system", "content": system_prompt}]
    for turn in (history or [])[-3:]:
        messages.append({"role": "user", "content": turn.get("question", "")})
        messages.append({"role": "assistant", "content": turn.get("answer", "")})
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
    return messages


def build_web_messages(question, web_context, history):
    messages = [{"role": "system", "content": WEB_SYSTEM_PROMPT}]
    for turn in (history or [])[-3:]:
        messages.append({"role": "user", "content": turn.get("question", "")})
        messages.append({"role": "assistant", "content": turn.get("answer", "")})
    messages.append({"role": "user", "content": f"Web search results:\n{web_context}\n\nQuestion: {question}"})
    return messages


GENERATION_ERROR_MESSAGE = "I ran into a problem generating a response. Please try again in a moment."
NO_ANSWER_MESSAGE = "I couldn't find anything relevant, in the document or on the web, for that question."
PAGE_NOT_FOUND_MESSAGE = "I couldn't find that page number in this document."


def _try_document(question, history, document):
    """Attempts a document-grounded answer. If the question directly
    references a page number, that exact page is fetched by its printed
    page number and answered directly -- this NEVER falls back to web
    search, since a live web search can never know what is on a specific
    page of the user's own uploaded document."""
    if not document:
        return None

    page_match = PAGE_REFERENCE_PATTERN.search(question)

    if page_match:
        human_page = int(page_match.group(1))
        page_chunk = get_page_chunks(document, human_page)

        if page_chunk is None:
            return {"answer": PAGE_NOT_FOUND_MESSAGE, "sources": [], "source_type": "document"}

        context = page_chunk.page_content
        sources = [{
            "page": page_chunk.metadata.get("page_label", human_page),
            "snippet": context[:300],
        }]
        messages = build_doc_messages(question, context, history, system_prompt=PAGE_SYSTEM_PROMPT)

        client = get_groq_client()
        try:
            response = client.chat.completions.create(model=GROQ_MODEL_NAME, messages=messages, temperature=0.2)
            answer = response.choices[0].message.content
        except Exception:
            return {"answer": GENERATION_ERROR_MESSAGE, "sources": [], "source_type": "document"}

        return {"answer": answer, "sources": sources, "source_type": "document"}

    chunks = get_document_chunks(question, document)
    if not chunks:
        return None

    context = "\n\n".join(c.page_content for c in chunks)
    sources = [
        {"page": c.metadata.get("page", 0) + 1, "snippet": c.page_content[:300]}
        for c in chunks
    ]
    messages = build_doc_messages(question, context, history)

    client = get_groq_client()
    try:
        response = client.chat.completions.create(model=GROQ_MODEL_NAME, messages=messages, temperature=0.2)
        answer = response.choices[0].message.content
    except Exception:
        return None

    if _is_dont_know(answer):
        return None

    sources = verify_citations(answer, sources)
    return {"answer": answer, "sources": sources, "source_type": "document"}


def _try_web(question, history):
    query = rewrite_query(question, history) if history else question
    web_results = search_web(query)
    if not web_results:
        return None

    web_context = "\n\n".join(f"{r['title']}\n{r['content']}\nSource: {r['url']}" for r in web_results)
    sources = [{"title": r["title"], "url": r["url"]} for r in web_results]
    messages = build_web_messages(question, web_context, history)
    return messages, sources


def answer_question(question: str, history: list = None, document: str = None) -> dict:
    doc_result = _try_document(question, history, document)
    if doc_result:
        return doc_result

    web_prep = _try_web(question, history)
    if not web_prep:
        return {"answer": NO_ANSWER_MESSAGE, "sources": [], "source_type": "none"}

    messages, sources = web_prep
    client = get_groq_client()
    try:
        response = client.chat.completions.create(model=GROQ_MODEL_NAME, messages=messages, temperature=0.2)
        answer = response.choices[0].message.content
    except Exception:
        return {"answer": GENERATION_ERROR_MESSAGE, "sources": [], "source_type": "none"}

    return {"answer": answer, "sources": sources, "source_type": "web"}


def stream_answer(question: str, history: list = None, document: str = None):
    doc_result = _try_document(question, history, document)

    if doc_result:
        words = doc_result["answer"].split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
        yield SOURCE_DELIMITER + json.dumps({"type": "document", "sources": doc_result["sources"]})
        return

    web_prep = _try_web(question, history)
    if not web_prep:
        yield NO_ANSWER_MESSAGE
        yield SOURCE_DELIMITER + json.dumps({"type": "none", "sources": []})
        return

    messages, sources = web_prep
    client = get_groq_client()
    try:
        stream = client.chat.completions.create(model=GROQ_MODEL_NAME, messages=messages, temperature=0.2, stream=True)
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception:
        yield GENERATION_ERROR_MESSAGE

    yield SOURCE_DELIMITER + json.dumps({"type": "web", "sources": sources})
