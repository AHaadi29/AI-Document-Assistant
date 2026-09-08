import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from app.config import CHROMA_DIR, EMBEDDING_MODEL_NAME

_embeddings = None
COLLECTION_NAME = "pdf_documents"


class SimpleDoc:
    """A minimal stand-in for a LangChain Document, so code that expects
    .page_content and .metadata works the same regardless of where the
    chunk came from."""

    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


def get_embeddings():
    """Loads the AI embedding model once and reuses it."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings


def get_vectorstore():
    """Returns a Chroma vector store connected to our persistent database."""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def store_chunks(chunks, source: str):
    """Stores a document's chunks, replacing any older chunks from the same file."""
    vectorstore = get_vectorstore()

    try:
        vectorstore.delete(where={"source": source})
    except Exception:
        pass

    vectorstore.add_documents(chunks)
    return len(chunks)


def list_documents():
    """Returns the distinct documents currently stored in the vector database."""
    vectorstore = get_vectorstore()
    data = vectorstore.get()

    sources = set()
    for meta in (data.get("metadatas") or []):
        if meta and meta.get("source"):
            sources.add(meta["source"])

    documents = [
        {"source": source, "filename": os.path.basename(source)}
        for source in sorted(sources)
    ]
    return documents


def get_first_chunk(source: str):
    """Returns the opening chunk (lowest page number) of a specific document."""
    vectorstore = get_vectorstore()
    data = vectorstore.get(where={"source": source})

    docs = data.get("documents") or []
    metas = data.get("metadatas") or []

    if not docs:
        return None

    pairs = list(zip(docs, metas))
    pairs.sort(key=lambda pair: (pair[1] or {}).get("page", 0))

    content, meta = pairs[0]
    return SimpleDoc(content, meta or {})


def get_page_chunks(source: str, requested_page: int):
    """Returns chunks matching a human-requested page number.

    Documents often have TWO different page numbering systems: the raw
    physical page position in the file, and the printed page number
    shown in the document's own footer/header (which is usually offset
    due to cover pages, tables of contents, etc). When a person asks
    "what is on page 99", they mean the printed number they can see on
    the page -- so we match against "page_label" (the printed number)
    first, and only fall back to the raw physical position if no
    page_label match is found.
    """
    vectorstore = get_vectorstore()
    data = vectorstore.get(where={"source": source})

    docs = data.get("documents") or []
    metas = data.get("metadatas") or []

    label_matches = [
        (doc, meta) for doc, meta in zip(docs, metas)
        if meta and str(meta.get("page_label", "")).strip() == str(requested_page)
    ]

    if label_matches:
        combined_text = "\n\n".join(doc for doc, meta in label_matches)
        return SimpleDoc(combined_text, label_matches[0][1])

    position_matches = [
        (doc, meta) for doc, meta in zip(docs, metas)
        if meta and meta.get("page") == requested_page - 1
    ]

    if not position_matches:
        return None

    combined_text = "\n\n".join(doc for doc, meta in position_matches)
    return SimpleDoc(combined_text, position_matches[0][1])
