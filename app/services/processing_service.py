from app.services.pdf_processor import load_and_split_document
from app.services.vector_store import store_chunks


def process_document(file_path: str) -> int:
    """Runs the full pipeline: extract, chunk, and store a document's content."""
    chunks = load_and_split_document(file_path)
    chunk_count = store_chunks(chunks, source=file_path)
    return chunk_count
