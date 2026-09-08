from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def get_loader_for_file(file_path: str):
    """Picks the right document loader based on the file's extension.
    Each file type needs its own reader since documents (PDF, Word, and
    plain text) are all structured completely differently under the hood.
    """
    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        return PyPDFLoader(file_path)
    elif extension == ".docx":
        return Docx2txtLoader(file_path)
    elif extension == ".txt":
        return TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {extension}")


def load_and_split_document(file_path: str):
    """Reads a document (PDF, DOCX, or TXT) and breaks it into small
    overlapping text chunks."""
    loader = get_loader_for_file(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(pages)
    return chunks