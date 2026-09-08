from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.upload_services import save_document
from app.services.processing_service import process_document
from app.services.pdf_processor import SUPPORTED_EXTENSIONS

router = APIRouter()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    extension = Path(file.filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types: {allowed}",
        )

    try:
        saved_path = save_document(file)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Could not save the uploaded file. Please try again.",
        )

    try:
        chunk_count = process_document(saved_path)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Could not process this file. It may be corrupted, empty, or in an unsupported format.",
        )

    return {
        "filename": file.filename,
        "source": saved_path,
        "message": "Document uploaded and processed successfully.",
        "chunks_stored": chunk_count,
    }