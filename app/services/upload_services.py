import os
import shutil
from fastapi import UploadFile
from app.config import UPLOAD_DIR


def save_document(file: UploadFile) -> str:
    """Saves an uploaded document to disk and returns its saved path."""
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return file_path
