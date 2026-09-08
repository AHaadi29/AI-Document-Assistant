from fastapi import APIRouter

from app.services.vector_store import list_documents

router = APIRouter()


@router.get("/documents")
async def get_documents():
    return {"documents": list_documents()}
