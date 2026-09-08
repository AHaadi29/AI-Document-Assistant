from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from app.config import templates

router = APIRouter(tags=["pages"])


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_name": "AI Document Assistant",
            "tagline": "Chat with any Document using RAG",
        },
    )
