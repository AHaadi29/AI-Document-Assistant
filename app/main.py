from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.pages import router as pages_router
from app.api import upload
from app.api import ask
from app.api import documents
from app.config import STATIC_DIR

app = FastAPI(
    title="AI Document Assistant",
    description="Chat with any Document using RAG",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(pages_router)
app.include_router(upload.router)
app.include_router(ask.router)
app.include_router(documents.router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catches any error we did not explicitly handle, so the user always
    gets a clean JSON response instead of a raw server crash."""
    return JSONResponse(
        status_code=500,
        content={"detail": "Something went wrong on our end. Please try again."},
    )
