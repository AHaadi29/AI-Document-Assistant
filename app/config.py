from pathlib import Path

from dotenv import load_dotenv
import os

from fastapi.templating import Jinja2Templates

load_dotenv()

APP_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"
UPLOAD_DIR = APP_DIR.parent / "uploaded_pdfs"
CHROMA_DIR = APP_DIR.parent / "chroma_db"

UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = "openai/gpt-oss-120b"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
