from typing import Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.qa_service import answer_question, stream_answer

router = APIRouter()


class HistoryTurn(BaseModel):
    question: str
    answer: str


class QuestionRequest(BaseModel):
    question: str
    history: list[HistoryTurn] = []
    document: Optional[str] = None


@router.post("/ask")
async def ask_question(request: QuestionRequest):
    history = [turn.model_dump() for turn in request.history]
    result = answer_question(request.question, history=history, document=request.document)
    return result


@router.post("/ask/stream")
async def ask_question_stream(request: QuestionRequest):
    history = [turn.model_dump() for turn in request.history]
    generator = stream_answer(request.question, history=history, document=request.document)
    return StreamingResponse(generator, media_type="text/plain")
