from pydantic import BaseModel
from typing import List, Optional

class DocumentUpload(BaseModel):
    filename: str
    content_type: str

class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

class Answer(BaseModel):
    question: str
    answer: str
    sources: List[str]
    confidence: Optional[float] = None

class DocumentInfo(BaseModel):
    filename: str
    chunks: int
    status: str