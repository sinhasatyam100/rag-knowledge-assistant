from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime


class QueryRequest(BaseModel):
    """
    What the user sends when asking a question.
    Pydantic validates this automatically — if the data doesn't
    match, it raises a clear error BEFORE your code runs.
    """
    question: str = Field(
        min_length=3,
        max_length=500,
        description="The question to ask the knowledge base"
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Number of chunks to retrieve"
    )
    
    @field_validator("question")
    def question_must_not_be_blank(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be blank or only whitespace")
        return v.strip()


class SourceDocument(BaseModel):
    """A single retrieved chunk with its metadata."""
    content: str
    source: str = "unknown"
    chunk_index: Optional[int] = None


class QueryResponse(BaseModel):
    """What your app sends back."""
    answer: str
    sources: list[SourceDocument]
    question: str
    retrieval_count: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)