from pydantic import BaseModel
from typing import Any


class OCRResponse(BaseModel):
    document_type: str
    display_name: str
    confidence: float
    detection_reason: str
    schema: dict[str, Any]
    extracted_fields: dict[str, Any]
    rag_similarity_score: float
    error: str | None = None
