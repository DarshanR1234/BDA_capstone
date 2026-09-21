from typing import Any, List, Optional
from pydantic import BaseModel, Field


class VerificationRequest(BaseModel):
    question: Optional[str] = Field(default=None, description="Direct question or viral claim to verify (e.g., 'Did Misbah test positive for COVID?')")
    title: Optional[str] = Field(default="", description="Headline or title of the news article")
    content: Optional[str] = Field(default="", description="Full body text of the article")
    url: Optional[str] = Field(default="", description="Original source URL if available")
    article_id: Optional[int] = Field(default=None, description="Optional article ID for dataset tracking")

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "क्या मिस्बाह-उल-हक वेस्टइंडीज दौरे पर कोरोना संक्रमित हुए थे?",
                "title": "वेस्टइंडीज दौरे पर कोरोना संक्रमित हुए पाकिस्तान के मुख्य कोच मिस्बाह-उल-हक",
                "content": "पाकिस्तान क्रिकेट टीम के मुख्य कोच मिस्बाह-उल-हक वेस्टइंडीज दौरे के समापन पर कोरोना वायरस पॉजिटिव पाए गए हैं।",
                "url": "https://example.com/cricket-news"
            }
        }
    }


class EvidenceItem(BaseModel):
    rank: int = 1
    evidence_id: Any = None
    article_id: Optional[int] = None
    title: str = ""
    content_snippet: Optional[str] = ""
    url: Optional[str] = ""
    semantic_score: Optional[float] = 0.0
    lexical_score: Optional[float] = 0.0
    claim_match: Optional[float] = 0.0
    entity_match: Optional[float] = 0.0
    number_date_match: Optional[float] = 0.0
    evidence_score: float = 0.0
    evidence_type: Optional[str] = "NEWS"
    strong_evidence: Optional[bool] = False


class VerificationResponse(BaseModel):
    status: str = "success"
    verdict: str = Field(..., description="FINAL VERDICT: REAL, FAKE, or UNCERTAIN")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    claim: str = Field(..., description="Main extracted factual claim")
    entities: List[str] = Field(default_factory=list, description="Extracted key named entities")
    numbers_dates: List[str] = Field(default_factory=list, description="Extracted dates, figures, and quantities")
    explanation: str = Field(..., description="Evidence-grounded rationale explaining the verdict")
    summary: str = Field(default="", description="High-level factual conclusion")
    evidence: List[EvidenceItem] = Field(default_factory=list, description="Top evidence articles retrieved from corpus")
    latency_seconds: float = Field(..., description="End-to-end processing time in seconds")


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    uptime_seconds: float = 0.0
    model_loaded: bool = True
    dataset_rows: int = 67587


class SampleArticle(BaseModel):
    article_id: int
    title: str
    content: str
    expected_label: str
    category: str


class SamplesListResponse(BaseModel):
    status: str = "success"
    count: int
    samples: List[SampleArticle]
