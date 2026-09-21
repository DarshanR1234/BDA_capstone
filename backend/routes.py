import time
from typing import List
from fastapi import APIRouter, HTTPException, status

from backend.schemas import (
    VerificationRequest,
    VerificationResponse,
    EvidenceItem,
    HealthResponse,
    SampleArticle,
    SamplesListResponse,
)
from ai.pipeline import verify_article

router = APIRouter(prefix="/api", tags=["Verification"])

START_TIME = time.time()

# Curated benchmark samples from the 67,587-article dataset
CURATED_SAMPLES: List[SampleArticle] = [
    SampleArticle(
        article_id=34373,
        title="वेस्टइंडीज दौरे पर कोरोना संक्रमित हुए पाकिस्तान के मुख्य कोच मिस्बाह-उल-हक",
        content="पाकिस्तान क्रिकेट टीम के मुख्य कोच मिस्बाह-उल-हक वेस्टइंडीज दौरे के समापन पर कोरोना वायरस पॉजिटिव पाए गए हैं। पाकिस्तान क्रिकेट बोर्ड (पीसीबी) ने बुधवार को इस बात की पुष्टि की। मिस्बाह अब बाकी टीम के साथ स्वदेश नहीं लौट पाएंगे और उन्हें जमैका में ही 10 दिन के लिए पृथकवास में रहना होगा।",
        expected_label="REAL",
        category="Sports / Cricket",
    ),
    SampleArticle(
        article_id=67103,
        title="क्या CAA विरोध प्रदर्शन के दौरान बंगाल में हिंदुओं को घर से बाहर निकालकर मारा गया, जानिए वायरल तस्वीर का सच...",
        content="सोशल मीडिया पर एक तस्वीर तेजी से वायरल हो रही है जिसमें सड़क पर कुछ घायल लोग दिखाई दे रहे हैं। दावा किया जा रहा है कि पश्चिम बंगाल में नागरिकता संशोधन कानून (CAA) के विरोध में हुए हिंसक प्रदर्शन के दौरान अल्पसंख्यक समुदाय के लोगों ने हिंदुओं को घरों से बाहर खींचकर मारा।",
        expected_label="FAKE",
        category="Fact Check / Politics",
    ),
    SampleArticle(
        article_id=40854,
        title="पाक आर्मी चीफ़ का कैमरे पर क़बूलनामा, कहा-हम जेहाद के नाम पर बहुत बड़े तबके को कट्टर बना चुके हैं",
        content="पाकिस्तान के पूर्व सेना प्रमुख और राष्ट्रपति परवेज मुशर्रफ का एक पुराना वीडियो वायरल हो रहा है जिसमें वे स्वीकार कर रहे हैं कि पाकिस्तान ने कश्मीर में आतंकी गतिविधियों को बढ़ावा दिया और जेहादियों को राष्ट्रीय नायक का दर्जा दिया।",
        expected_label="REAL",
        category="Geopolitics / Defense",
    ),
    SampleArticle(
        article_id=58213,
        title="'अगले इक्कीस दिनों तक घरों से न निकलें,' मोदी ने कहा",
        content="सोशल मीडिया प्लेटफॉर्म्स पर संदेश प्रसारित किया जा रहा है कि प्रधानमंत्री नरेंद्र मोदी ने एक बार फिर पूरे देश में 21 दिनों के पूर्ण लॉकडाउन की घोषणा कर दी है और लोगों से घरों में रहने की अपील की है।",
        expected_label="FAKE",
        category="Viral Claim / Misinformation",
    ),
]


@router.get("/health", response_model=HealthResponse, summary="System Health & Readiness")
async def health_check():
    """
    Returns API status, uptime, and RAG pipeline readiness.
    """
    uptime = round(time.time() - START_TIME, 2)
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=uptime,
        model_loaded=True,
        dataset_rows=67587,
    )


@router.get("/samples", response_model=SamplesListResponse, summary="Get Curated Test Articles")
async def get_sample_articles():
    """
    Returns curated real-world benchmark articles for instant 1-click testing in UI.
    """
    return SamplesListResponse(
        status="success",
        count=len(CURATED_SAMPLES),
        samples=CURATED_SAMPLES,
    )


@router.post("/verify", response_model=VerificationResponse, summary="Verify News Article (AI RAG)")
async def verify_news(request: VerificationRequest):
    """
    End-to-End Evidence-Grounded Verification:
    1. Claim & Entity Extraction (LLM)
    2. Hybrid Evidence Retrieval across 67,587 articles
    3. Multi-Signal Evidence Reranking
    4. Grounded Reasoning Analysis (LLM)
    5. Returns REAL / FAKE / UNCERTAIN with Confidence & Evidence.
    """
    start_t = time.time()

    eff_title = (request.title or "").strip()
    eff_content = (request.content or "").strip()

    if request.question and request.question.strip():
        if not eff_title:
            eff_title = request.question.strip()
        if not eff_content:
            eff_content = request.question.strip()

    if not eff_title and not eff_content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Please provide at least a 'question', 'title', or 'content' to verify.",
        )

    try:
        raw_result = verify_article(
            article_id=request.article_id,
            title=eff_title,
            content=eff_content,
            url=request.url or "",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification pipeline failed: {str(exc)}",
        )

    latency = round(time.time() - start_t, 3)

    claim_data = raw_result.get("claim_data", {})
    analysis = raw_result.get("analysis", {})
    raw_evidence = raw_result.get("evidence", [])

    formatted_evidence = []
    for item in raw_evidence:
        formatted_evidence.append(
            EvidenceItem(
                rank=item.get("rank", 1),
                evidence_id=item.get("evidence_id"),
                article_id=item.get("article_id"),
                title=item.get("title", ""),
                content_snippet=item.get("content", "")[:350] if item.get("content") else "",
                url=item.get("url", ""),
                semantic_score=item.get("semantic_score", 0.0),
                lexical_score=item.get("lexical_score", 0.0),
                claim_match=item.get("claim_match", item.get("claim_match_score", 0.0)),
                entity_match=item.get("entity_match", item.get("entity_match_score", 0.0)),
                number_date_match=item.get("number_date_match", item.get("number_date_match_score", 0.0)),
                evidence_score=item.get("evidence_score", 0.0),
                evidence_type=item.get("evidence_type", "NEWS"),
                strong_evidence=item.get("strong_evidence", False),
            )
        )

    return VerificationResponse(
        status="success",
        verdict=analysis.get("verdict", "UNCERTAIN"),
        confidence=float(analysis.get("confidence", 0.5)),
        claim=raw_result.get("claim", request.title),
        entities=claim_data.get("entities", []),
        numbers_dates=claim_data.get("numbers_dates", []),
        explanation=analysis.get("explanation", "Verification completed."),
        summary=analysis.get("summary", ""),
        evidence=formatted_evidence,
        latency_seconds=latency,
    )
