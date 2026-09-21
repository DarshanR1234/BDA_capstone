from ai.claim_extractor import extract_claim
from ai.hybrid_retriever import retrieve
from ai.reranker import rerank
from ai.llm_analyzer import analyze_claim


def verify_article(
    article_id,
    title,
    content,
    url="",
):
    """
    Complete fake-news verification pipeline.

    Flow:

        Article
           ↓
        Claim Extraction
           ↓
        Hybrid Retrieval
           ↓
        Evidence Reranking
           ↓
        LLM Evidence Analysis
           ↓
        FAKE / REAL / UNCERTAIN
    """

    if article_id is not None:
        try:
            article_id = int(article_id)
        except (ValueError, TypeError):
            article_id = None

    title = str(title or "")
    content = str(content or "")
    url = str(url or "")

    article_text = (
        f"{title}\n\n{content}"
    )

    # ========================================================
    # 1. CLAIM EXTRACTION
    # ========================================================

    claim_data = extract_claim(
        article_text
    )

    claim = claim_data.get(
        "main_claim",
        ""
    )

    if not claim:
        raise ValueError(
            f"Claim extraction returned empty claim "
            f"for article {article_id}"
        )

    entities = claim_data.get(
        "entities",
        []
    )

    numbers_dates = claim_data.get(
        "numbers_dates",
        []
    )

    # ========================================================
    # 2. HYBRID RETRIEVAL
    # ========================================================

    retrieved = retrieve(
        claim=claim,
        source_title=title,
        source_content=content,
        entities=entities,
        numbers_dates=numbers_dates,
        top_k=12,
        exclude_article_id=article_id,
    )

    if retrieved is None:
        retrieved = []

    # ========================================================
    # 3. EVIDENCE RERANKING
    # ========================================================

    evidence = rerank(
        query=claim,
        candidates=retrieved,
        top_k=3,
        entities=entities,
        numbers_dates=numbers_dates,
        source_title=title,
        source_content=content,
    )

    if evidence is None:
        evidence = []

    # ========================================================
    # 4. LLM ANALYSIS
    # ========================================================

    analysis = analyze_claim(
        claim=claim,
        news_text=article_text,
        evidence=evidence,
    )

    if analysis is None:
        raise ValueError(
            f"LLM analyzer returned no result "
            f"for article {article_id}"
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {
        "article_id": article_id,
        "url": url,

        "claim": claim,

        "claim_data": claim_data,

        "evidence": evidence,

        "analysis": analysis,
    }