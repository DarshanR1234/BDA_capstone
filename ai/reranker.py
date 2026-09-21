# ai/reranker.py

import re
import unicodedata
from typing import Any, Dict, List


# ============================================================
# CONFIGURATION
# ============================================================

MIN_CLAIM_MATCH = 0.18
MIN_STRONG_RELEVANCE = 0.45

FACT_CHECK_TERMS = [
    "fact check",
    "fact-check",
    "factcheck",
    "fact check:",
    "पड़ताल",
    "फैक्ट चेक",
    "फैक्टचेक",
    "सच क्या है",
    "वायरल दावे का सच",
    "वायरल तस्वीर का सच",
    "दावे की सच्चाई",
    "झूठा दावा",
    "गलत दावा",
    "सच्चाई क्या है",
]

FALSE_TERMS = [
    "false",
    "fake",
    "fabricated",
    "misleading",
    "false claim",
    "fake news",
    "incorrect",
    "wrong",
    "गलत",
    "झूठा",
    "झूठी",
    "फर्जी",
    "भ्रामक",
    "नकली",
]

TRUE_TERMS = [
    "true",
    "correct",
    "verified",
    "accurate",
    "सही",
    "सच",
    "सत्य",
    "पुष्टि",
]

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "into", "after", "before", "during",
    "said", "says", "was", "were", "has", "have", "had", "will", "would", "could", "should",
    "about", "their", "there", "they", "them", "india", "indian", "according", "report", "reports",
    "claim", "claims", "news", "और", "यह", "इस", "उस", "एक", "के", "की", "का", "में", "से",
    "पर", "को", "भी", "है", "हैं", "था", "थी", "थे", "हुए", "हुआ", "हुई", "ने", "हो", "गया", "गई"
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: Any) -> str:

    if text is None:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        str(text)
    )

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def tokenize(text: str) -> List[str]:

    return re.findall(
        r"[^\W_]+",
        normalize_text(text),
        flags=re.UNICODE
    )


def token_set(text: str) -> set:

    return set(
        tokenize(text)
    )


def get_float(
    item: Dict[str, Any],
    *keys,
    default: float = 0.0
) -> float:

    for key in keys:

        value = item.get(key)

        if value is None:
            continue

        try:
            return float(value)
        except (
            TypeError,
            ValueError
        ):
            continue

    return default


# ============================================================
# EVIDENCE ID
# ============================================================

def get_evidence_id(
    candidate: Dict[str, Any]
) -> str:

    evidence_id = candidate.get(
        "evidence_id"
    )

    if evidence_id is not None:

        evidence_id = str(
            evidence_id
        ).strip()

        if evidence_id:
            return evidence_id

    article_id = candidate.get(
        "article_id"
    )

    if article_id is not None:

        return f"E{article_id}"

    return "EUNKNOWN"


# ============================================================
# CLAIM OVERLAP
# ============================================================

def calculate_claim_overlap(
    query: str,
    title: str,
    content: str
) -> float:

    query_tokens = token_set(
        query
    )

    if not query_tokens:
        return 0.0

    distinctive_tokens = {
        t for t in query_tokens
        if t not in STOPWORDS and len(t) > 1
    }
    eval_tokens = distinctive_tokens if distinctive_tokens else query_tokens

    title_tokens = token_set(
        title
    )

    content_tokens = token_set(
        content
    )

    # Title is more important than arbitrary
    # content-word overlap.

    title_overlap = (
        len(
            eval_tokens
            & title_tokens
        )
        / len(eval_tokens)
    )

    content_overlap = (
        len(
            eval_tokens
            & content_tokens
        )
        / len(eval_tokens)
    )

    score = (
        0.70 * title_overlap
        + 0.30 * content_overlap
    )

    return min(
        max(score, 0.0),
        1.0
    )


# ============================================================
# DIRECT TITLE MATCH
# ============================================================

def calculate_directness(
    query: str,
    title: str
) -> float:

    q_tokens = token_set(
        query
    )

    t_tokens = token_set(
        title
    )

    if not q_tokens or not t_tokens:
        return 0.0

    return min(
        len(q_tokens & t_tokens)
        / len(q_tokens),
        1.0
    )


# ============================================================
# FACT CHECK DETECTION
# ============================================================

def contains_terms(
    text: str,
    terms: List[str]
) -> bool:

    text = normalize_text(
        text
    )

    for term in terms:

        if normalize_text(term) in text:
            return True

    return False


def calculate_fact_check_score(
    title: str,
    content: str
) -> float:

    title = normalize_text(
        title
    )

    content = normalize_text(
        content
    )

    title_match = contains_terms(
        title,
        FACT_CHECK_TERMS
    )

    content_match = contains_terms(
        content,
        FACT_CHECK_TERMS
    )

    if title_match:
        return 1.0

    if content_match:
        return 0.70

    return 0.0


# ============================================================
# FACT CHECK VERDICT TYPE
# ============================================================

def calculate_false_score(
    title: str,
    content: str
) -> float:

    text = normalize_text(
        f"{title} {content}"
    )

    matches = 0

    for term in FALSE_TERMS:

        if normalize_text(term) in text:
            matches += 1

    if matches == 0:
        return 0.0

    return min(
        0.5 + (matches * 0.10),
        1.0
    )


def calculate_true_score(
    title: str,
    content: str
) -> float:

    text = normalize_text(
        f"{title} {content}"
    )

    matches = 0

    for term in TRUE_TERMS:

        if normalize_text(term) in text:
            matches += 1

    if matches == 0:
        return 0.0

    return min(
        0.5 + (matches * 0.10),
        1.0
    )


# ============================================================
# EVIDENCE TYPE
# ============================================================

def classify_evidence_type(
    title: str,
    content: str,
    existing_type: str = ""
) -> str:

    existing_type = str(
        existing_type or ""
    ).upper().strip()

    # Preserve the type produced by the retriever
    # whenever it is already meaningful.

    if existing_type in {
        "FACT_CHECK",
        "FACT_CHECK_FALSE",
        "FACT_CHECK_TRUE",
        "NEWS",
    }:
        return existing_type

    fact_score = calculate_fact_check_score(
        title,
        content
    )

    if fact_score >= 0.70:

        false_score = calculate_false_score(
            title,
            content
        )

        true_score = calculate_true_score(
            title,
            content
        )

        if false_score >= true_score and false_score >= 0.55:
            return "FACT_CHECK_FALSE"

        if true_score > false_score and true_score >= 0.55:
            return "FACT_CHECK_TRUE"

        return "FACT_CHECK"

    return "NEWS"


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def clamp(
    value: float
) -> float:

    return min(
        max(value, 0.0),
        1.0
    )


# ============================================================
# MAIN EVIDENCE SCORE
# ============================================================

def calculate_final_score(
    semantic_score: float,
    lexical_score: float,
    claim_lexical_score: float,
    phrase_score: float,
    entity_score: float,
    number_date_score: float,
    fact_check_score: float,
    directness_score: float,
    retriever_relevance: float
) -> float:

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # These signals are deliberately combined instead of
    # relying on semantic similarity alone.
    # --------------------------------------------------------

    score = (

        0.15 * semantic_score

        + 0.10 * lexical_score

        + 0.22 * claim_lexical_score

        + 0.15 * phrase_score

        + 0.15 * entity_score

        + 0.09 * number_date_score

        + 0.10 * fact_check_score

        + 0.04 * directness_score
    )

    return clamp(
        score
    )


# ============================================================
# STRONG EVIDENCE DECISION
# ============================================================

def determine_strong_evidence(
    evidence_type: str,
    final_score: float,
    semantic_score: float,
    lexical_score: float,
    claim_lexical_score: float,
    phrase_score: float,
    entity_score: float,
    number_date_score: float,
    fact_check_score: float,
    directness_score: float
) -> bool:

    # ========================================================
    # FACT CHECK EVIDENCE
    # ========================================================

    if evidence_type in {
        "FACT_CHECK",
        "FACT_CHECK_FALSE",
        "FACT_CHECK_TRUE",
    }:

        # A fact-check must still be about the same claim.
        if (
            claim_lexical_score >= 0.20
            and (
                phrase_score >= 0.10
                or entity_score >= 0.20
                or lexical_score >= 0.15
            )
            and final_score >= 0.30
        ):
            return True

    # ========================================================
    # VERY STRONG NORMAL NEWS
    # ========================================================

    if (
        final_score >= 0.55
        and semantic_score >= 0.70
        and claim_lexical_score >= 0.30
        and (
            entity_score >= 0.20
            or phrase_score >= 0.20
            or number_date_score >= 0.20
        )
    ):
        return True

    # ========================================================
    # EXACT ENTITY + CLAIM MATCH
    # ========================================================

    if (
        final_score >= 0.45
        and claim_lexical_score >= 0.35
        and entity_score >= 0.30
        and (
            lexical_score >= 0.15
            or phrase_score >= 0.15
        )
    ):
        return True

    # ========================================================
    # EXACT PHRASE / NUMBER / DATE SUPPORT
    # ========================================================

    if (
        final_score >= 0.45
        and claim_lexical_score >= 0.30
        and (
            phrase_score >= 0.30
            or number_date_score >= 0.40
        )
        and (
            entity_score >= 0.20
            or directness_score >= 0.30
        )
    ):
        return True

    return False


# ============================================================
# RERANK
# ============================================================

def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 3,
    entities: List[str] = None,
    numbers_dates: List[str] = None,
    source_title: str = "",
    source_content: str = ""
) -> List[Dict[str, Any]]:

    if not candidates:
        return []

    entities = entities or []
    numbers_dates = numbers_dates or []

    ranked = []

    for candidate in candidates:

        title = str(
            candidate.get(
                "title",
                ""
            ) or ""
        )

        content = str(
            candidate.get(
                "content",
                ""
            ) or ""
        )

        # ====================================================
        # PRESERVE RETRIEVER SIGNALS
        # ====================================================

        semantic_score = clamp(
            get_float(
                candidate,
                "semantic_score"
            )
        )

        lexical_score = clamp(
            get_float(
                candidate,
                "lexical_score"
            )
        )

        claim_lexical_score = clamp(
            get_float(
                candidate,
                "claim_lexical_score",
                "claim_lexical"
            )
        )

        phrase_score = clamp(
            get_float(
                candidate,
                "phrase_match_score",
                "phrase_score"
            )
        )

        entity_score = clamp(
            get_float(
                candidate,
                "entity_match_score",
                "entity_match"
            )
        )

        number_date_score = clamp(
            get_float(
                candidate,
                "number_date_match_score",
                "number_date_match"
            )
        )

        retriever_relevance = clamp(
            get_float(
                candidate,
                "relevance_score",
                "retrieval_score"
            )
        )

        # ====================================================
        # FALLBACK CLAIM MATCH
        # ====================================================

        calculated_claim_match = (
            calculate_claim_overlap(
                query,
                title,
                content
            )
        )

        # Use the retriever score when available.
        if claim_lexical_score > 0.0:

            claim_match = max(
                claim_lexical_score,
                calculated_claim_match
            )

        else:

            claim_match = calculated_claim_match

        claim_match = clamp(
            claim_match
        )

        # ====================================================
        # DIRECTNESS
        # ====================================================

        directness_score = (
            calculate_directness(
                query,
                title
            )
        )

        # ====================================================
        # FACT CHECK
        # ====================================================

        existing_fact_score = get_float(
            candidate,
            "fact_check_score"
        )

        detected_fact_score = (
            calculate_fact_check_score(
                title,
                content
            )
        )

        fact_score = max(
            existing_fact_score,
            detected_fact_score
        )

        # ====================================================
        # EVIDENCE TYPE
        # ====================================================

        evidence_type = classify_evidence_type(
            title=title,
            content=content,
            existing_type=candidate.get(
                "evidence_type",
                ""
            )
        )

        # ====================================================
        # FINAL SCORE
        # ====================================================

        final_score = calculate_final_score(

            semantic_score=semantic_score,

            lexical_score=lexical_score,

            claim_lexical_score=claim_match,

            phrase_score=phrase_score,

            entity_score=entity_score,

            number_date_score=number_date_score,

            fact_check_score=fact_score,

            directness_score=directness_score,

            retriever_relevance=retriever_relevance
        )

        # ====================================================
        # STRONG EVIDENCE
        # ====================================================

        strong = determine_strong_evidence(

            evidence_type=evidence_type,

            final_score=final_score,

            semantic_score=semantic_score,

            lexical_score=lexical_score,

            claim_lexical_score=claim_match,

            phrase_score=phrase_score,

            entity_score=entity_score,

            number_date_score=number_date_score,

            fact_check_score=fact_score,

            directness_score=directness_score
        )

        # ====================================================
        # RESULT
        # ====================================================

        result = dict(
            candidate
        )

        # GUARANTEE ID

        result["evidence_id"] = (
            get_evidence_id(
                candidate
            )
        )

        # Preserve article ID

        if candidate.get(
            "article_id"
        ) is not None:

            try:

                result["article_id"] = int(
                    candidate["article_id"]
                )

            except (
                TypeError,
                ValueError
            ):

                result["article_id"] = (
                    candidate["article_id"]
                )

        # ====================================================
        # STANDARDIZED OUTPUT
        # ====================================================

        result.update({

            "semantic_score": round(
                semantic_score,
                4
            ),

            "lexical_score": round(
                lexical_score,
                4
            ),

            "claim_lexical_score": round(
                claim_match,
                4
            ),

            "claim_match": round(
                claim_match,
                4
            ),

            "claim_match_score": round(
                claim_match,
                4
            ),

            "phrase_match_score": round(
                phrase_score,
                4
            ),

            "entity_match_score": round(
                entity_score,
                4
            ),

            "entity_match": round(
                entity_score,
                4
            ),

            "number_date_match_score": round(
                number_date_score,
                4
            ),

            "number_date_match": round(
                number_date_score,
                4
            ),

            "fact_check_score": round(
                fact_score,
                4
            ),

            "directness_score": round(
                directness_score,
                4
            ),

            "retriever_relevance": round(
                retriever_relevance,
                4
            ),

            "evidence_score": round(
                final_score,
                4
            ),

            "relevance_score": round(
                final_score,
                4
            ),

            "evidence_type": evidence_type,

            "strong_evidence": bool(
                strong
            ),
        })

        ranked.append(
            result
        )

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    unique = {}

    for item in ranked:

        evidence_id = item[
            "evidence_id"
        ]

        if evidence_id not in unique:

            unique[evidence_id] = item

        else:

            # Keep higher-scoring duplicate.
            if (
                item["evidence_score"]
                >
                unique[evidence_id][
                    "evidence_score"
                ]
            ):
                unique[evidence_id] = item

    ranked = list(
        unique.values()
    )

    # ========================================================
    # SORT
    # ========================================================

    ranked.sort(
        key=lambda item: (
            int(
                bool(
                    item.get(
                        "strong_evidence",
                        False
                    )
                )
            ),

            float(
                item.get(
                    "evidence_score",
                    0.0
                )
            ),

            float(
                item.get(
                    "claim_match",
                    0.0
                )
            ),

            float(
                item.get(
                    "entity_match",
                    0.0
                )
            ),

            float(
                item.get(
                    "phrase_match_score",
                    0.0
                )
            )
        ),
        reverse=True
    )

    # ========================================================
    # IMPORTANT SELECTION LOGIC
    # ========================================================
    #
    # We do NOT return only one weak result.
    #
    # The LLM needs several pieces of evidence to determine
    # whether the claim is supported, contradicted or uncertain.
    #
    # But we prioritize strong evidence first.
    # ========================================================

    strong_items = [
        item
        for item in ranked
        if item.get(
            "strong_evidence",
            False
        )
    ]

    weak_items = [
        item
        for item in ranked
        if not item.get(
            "strong_evidence",
            False
        )
    ]

    selected = []

    # Strong evidence first

    for item in strong_items:

        if len(selected) >= top_k:
            break

        selected.append(
            item
        )

    # Fill remaining slots with best candidates

    if len(selected) < top_k:

        for item in weak_items:

            if len(selected) >= top_k:
                break

            selected.append(
                item
            )

    # ========================================================
    # ASSIGN RANK
    # ========================================================

    for rank, item in enumerate(
        selected,
        start=1
    ):

        item["rank"] = rank

    return selected


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    candidates = [

        {
            "evidence_id": "E1",
            "article_id": 1,
            "title": (
                "India GDP growth fact check: "
                "viral claim is false"
            ),
            "content": (
                "Fact check shows that the "
                "viral claim about India's "
                "GDP growth is false."
            ),
            "semantic_score": 0.70,
            "lexical_score": 0.55,
            "claim_lexical_score": 0.60,
            "phrase_match_score": 0.45,
            "entity_match_score": 0.80,
            "number_date_match_score": 0.50,
            "relevance_score": 0.75,
            "evidence_type": "FACT_CHECK_FALSE",
        },

        {
            "evidence_id": "E2",
            "article_id": 2,
            "title": (
                "India wins cricket tournament"
            ),
            "content": (
                "India won an important "
                "cricket tournament."
            ),
            "semantic_score": 0.25,
            "lexical_score": 0.02,
            "claim_lexical_score": 0.01,
            "phrase_match_score": 0.00,
            "entity_match_score": 0.50,
            "number_date_match_score": 0.00,
            "relevance_score": 0.10,
            "evidence_type": "NEWS",
        }
    ]

    results = rerank(

        query=(
            "India GDP growth viral "
            "claim is false"
        ),

        candidates=candidates,

        top_k=3,

        entities=[
            "India"
        ],

        numbers_dates=[]
    )

    print()
    print("=" * 80)
    print("RERANKER TEST")
    print("=" * 80)

    for item in results:

        print()
        print(
            "Evidence ID:",
            item["evidence_id"]
        )

        print(
            "Title:",
            item["title"]
        )

        print(
            "Semantic:",
            item["semantic_score"]
        )

        print(
            "Lexical:",
            item["lexical_score"]
        )

        print(
            "Claim:",
            item["claim_match"]
        )

        print(
            "Phrase:",
            item["phrase_match_score"]
        )

        print(
            "Entity:",
            item["entity_match"]
        )

        print(
            "Evidence Score:",
            item["evidence_score"]
        )

        print(
            "Type:",
            item["evidence_type"]
        )

        print(
            "Strong:",
            item["strong_evidence"]
        )

    print()
    print("=" * 80)
    print("Reranker validation: PASS")
    print("=" * 80)