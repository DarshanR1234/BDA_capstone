import json
import os
import re

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not found in .env"
    )


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "openai/gpt-oss-120b"

client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(text):
    if text is None:
        return ""

    return str(text).strip()


def truncate_text(
    text,
    max_chars=3000,
):
    text = clean_text(text)

    if len(text) <= max_chars:
        return text

    return (
        text[: max_chars // 2]
        + "\n...[TRUNCATED]...\n"
        + text[-max_chars // 2 :]
    )


# ============================================================
# EVIDENCE EXCERPT
# ============================================================

def build_evidence_excerpt(
    content,
    max_chars=3000,
):

    content = clean_text(content)

    if len(content) <= max_chars:
        return content

    half = max_chars // 2

    return (
        content[:half]
        + "\n...[TRUNCATED]...\n"
        + content[-half:]
    )


# ============================================================
# EVIDENCE BLOCK
# ============================================================

def build_evidence_block(
    evidence,
):

    if not evidence:
        return "NO EVIDENCE WAS RETRIEVED."

    blocks = []

    for position, item in enumerate(
        evidence,
        start=1,
    ):

        # ----------------------------------------------------
        # Evidence ID
        #
        # New retriever uses IDs such as:
        # E29602
        #
        # Keep the ID as a string.
        # ----------------------------------------------------

        evidence_id = str(
            item.get(
                "evidence_id",
                f"E{position}",
            )
        )

        article_id = item.get(
            "article_id",
            "UNKNOWN",
        )

        title = clean_text(
            item.get(
                "title",
                "",
            )
        )

        content = build_evidence_excerpt(
            item.get(
                "content",
                "",
            ),
            max_chars=3000,
        )

        url = clean_text(
            item.get(
                "url",
                "",
            )
        )

        label = item.get(
            "label_name",
            "UNKNOWN",
        )

        evidence_type = item.get(
            "evidence_type",
            "UNKNOWN",
        )

        semantic_score = float(
            item.get(
                "semantic_score",
                0.0,
            )
        )

        lexical_score = float(
            item.get(
                "lexical_score",
                0.0,
            )
        )

        entity_match = float(
            item.get(
                "entity_match",
                0.0,
            )
        )

        number_date_match = float(
            item.get(
                "number_date_match",
                0.0,
            )
        )

        claim_match = float(
            item.get(
                "claim_match",
                item.get(
                    "claim_overlap",
                    0.0,
                ),
            )
        )

        evidence_score = float(
            item.get(
                "evidence_score",
                item.get(
                    "relevance_score",
                    0.0,
                ),
            )
        )

        strong_evidence = item.get(
            "strong_evidence",
            False,
        )

        block = f"""
EVIDENCE {position}
--------------------
Evidence ID: {evidence_id}
Article ID: {article_id}

Title:
{title}

URL:
{url}

Dataset label:
{label}

Evidence type:
{evidence_type}

Retrieval information:
- Semantic score: {semantic_score:.4f}
- Lexical score: {lexical_score:.4f}
- Entity match: {entity_match:.4f}
- Number/date match: {number_date_match:.4f}
- Claim match: {claim_match:.4f}
- Evidence score: {evidence_score:.4f}
- Strong evidence: {strong_evidence}

Article content:
{content}
"""

        blocks.append(
            block.strip()
        )

    return "\n\n".join(
        blocks
    )


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an evidence-grounded news verification system.

Your task is to determine whether the MAIN CLAIM in a news article is:

- FAKE
- REAL
- UNCERTAIN

You MUST use only the supplied article and supplied evidence.

Do NOT use outside knowledge.

IMPORTANT RULES:

1. Reporting a claim does not prove that the claim is true.

2. A similar topic is NOT sufficient evidence.

3. Evidence must refer to the same people, organizations,
   locations, dates, numbers, events, or specific claim whenever
   possible.

4. Do not treat semantic similarity alone as proof.

5. Do not treat the dataset label of an evidence article as proof.
   Dataset labels are metadata only.

6. An explicit fact-check article about the SAME claim can be
   strong evidence.

7. A fact-check article about a DIFFERENT event, location,
   person, date, or claim must NOT be treated as proof.

8. If evidence directly contradicts the main claim, the verdict
   can be FAKE.

9. If evidence directly supports the main claim, the verdict
   can be REAL.

10. If the evidence is related but does not establish the exact
    claim, return UNCERTAIN.

11. Conflicting direct evidence should normally result in
    UNCERTAIN unless the evidence clearly establishes one side.

12. Historical claims must be judged according to the supplied
    evidence. Do not assume that an article is fake merely
    because its dataset label is FAKE.

13. Do not simply copy the dataset label.

14. Do not invent evidence.

15. Do not claim that evidence proves something it does not
    actually state.

16. Be especially careful with:
    - names
    - locations
    - dates
    - numbers
    - political statements
    - deaths
    - attacks
    - government announcements
    - quotations
    - photographs
    - videos

Return ONLY valid JSON.

Required JSON structure:

{
    "verdict": "FAKE | REAL | UNCERTAIN",
    "confidence": 0.0,
    "claim": "main claim",
    "evidence_analysis": [
        {
            "evidence_id": "E123",
            "relationship": "supports | contradicts | unrelated | insufficient"
        }
    ],
    "explanation": "Detailed evidence-grounded explanation.",
    "summary": "Short final explanation."
}

Confidence must be a number between 0.0 and 1.0.
"""


# ============================================================
# JSON CLEANING
# ============================================================

def extract_json(text):

    text = clean_text(text)

    # Remove markdown fences if model adds them.
    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"^```\s*",
        "",
        text,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    text = text.strip()

    # Direct JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find first JSON object.
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:

        candidate = text[
            start : end + 1
        ]

        try:
            return json.loads(
                candidate
            )
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "LLM response did not contain valid JSON."
    )


# ============================================================
# VALIDATE ANALYSIS
# ============================================================

def validate_analysis(
    data,
    claim,
):

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            "LLM response is not a JSON object."
        )

    verdict = str(
        data.get(
            "verdict",
            "UNCERTAIN",
        )
    ).upper().strip()

    if verdict not in {
        "FAKE",
        "REAL",
        "UNCERTAIN",
    }:

        verdict = "UNCERTAIN"

    try:

        confidence = float(
            data.get(
                "confidence",
                0.0,
            )
        )

    except Exception:

        confidence = 0.0

    confidence = max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )

    evidence_analysis = data.get(
        "evidence_analysis",
        [],
    )

    if not isinstance(
        evidence_analysis,
        list,
    ):
        evidence_analysis = []

    cleaned_evidence = []

    for item in evidence_analysis:

        if not isinstance(
            item,
            dict,
        ):
            continue

        evidence_id = str(
            item.get(
                "evidence_id",
                "",
            )
        )

        relationship = str(
            item.get(
                "relationship",
                "insufficient",
            )
        ).lower().strip()

        if relationship not in {
            "supports",
            "contradicts",
            "unrelated",
            "insufficient",
        }:

            relationship = "insufficient"

        cleaned_evidence.append(
            {
                "evidence_id": evidence_id,
                "relationship": relationship,
            }
        )

    explanation = clean_text(
        data.get(
            "explanation",
            "",
        )
    )

    summary = clean_text(
        data.get(
            "summary",
            "",
        )
    )

    if not explanation:
        explanation = (
            "The available evidence was insufficient "
            "to establish the claim."
        )

    if not summary:
        summary = explanation

    return {
        "verdict": verdict,
        "confidence": round(
            confidence,
            4,
        ),
        "claim": claim,
        "evidence_analysis": cleaned_evidence,
        "explanation": explanation,
        "summary": summary,
    }


# ============================================================
# MAIN ANALYZER
# ============================================================

def analyze_claim(
    claim,
    news_text,
    evidence,
):

    claim = clean_text(
        claim
    )

    news_text = clean_text(
        news_text
    )

    if not claim:

        raise ValueError(
            "analyze_claim() received an empty claim."
        )

    evidence_block = build_evidence_block(
        evidence
    )

    user_prompt = f"""
MAIN CLAIM
==========
{claim}

SOURCE NEWS ARTICLE
===================
{truncate_text(
    news_text,
    max_chars=5000,
)}

SUPPLIED EVIDENCE
=================
{evidence_block}

TASK
====
Determine whether the MAIN CLAIM is FAKE, REAL, or UNCERTAIN.

Base your decision only on the supplied evidence.

Remember:

- Same topic is not enough.
- Same event matters.
- Same person matters.
- Same location matters.
- Same date matters.
- Same numbers matter.
- Explicit fact-checking of the exact claim is strong evidence.
- Dataset labels are metadata, not proof.

Return only valid JSON.
"""

    # ========================================================
    # FIRST REQUEST
    # ========================================================

    try:

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0,
            response_format={
                "type": "json_object"
            },
        )

        raw_content = (
            response.choices[0]
            .message
            .content
        )

        parsed = extract_json(
            raw_content
        )

        return validate_analysis(
            parsed,
            claim,
        )

    except Exception as first_error:

        # ====================================================
        # RETRY WITH SMALLER PROMPT
        # ====================================================

        retry_prompt = f"""
Verify this claim using ONLY the evidence below.

CLAIM:
{claim}

EVIDENCE:
{evidence_block}

Return JSON only:

{{
  "verdict": "FAKE | REAL | UNCERTAIN",
  "confidence": 0.0,
  "claim": "{claim}",
  "evidence_analysis": [],
  "explanation": "",
  "summary": ""
}}

If the evidence does not establish the exact claim,
use UNCERTAIN.
"""

        try:

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": retry_prompt,
                    },
                ],
                temperature=0,
                response_format={
                    "type": "json_object"
                },
            )

            raw_content = (
                response.choices[0]
                .message
                .content
            )

            parsed = extract_json(
                raw_content
            )

            return validate_analysis(
                parsed,
                claim,
            )

        except Exception as second_error:

            raise RuntimeError(
                "LLM analysis failed.\n"
                f"First attempt: {first_error}\n"
                f"Retry attempt: {second_error}"
            )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    test_claim = (
        "The government announced a new rule "
        "for schools."
    )

    test_evidence = [
        {
            "evidence_id": "E123",
            "article_id": 123,
            "title": "Example evidence article",
            "content": (
                "The government announced "
                "a new rule for schools."
            ),
            "url": "",
            "label_name": "REAL",
            "evidence_type": "NEWS",
            "semantic_score": 0.90,
            "lexical_score": 0.80,
            "entity_match": 1.0,
            "number_date_match": 1.0,
            "claim_match": 0.90,
            "evidence_score": 0.90,
            "strong_evidence": True,
        }
    ]

    result = analyze_claim(
        claim=test_claim,
        news_text=test_claim,
        evidence=test_evidence,
    )

    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False,
        )
    )