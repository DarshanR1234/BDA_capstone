import os
import json
from dotenv import load_dotenv
from groq import Groq


load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not found. Add it to the .env file."
    )

client = Groq(api_key=API_KEY)

MODEL = "openai/gpt-oss-120b"


SYSTEM_PROMPT = """
You are a professional fact-checking claim extraction system.

Your task is ONLY to extract factual claims from a news article.

You must NOT decide whether the claim is true or false.

Rules:

1. Identify the MAIN factual assertion made by the article.
2. If the article is a fact-check article, extract the ORIGINAL CLAIM
   being fact-checked, NOT the fact-checker's conclusion.
3. Preserve:
   - people
   - organizations
   - places
   - dates
   - numbers
   - events
   - laws/policies
   - specific factual details
4. Do not invent information.
5. Do not add facts from your own knowledge.
6. Ignore opinions, emotional language and commentary.
7. The main claim should be a complete factual statement.
8. Extract important entities separately.
9. Extract important numbers/dates separately.
10. If there are multiple claims, identify the central claim and list
    the remaining important factual assertions as sub_claims.

Return ONLY valid JSON.

JSON format:

{
    "main_claim": "string",
    "sub_claims": ["string"],
    "entities": ["string"],
    "numbers_dates": ["string"],
    "claim_type": "event|statement|policy|statistics|prediction|other",
    "is_fact_check_article": true,
    "claim_status": "unknown"
}
"""


def extract_claim(news_text: str) -> dict:

    if not news_text or not news_text.strip():
        raise ValueError("news_text cannot be empty.")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": news_text[:8000]
            }
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content

    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Claim extractor returned invalid JSON:\n{content}"
        ) from exc

    required_fields = [
        "main_claim",
        "sub_claims",
        "entities",
        "numbers_dates",
        "claim_type",
        "is_fact_check_article",
        "claim_status"
    ]

    for field in required_fields:
        if field not in result:
            result[field] = [] if field in [
                "sub_claims",
                "entities",
                "numbers_dates"
            ] else ""

    if not result["main_claim"]:
        raise RuntimeError(
            "Claim extraction failed: main_claim is empty."
        )

    return result