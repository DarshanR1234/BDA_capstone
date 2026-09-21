import os
import sys
import pandas as pd


# =========================================================
# Project root
# =========================================================

PROJECT_ROOT = r"D:\fake-news-bda"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


from ai.pipeline import verify_article


# =========================================================
# Dataset
# =========================================================

DATASET_PATH = (
    r"D:\fake-news-bda\data\processed\news_cleaned.csv"
)


# =========================================================
# Test article IDs
# =========================================================

TEST_IDS = [
    3231,
    40854,
    67103,
    34373,
    40135,
    58213
]


# =========================================================
# Load dataset
# =========================================================

print("\n")
print("=" * 80)
print("FAKE NEWS BDA — AI/RAG PIPELINE TEST")
print("=" * 80)

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET_PATH,
    encoding="utf-8-sig"
)

print(
    f"Dataset rows: {len(df)}"
)


# =========================================================
# Validate dataset
# =========================================================

if len(df) != 67587:
    raise RuntimeError(
        f"Expected 67587 rows, got {len(df)}"
    )

required_columns = [
    "article_id",
    "url",
    "title",
    "content",
    "label"
]

for column in required_columns:

    if column not in df.columns:
        raise RuntimeError(
            f"Missing column: {column}"
        )


print(
    "Dataset validation: PASS"
)


# =========================================================
# Test each article
# =========================================================

for article_id in TEST_IDS:

    row = df[
        df["article_id"] == article_id
    ]

    if row.empty:

        print(
            f"\nArticle {article_id} NOT FOUND."
        )

        continue

    row = row.iloc[0]

    title = str(
        row["title"]
    )

    content = str(
        row["content"]
    )

    url = str(
        row["url"]
    )

    dataset_label = int(
        row["label"]
    )

    dataset_label_name = (
        "FAKE"
        if dataset_label == 0
        else "REAL"
    )

    print("\n")
    print("=" * 80)
    print(
        f"ARTICLE ID: {article_id}"
    )
    print("=" * 80)

    print(
        f"\nDataset label: "
        f"{dataset_label_name}"
    )

    print(
        f"\nTitle:\n{title[:500]}"
    )

    try:

        result = verify_article(
            article_id=article_id,
            title=title,
            content=content,
            url=url
        )

    except Exception as exc:

        print(
            "\nPIPELINE ERROR:"
        )

        print(
            repr(exc)
        )

        continue

    # =====================================================
    # Claim
    # =====================================================

    print("\n")
    print("-" * 80)
    print("EXTRACTED CLAIM")
    print("-" * 80)

    print(
        result["claim"]
    )

    claim_data = result[
        "claim_data"
    ]

    print(
        "\nEntities:",
        claim_data.get(
            "entities",
            []
        )
    )

    print(
        "Dates/Numbers:",
        claim_data.get(
            "numbers_dates",
            []
        )
    )

    # =====================================================
    # Evidence
    # =====================================================

    print("\n")
    print("-" * 80)
    print("SELECTED EVIDENCE")
    print("-" * 80)

    evidence = result[
        "evidence"
    ]

    if not evidence:

        print(
            "No evidence selected."
        )

    else:

        for item in evidence:

            print(
                f"\n[{item['rank']}] "
                f"Evidence ID: "
                f"{item['evidence_id']}"
            )

            print(
                f"Title: "
                f"{item['title'][:300]}"
            )

            print(
                f"Semantic: "
                f"{item.get('semantic_score', 0):.3f}"
            )

            print(
                f"Lexical: "
                f"{item.get('lexical_score', 0):.3f}"
            )

            print(
                f"Entity Match: "
                f"{float(item.get('entity_match', item.get('entity_match_score', 0))):.3f}"
            )

            print(
                f"Date/Number Match: "
                f"{float(item.get('number_date_match', item.get('number_date_match_score', 0))):.3f}"
            )

            print(
                f"Claim Match: "
                f"{float(item.get('claim_match', item.get('claim_match_score', item.get('phrase_match_score', 0)))):.3f}"
            )

            print(
                f"Evidence Score: "
                f"{item.get('evidence_score', 0):.3f}"
            )

            print(
                f"Evidence Type: "
                f"{item.get('evidence_type', 'NEWS')}"
            )

            print(
                f"Strong Evidence: "
                f"{item.get('strong_evidence', False)}"
            )

    # =====================================================
    # Final AI result
    # =====================================================

    analysis = result[
        "analysis"
    ]

    print("\n")
    print("-" * 80)
    print("FINAL AI VERDICT")
    print("-" * 80)

    print(
        f"\nVerdict: "
        f"{analysis['verdict']}"
    )

    print(
        f"Confidence: "
        f"{analysis['confidence']:.2f}"
    )

    print(
        f"\nExplanation:\n"
        f"{analysis.get('explanation', '')}"
    )

    print(
        f"\nSummary:\n"
        f"{analysis.get('summary', '')}"
    )

    print("\n")
    print("=" * 80)


print("\n")
print("=" * 80)
print("TEST COMPLETED")
print("=" * 80)