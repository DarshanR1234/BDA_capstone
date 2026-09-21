import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\fake-news-bda")
AI_DIR = PROJECT_ROOT / "ai"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from pipeline import verify_article


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "news_cleaned.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "evaluation"
)

EXPECTED_DATASET_ROWS = 67587

# 5 FAKE + 5 REAL = 10 articles
SAMPLE_PER_CLASS = 5

RANDOM_STATE = 42


# ============================================================
# HELPERS
# ============================================================

def label_to_verdict(label):
    """
    HinFakeNews-V1 mapping:

        0 = FAKE
        1 = REAL
    """
    label = int(label)

    if label == 0:
        return "FAKE"

    if label == 1:
        return "REAL"

    return "UNKNOWN"


def verdict_to_binary(verdict):
    """
    Convert system verdict to evaluation label.

        FAKE = 0
        REAL = 1
        UNCERTAIN = None
    """
    if verdict == "FAKE":
        return 0

    if verdict == "REAL":
        return 1

    return None


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 90)
print("AI-POWERED FAKE NEWS DETECTION - EVALUATION")
print("=" * 90)

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET_PATH,
    encoding="utf-8-sig",
)

print(f"Dataset rows: {len(df):,}")

if len(df) != EXPECTED_DATASET_ROWS:
    raise ValueError(
        f"Expected {EXPECTED_DATASET_ROWS:,} rows, "
        f"found {len(df):,}"
    )

required_columns = {
    "article_id",
    "url",
    "title",
    "content",
    "label",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

if df["label"].isna().any():
    raise ValueError("Dataset contains missing labels.")

if not set(df["label"].astype(int).unique()).issubset({0, 1}):
    raise ValueError("Dataset contains labels other than 0 and 1.")

print("Dataset validation: PASS")


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

label_counts = df["label"].value_counts().sort_index()

print("\nDataset labels:")

for label, count in label_counts.items():
    label = int(label)

    if label == 0:
        print(f"  0 = FAKE : {count:,}")
    elif label == 1:
        print(f"  1 = REAL : {count:,}")


# ============================================================
# CREATE BALANCED EVALUATION SAMPLE
# ============================================================

fake_df = df[df["label"] == 0].sample(
    n=SAMPLE_PER_CLASS,
    random_state=RANDOM_STATE,
)

real_df = df[df["label"] == 1].sample(
    n=SAMPLE_PER_CLASS,
    random_state=RANDOM_STATE,
)

evaluation_df = pd.concat(
    [fake_df, real_df],
    ignore_index=True,
)

evaluation_df = evaluation_df.sample(
    frac=1,
    random_state=RANDOM_STATE,
).reset_index(drop=True)

print("\nEvaluation sample:")
print(f"  FAKE : {len(fake_df)}")
print(f"  REAL : {len(real_df)}")
print(f"  TOTAL: {len(evaluation_df)}")


# ============================================================
# EVALUATE ONE ARTICLE
# ============================================================

def evaluate_article(row, index, total):

    article_id = int(row["article_id"])
    true_label = int(row["label"])
    true_verdict = label_to_verdict(true_label)

    title = str(row["title"])
    content = str(row["content"])
    url = str(row["url"])

    print("\n")
    print("=" * 90)
    print(f"TEST ARTICLE {index}/{total}")
    print("=" * 90)

    print(f"Article ID : {article_id}")
    print(f"True Label : {true_verdict}")
    print(f"Title      : {title[:150]}")

    start_time = time.perf_counter()

    # ========================================================
    # COMPLETE PRODUCTION PIPELINE
    # ========================================================

    print("\nRunning complete verification pipeline...")
    print("  1. Claim extraction")
    print("  2. Hybrid evidence retrieval")
    print("  3. Evidence reranking")
    print("  4. LLM evidence-grounded analysis")

    result = verify_article(
        article_id=article_id,
        title=title,
        content=content,
        url=url,
    )

    elapsed = time.perf_counter() - start_time

    # ========================================================
    # EXTRACT PIPELINE RESULTS
    # ========================================================

    claim_data = result.get("claim_data", {})
    claim = result.get("claim", "")

    evidence = result.get("evidence", [])

    analysis = result.get("analysis", {})

    predicted_verdict = str(
        analysis.get("verdict", "ERROR")
    ).upper()

    confidence = float(
        analysis.get("confidence", 0.0)
    )

    correct = predicted_verdict == true_verdict

    # ========================================================
    # PRINT CLAIM
    # ========================================================

    print("\nCLAIM")
    print("-" * 50)
    print(claim[:500])

    print("\nClaim type:")
    print(claim_data.get("claim_type"))

    print("\nEntities:")
    print(
        ", ".join(
            str(x)
            for x in claim_data.get("entities", [])
        )
    )

    print("\nNumbers / Dates:")
    print(
        ", ".join(
            str(x)
            for x in claim_data.get("numbers_dates", [])
        )
    )

    # ========================================================
    # PRINT EVIDENCE
    # ========================================================

    print("\nSELECTED EVIDENCE")
    print("-" * 50)

    for position, item in enumerate(evidence, start=1):

        print(
            f"\nEvidence {position}"
        )

        print(
            f"  Evidence ID       : "
            f"{item.get('evidence_id')}"
        )

        print(
            f"  Article ID        : "
            f"{item.get('article_id')}"
        )

        print(
            f"  Label             : "
            f"{item.get('label_name')}"
        )

        print(
            f"  Evidence type     : "
            f"{item.get('evidence_type')}"
        )

        print(
            f"  Semantic score    : "
            f"{float(item.get('semantic_score', 0)):.4f}"
        )

        print(
            f"  Lexical score     : "
            f"{float(item.get('lexical_score', 0)):.4f}"
        )

        print(
            f"  Entity match      : "
            f"{float(item.get('entity_match', 0)):.4f}"
        )

        print(
            f"  Number/date match : "
            f"{float(item.get('number_date_match', 0)):.4f}"
        )

        print(
            f"  Claim match       : "
            f"{float(item.get('claim_match', 0)):.4f}"
        )

        print(
            f"  Evidence score    : "
            f"{float(item.get('evidence_score', 0)):.4f}"
        )

        print(
            f"  Strong evidence   : "
            f"{item.get('strong_evidence')}"
        )

        print(
            f"  Title             : "
            f"{str(item.get('title', ''))[:180]}"
        )

    # ========================================================
    # PRINT LLM ANALYSIS
    # ========================================================

    print("\nLLM ANALYSIS")
    print("-" * 50)

    print(
        f"True verdict      : {true_verdict}"
    )

    print(
        f"Predicted verdict : {predicted_verdict}"
    )

    print(
        f"Confidence        : {confidence:.4f}"
    )

    print(
        f"Correct           : {correct}"
    )

    print(
        f"\nExplanation:\n"
        f"{analysis.get('explanation', '')}"
    )

    print(
        f"\nSummary:\n"
        f"{analysis.get('summary', '')}"
    )

    print(
        f"\nProcessing time   : {elapsed:.2f} seconds"
    )

    # ========================================================
    # RETURN RESULT
    # ========================================================

    top_evidence = (
        evidence[0]
        if evidence
        else {}
    )

    return {
        "article_id": article_id,
        "true_label": true_label,
        "true_verdict": true_verdict,

        "predicted_verdict": predicted_verdict,
        "predicted_binary": verdict_to_binary(
            predicted_verdict
        ),

        "confidence": confidence,
        "correct": correct,

        "claim": claim,
        "claim_type": claim_data.get(
            "claim_type"
        ),

        "is_fact_check_article": claim_data.get(
            "is_fact_check_article"
        ),

        "claim_status": claim_data.get(
            "claim_status"
        ),

        "entities": json.dumps(
            claim_data.get("entities", []),
            ensure_ascii=False,
        ),

        "numbers_dates": json.dumps(
            claim_data.get("numbers_dates", []),
            ensure_ascii=False,
        ),

        "retrieved_count": len(
            evidence
        ),

        "reranked_count": len(
            evidence
        ),

        "top_evidence_article_id": top_evidence.get(
            "article_id"
        ),

        "top_evidence_id": top_evidence.get(
            "evidence_id"
        ),

        "top_evidence_type": top_evidence.get(
            "evidence_type"
        ),

        "top_evidence_score": top_evidence.get(
            "evidence_score"
        ),

        "explanation": analysis.get(
            "explanation",
            "",
        ),

        "summary": analysis.get(
            "summary",
            "",
        ),

        "processing_time_seconds": round(
            elapsed,
            3,
        ),

        "error": None,
    }


# ============================================================
# RUN EVALUATION
# ============================================================

results = []

total_articles = len(evaluation_df)

for index, row in evaluation_df.iterrows():

    try:

        result = evaluate_article(
            row=row,
            index=index + 1,
            total=total_articles,
        )

        results.append(result)

    except Exception as exc:

        print("\nERROR PROCESSING ARTICLE")
        print("-" * 50)

        print(
            f"Article ID: {row['article_id']}"
        )

        print(
            f"Error: {exc}"
        )

        results.append(
            {
                "article_id": int(
                    row["article_id"]
                ),

                "true_label": int(
                    row["label"]
                ),

                "true_verdict": label_to_verdict(
                    row["label"]
                ),

                "predicted_verdict": "ERROR",
                "predicted_binary": None,

                "confidence": None,
                "correct": False,

                "claim": None,
                "claim_type": None,
                "is_fact_check_article": None,
                "claim_status": None,

                "entities": None,
                "numbers_dates": None,

                "retrieved_count": None,
                "reranked_count": None,

                "top_evidence_article_id": None,
                "top_evidence_id": None,
                "top_evidence_type": None,
                "top_evidence_score": None,

                "explanation": None,
                "summary": None,

                "processing_time_seconds": None,

                "error": str(exc),
            }
        )


# ============================================================
# SAVE ARTICLE-LEVEL RESULTS
# ============================================================

results_df = pd.DataFrame(results)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

results_csv = (
    RESULTS_DIR
    / "evaluation_results.csv"
)

results_df.to_csv(
    results_csv,
    index=False,
    encoding="utf-8-sig",
)

print("\nEvaluation results saved:")
print(results_csv)


# ============================================================
# METRIC CALCULATION
# ============================================================

print("\n")
print("=" * 90)
print("EVALUATION METRICS")
print("=" * 90)


total_count = len(results_df)

uncertain_count = int(
    (
        results_df["predicted_verdict"]
        == "UNCERTAIN"
    ).sum()
)

error_count = int(
    (
        results_df["predicted_verdict"]
        == "ERROR"
    ).sum()
)

valid_df = results_df[
    results_df["predicted_verdict"].isin(
        ["FAKE", "REAL"]
    )
].copy()

valid_count = len(valid_df)

correct_count = int(
    valid_df["correct"].sum()
)

print(
    f"\nTotal test articles : {total_count}"
)

print(
    f"Valid predictions   : {valid_count}"
)

print(
    f"UNCERTAIN           : {uncertain_count}"
)

print(
    f"ERROR               : {error_count}"
)

print(
    f"Correct predictions : {correct_count}"
)


# ============================================================
# BINARY CLASSIFICATION METRICS
# ============================================================

if valid_count > 0:

    y_true = valid_df[
        "true_label"
    ].astype(int)

    y_pred = valid_df[
        "predicted_verdict"
    ].map(
        {
            "FAKE": 0,
            "REAL": 1,
        }
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    fake_precision = precision_score(
        y_true,
        y_pred,
        pos_label=0,
        zero_division=0,
    )

    fake_recall = recall_score(
        y_true,
        y_pred,
        pos_label=0,
        zero_division=0,
    )

    fake_f1 = f1_score(
        y_true,
        y_pred,
        pos_label=0,
        zero_division=0,
    )

    real_precision = precision_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    real_recall = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    real_f1 = f1_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[
            "FAKE",
            "REAL",
        ],
        output_dict=True,
        zero_division=0,
    )

else:

    accuracy = 0.0

    fake_precision = 0.0
    fake_recall = 0.0
    fake_f1 = 0.0

    real_precision = 0.0
    real_recall = 0.0
    real_f1 = 0.0

    macro_f1 = 0.0
    weighted_f1 = 0.0

    cm = [[0, 0], [0, 0]]

    report = {}


# ============================================================
# PRINT CLASSIFICATION METRICS
# ============================================================

print("\nBinary Classification Metrics")
print("-" * 50)

print(
    f"Accuracy           : {accuracy:.4f}"
)

print(
    f"Fake Precision     : {fake_precision:.4f}"
)

print(
    f"Fake Recall        : {fake_recall:.4f}"
)

print(
    f"Fake F1 Score      : {fake_f1:.4f}"
)

print(
    f"Real Precision     : {real_precision:.4f}"
)

print(
    f"Real Recall        : {real_recall:.4f}"
)

print(
    f"Real F1 Score      : {real_f1:.4f}"
)

print(
    f"Macro F1           : {macro_f1:.4f}"
)

print(
    f"Weighted F1        : {weighted_f1:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix")
print("-" * 50)

print(
    "                 Predicted"
)

print(
    "                 FAKE    REAL"
)

print(
    f"Actual FAKE      "
    f"{cm[0][0]:<7} "
    f"{cm[0][1]}"
)

print(
    f"Actual REAL      "
    f"{cm[1][0]:<7} "
    f"{cm[1][1]}"
)


confusion_df = pd.DataFrame(
    cm,
    index=[
        "Actual_FAKE",
        "Actual_REAL",
    ],
    columns=[
        "Predicted_FAKE",
        "Predicted_REAL",
    ],
)

confusion_df.to_csv(
    RESULTS_DIR / "confusion_matrix.csv",
    encoding="utf-8-sig",
)


# ============================================================
# UNCERTAIN / ERROR / CONFIDENCE METRICS
# ============================================================

uncertain_rate = (
    uncertain_count / total_count
    if total_count > 0
    else 0.0
)

error_rate = (
    error_count / total_count
    if total_count > 0
    else 0.0
)

avg_confidence = (
    results_df["confidence"]
    .dropna()
    .mean()
    if "confidence" in results_df
    else 0.0
)

avg_processing_time = (
    results_df[
        "processing_time_seconds"
    ]
    .dropna()
    .mean()
    if "processing_time_seconds" in results_df
    else 0.0
)

correct_rate_all = (
    correct_count / total_count
    if total_count > 0
    else 0.0
)

coverage_rate = (
    valid_count / total_count
    if total_count > 0
    else 0.0
)

conditional_accuracy = (
    correct_count / valid_count
    if valid_count > 0
    else 0.0
)


print("\nAdditional Metrics")
print("-" * 50)

print(
    f"UNCERTAIN rate       : "
    f"{uncertain_rate:.4f}"
)

print(
    f"ERROR rate           : "
    f"{error_rate:.4f}"
)

print(
    f"Coverage rate        : "
    f"{coverage_rate:.4f}"
)

print(
    f"Accuracy all cases   : "
    f"{correct_rate_all:.4f}"
)

print(
    f"Accuracy valid cases : "
    f"{conditional_accuracy:.4f}"
)

print(
    f"Average confidence   : "
    f"{avg_confidence:.4f}"
)

print(
    f"Average processing   : "
    f"{avg_processing_time:.2f} sec/article"
)


# ============================================================
# SAVE METRICS JSON
# ============================================================

metrics = {
    "dataset": "HinFakeNews-V1",

    "dataset_label_mapping": {
        "0": "FAKE",
        "1": "REAL",
    },

    "total_dataset_rows": int(
        len(df)
    ),

    "evaluation_sample_size": int(
        total_count
    ),

    "sample_per_class": int(
        SAMPLE_PER_CLASS
    ),

    "fake_samples": int(
        (
            evaluation_df["label"]
            == 0
        ).sum()
    ),

    "real_samples": int(
        (
            evaluation_df["label"]
            == 1
        ).sum()
    ),

    "valid_predictions": int(
        valid_count
    ),

    "uncertain_count": int(
        uncertain_count
    ),

    "error_count": int(
        error_count
    ),

    "correct_predictions": int(
        correct_count
    ),

    "accuracy_all_cases": round(
        float(correct_rate_all),
        4,
    ),

    "accuracy_valid_cases": round(
        float(conditional_accuracy),
        4,
    ),

    "coverage_rate": round(
        float(coverage_rate),
        4,
    ),

    "fake_precision": round(
        float(fake_precision),
        4,
    ),

    "fake_recall": round(
        float(fake_recall),
        4,
    ),

    "fake_f1": round(
        float(fake_f1),
        4,
    ),

    "real_precision": round(
        float(real_precision),
        4,
    ),

    "real_recall": round(
        float(real_recall),
        4,
    ),

    "real_f1": round(
        float(real_f1),
        4,
    ),

    "macro_f1": round(
        float(macro_f1),
        4,
    ),

    "weighted_f1": round(
        float(weighted_f1),
        4,
    ),

    "uncertain_rate": round(
        float(uncertain_rate),
        4,
    ),

    "error_rate": round(
        float(error_rate),
        4,
    ),

    "average_confidence": round(
        float(avg_confidence),
        4,
    ),

    "average_processing_time_seconds": round(
        float(avg_processing_time),
        3,
    ),

    "classification_report": report,
}


metrics_path = (
    RESULTS_DIR / "metrics.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metrics,
        f,
        indent=4,
        ensure_ascii=False,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 90)
print("EVALUATION COMPLETE")
print("=" * 90)

print(
    f"\nAccuracy (all cases) : "
    f"{correct_rate_all:.2%}"
)

print(
    f"Accuracy (valid)     : "
    f"{conditional_accuracy:.2%}"
)

print(
    f"Fake F1              : "
    f"{fake_f1:.2%}"
)

print(
    f"Real F1              : "
    f"{real_f1:.2%}"
)

print(
    f"Macro F1             : "
    f"{macro_f1:.2%}"
)

print(
    f"Coverage             : "
    f"{coverage_rate:.2%}"
)

print(
    f"UNCERTAIN            : "
    f"{uncertain_rate:.2%}"
)

print(
    f"Average time         : "
    f"{avg_processing_time:.2f} sec/article"
)

print("\nFiles generated:")

print(
    f"  {results_csv}"
)

print(
    f"  {metrics_path}"
)

print(
    f"  {RESULTS_DIR / 'confusion_matrix.csv'}"
)

print("\n" + "=" * 90)