import pandas as pd
from pathlib import Path
import csv


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "HinFakeNews-V1.xlsx"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "news_cleaned.csv"


# ============================================================
# LOAD DATASET
# ============================================================

print("\n" + "=" * 60)
print("FAKE NEWS BDA - DATA PREPROCESSING")
print("=" * 60)

print("\n[1] Loading dataset...")

df = pd.read_excel(INPUT_FILE)

print(f"Original rows: {len(df)}")
print(f"Original columns: {list(df.columns)}")


# ============================================================
# REMOVE DUPLICATES
# ============================================================

print("\n[2] Removing duplicate articles...")

before = len(df)

df = df.drop_duplicates()

duplicates_removed = before - len(df)

print(f"Duplicates removed: {duplicates_removed}")
print(f"Rows after deduplication: {len(df)}")


# ============================================================
# CLEAN TEXT
# ============================================================

print("\n[3] Cleaning text columns...")

df["TITLE"] = df["TITLE"].fillna("").astype(str).str.strip()
df["CONTENT"] = df["CONTENT"].fillna("").astype(str).str.strip()
df["URL"] = df["URL"].fillna("").astype(str).str.strip()


# ============================================================
# RENAME COLUMNS
# ============================================================

df = df.rename(
    columns={
        "URL": "url",
        "TITLE": "title",
        "CONTENT": "content",
        "BOOL": "label"
    }
)


# ============================================================
# ENSURE LABEL IS INTEGER
# ============================================================

df["label"] = pd.to_numeric(df["label"], errors="coerce")

df = df.dropna(subset=["label"])

df["label"] = df["label"].astype(int)


# ============================================================
# ADD ARTICLE ID
# ============================================================

df.insert(
    0,
    "article_id",
    range(1, len(df) + 1)
)


# ============================================================
# FINAL COLUMN ORDER
# ============================================================

df = df[
    [
        "article_id",
        "url",
        "title",
        "content",
        "label"
    ]
]


# ============================================================
# SAVE PROPERLY QUOTED UTF-8 CSV
# ============================================================

print("\n[4] Saving processed dataset...")

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
    quoting=csv.QUOTE_ALL,
    quotechar='"',
    escapechar="\\",
    lineterminator="\n"
)


# ============================================================
# VALIDATION
# ============================================================

print("\n[5] Validation")
print("-" * 60)

print(f"Final articles: {len(df)}")

print("\nLabel distribution:")
print(df["label"].value_counts().sort_index())

print("\nMissing values:")
print(df.isnull().sum())

print("\nFinal columns:")
print(list(df.columns))

print("\nOutput file:")
print(OUTPUT_FILE)

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)