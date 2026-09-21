from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\fake-news-bda")

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "news_cleaned.csv"
RAG_DIR = PROJECT_ROOT / "results" / "rag"

RAG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 5

# We use word + character features.
# Character features are useful for Hindi and mixed
# Hindi-English text because they are less dependent
# on perfect word tokenization.
WORD_NGRAM_RANGE = (1, 2)
CHAR_NGRAM_RANGE = (3, 5)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("BUILDING RAG RETRIEVAL INDEX")
print("=" * 70)

print("\n[1] Loading dataset...")
print(f"Dataset: {DATA_PATH}")

df = pd.read_csv(
    DATA_PATH,
    encoding="utf-8-sig"
)

print(f"Rows loaded: {len(df)}")


# ============================================================
# VALIDATION
# ============================================================

EXPECTED_ROWS = 67587

if len(df) != EXPECTED_ROWS:
    raise ValueError(
        f"Dataset validation failed. "
        f"Expected {EXPECTED_ROWS} rows but found {len(df)}."
    )

required_columns = [
    "article_id",
    "url",
    "title",
    "content",
    "label"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("Dataset validation: PASS")


# ============================================================
# CLEAN TEXT
# ============================================================

print("\n[2] Preparing searchable text...")

df["title"] = df["title"].fillna("").astype(str)
df["content"] = df["content"].fillna("").astype(str)

# Title is repeated so that important headline terms
# receive stronger representation in retrieval.
df["search_text"] = (
    df["title"] + " " +
    df["title"] + " " +
    df["content"]
)

df["search_text"] = (
    df["search_text"]
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

print("Search text prepared.")


# ============================================================
# WORD TF-IDF
# ============================================================

print("\n[3] Building word-level TF-IDF index...")

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=WORD_NGRAM_RANGE,
    min_df=2,
    max_df=0.98,
    sublinear_tf=True,
    max_features=300000
)

word_matrix = word_vectorizer.fit_transform(
    df["search_text"]
)

print(f"Word matrix shape: {word_matrix.shape}")


# ============================================================
# CHARACTER TF-IDF
# ============================================================

print("\n[4] Building character-level TF-IDF index...")

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=CHAR_NGRAM_RANGE,
    min_df=3,
    max_features=200000,
    sublinear_tf=True
)

char_matrix = char_vectorizer.fit_transform(
    df["search_text"]
)

print(f"Character matrix shape: {char_matrix.shape}")


# ============================================================
# SAVE INDEX
# ============================================================

print("\n[5] Saving RAG index...")

dump(
    word_vectorizer,
    RAG_DIR / "word_vectorizer.joblib"
)

dump(
    char_vectorizer,
    RAG_DIR / "char_vectorizer.joblib"
)

dump(
    word_matrix,
    RAG_DIR / "word_matrix.joblib"
)

dump(
    char_matrix,
    RAG_DIR / "char_matrix.joblib"
)


# Save only the metadata required by the retriever.
metadata = df[
    [
        "article_id",
        "url",
        "title",
        "content",
        "label"
    ]
].copy()

metadata.to_pickle(
    RAG_DIR / "metadata.pkl"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RAG INDEX CREATED SUCCESSFULLY")
print("=" * 70)

print(f"Articles indexed : {len(df):,}")
print(f"Word features    : {word_matrix.shape[1]:,}")
print(f"Character features: {char_matrix.shape[1]:,}")
print(f"Top-K retrieval  : {TOP_K}")

print("\nFiles created:")

for file in sorted(RAG_DIR.iterdir()):
    print(f"  - {file.name}")

print("\n" + "=" * 70)
print("INDEX BUILD COMPLETED")
print("=" * 70)