from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\fake-news-bda")

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "news_cleaned.csv"
)

RAG_DIR = PROJECT_ROOT / "results" / "rag"

RAG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

BATCH_SIZE = 32

# Number of characters from content used for the embedding.
CONTENT_LIMIT = 1500


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("BUILDING SEMANTIC RAG INDEX")
print("=" * 70)

print("\n[1] Loading dataset...")
print(f"Dataset: {DATA_PATH}")

df = pd.read_csv(
    DATA_PATH,
    encoding="utf-8-sig"
)

print(f"Rows loaded: {len(df):,}")


# ============================================================
# VALIDATION
# ============================================================

EXPECTED_ROWS = 67587

if len(df) != EXPECTED_ROWS:
    raise ValueError(
        f"Dataset validation failed. "
        f"Expected {EXPECTED_ROWS} rows "
        f"but found {len(df)}."
    )

required_columns = [
    "article_id",
    "url",
    "title",
    "content",
    "label"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("Dataset validation: PASS")


# ============================================================
# PREPARE DOCUMENT TEXT
# ============================================================

print("\n[2] Preparing documents for embedding...")

df["title"] = (
    df["title"]
    .fillna("")
    .astype(str)
)

df["content"] = (
    df["content"]
    .fillna("")
    .astype(str)
)

# Title is repeated to give the headline stronger influence.
df["embedding_text"] = (
    df["title"]
    + " "
    + df["title"]
    + " "
    + df["content"].str[:CONTENT_LIMIT]
)

df["embedding_text"] = (
    df["embedding_text"]
    .str.replace(
        r"\s+",
        " ",
        regex=True
    )
    .str.strip()
)

print("Document preparation completed.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\n[3] Loading multilingual embedding model...")

print(f"Model: {MODEL_NAME}")

model = SentenceTransformer(
    MODEL_NAME
)

print("Embedding model loaded.")


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("\n[4] Generating document embeddings...")
print(f"Batch size: {BATCH_SIZE}")
print(f"Documents: {len(df):,}")

embeddings = model.encode(
    df["embedding_text"].tolist(),
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True
)

print("\nEmbedding generation completed.")

print(
    f"Embedding matrix shape: {embeddings.shape}"
)

print(
    f"Embedding dtype: {embeddings.dtype}"
)


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

print("\n[5] Saving semantic index...")

embedding_path = (
    RAG_DIR
    / "semantic_embeddings.npy"
)

np.save(
    embedding_path,
    embeddings.astype(np.float32)
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = df[
    [
        "article_id",
        "url",
        "title",
        "content",
        "label"
    ]
].copy()

metadata_path = (
    RAG_DIR
    / "semantic_metadata.pkl"
)

metadata.to_pickle(
    metadata_path
)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

config = {
    "model_name": MODEL_NAME,
    "embedding_dimension": int(embeddings.shape[1]),
    "document_count": int(len(df)),
    "batch_size": BATCH_SIZE,
    "content_limit": CONTENT_LIMIT,
    "normalized": True
}

config_path = (
    RAG_DIR
    / "semantic_config.pkl"
)

pd.to_pickle(
    config,
    config_path
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n[6] Validating semantic index...")

loaded_embeddings = np.load(
    embedding_path,
    mmap_mode="r"
)

if loaded_embeddings.shape[0] != EXPECTED_ROWS:
    raise ValueError(
        "Embedding count does not match dataset."
    )

if loaded_embeddings.shape[1] != 384:
    raise ValueError(
        f"Expected 384-dimensional embeddings, "
        f"found {loaded_embeddings.shape[1]}."
    )

if len(metadata) != EXPECTED_ROWS:
    raise ValueError(
        "Metadata count does not match dataset."
    )

print("Embedding validation: PASS")
print("Metadata validation: PASS")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SEMANTIC RAG INDEX CREATED SUCCESSFULLY")
print("=" * 70)

print(
    f"Articles indexed       : {len(df):,}"
)

print(
    f"Embedding dimensions   : {embeddings.shape[1]}"
)

print(
    f"Embedding model        : {MODEL_NAME}"
)

print(
    f"Content characters     : {CONTENT_LIMIT}"
)

print("\nFiles created:")

for filename in [
    "semantic_embeddings.npy",
    "semantic_metadata.pkl",
    "semantic_config.pkl"
]:

    path = RAG_DIR / filename

    size_mb = path.stat().st_size / (
        1024 * 1024
    )

    print(
        f"  - {filename} "
        f"({size_mb:.2f} MB)"
    )

print("\n" + "=" * 70)
print("SEMANTIC INDEX BUILD COMPLETED")
print("=" * 70)