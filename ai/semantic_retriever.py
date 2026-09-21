# ai/semantic_retriever.py

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\fake-news-bda")
RAG_DIR = PROJECT_ROOT / "results" / "rag"


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_TOP_K = 5


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading multilingual embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded.")


# ============================================================
# LOAD SEMANTIC INDEX
# ============================================================

print("\nLoading semantic index...")

embeddings = np.load(
    RAG_DIR / "semantic_embeddings.npy",
    mmap_mode="r"
)

metadata = pd.read_pickle(
    RAG_DIR / "semantic_metadata.pkl"
)

print(
    f"Loaded embeddings: {embeddings.shape}"
)

print(
    f"Loaded metadata: {len(metadata):,} articles"
)


# ============================================================
# VALIDATION
# ============================================================

if embeddings.shape[0] != len(metadata):
    raise ValueError(
        "Embedding count and metadata count do not match."
    )

if embeddings.shape[1] != 384:
    raise ValueError(
        f"Expected 384-dimensional embeddings, "
        f"found {embeddings.shape[1]}."
    )

print("Semantic index validation: PASS")


# ============================================================
# RETRIEVAL FUNCTION
# ============================================================

def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    exclude_article_id: int | None = None
):
    if not query or not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    query = query.strip()

    top_k = min(
        top_k,
        len(metadata)
    )

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )[0]

    scores = np.asarray(
        embeddings @ query_embedding
    ).reshape(-1)

    # Retrieve extra candidates so that the source article
    # can be excluded without reducing the requested count.
    search_k = min(
        top_k + 10,
        len(metadata)
    )

    top_indices = np.argpartition(
        scores,
        -search_k
    )[-search_k:]

    top_indices = top_indices[
        np.argsort(
            scores[top_indices]
        )[::-1]
    ]

    results = []

    for index in top_indices:

        row = metadata.iloc[index]

        article_id = int(
            row["article_id"]
        )

        # Prevent evaluation/source leakage.
        if (
            exclude_article_id is not None
            and article_id == int(exclude_article_id)
        ):
            continue

        label = int(
            row["label"]
        )

        if label == 0:
            label_name = "FAKE"
        elif label == 1:
            label_name = "REAL"
        else:
            label_name = "UNKNOWN"

        content = str(
            row["content"]
        )

        results.append({
            "rank": len(results) + 1,
            "article_id": article_id,
            "title": str(row["title"]),
            "content": content,
            "url": str(row["url"]),
            "label": label,
            "label_name": label_name,
            "relevance_score": round(
                float(scores[index]),
                4
            )
        })

        if len(results) >= top_k:
            break

    return results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def print_results(
    query: str,
    results
):
    """Display retrieval results."""

    print("\n" + "=" * 90)
    print("QUERY")
    print("=" * 90)

    print(query)

    print("\n" + "=" * 90)
    print("SEMANTICALLY RETRIEVED EVIDENCE")
    print("=" * 90)

    for result in results:

        print("\n" + "-" * 90)

        print(
            f"Rank       : "
            f"{result['rank']}"
        )

        print(
            f"Article ID : "
            f"{result['article_id']}"
        )

        print(
            f"Label      : "
            f"{result['label_name']}"
        )

        print(
            f"Similarity : "
            f"{result['relevance_score']}"
        )

        print(
            f"Title      : "
            f"{result['title']}"
        )

        print(
            f"URL        : "
            f"{result['url']}"
        )

        print(
            "Evidence   : "
            f"{result['content'][:700]}"
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 90)
    print("SEMANTIC RAG RETRIEVER TEST")
    print("=" * 90)

    # --------------------------------------------------------
    # English test
    # --------------------------------------------------------

    english_query = (
        "The government has announced a new rule "
        "for schools across India."
    )

    english_results = retrieve(
        english_query,
        top_k=5
    )

    print_results(
        english_query,
        english_results
    )

    # --------------------------------------------------------
    # Hindi test
    # --------------------------------------------------------

    hindi_query = (
        "भारत सरकार ने स्कूलों के लिए नया नियम लागू किया है।"
    )

    hindi_results = retrieve(
        hindi_query,
        top_k=5
    )

    print_results(
        hindi_query,
        hindi_results
    )

    print("\n" + "=" * 90)
    print("SEMANTIC RETRIEVAL TEST COMPLETED")
    print("=" * 90)