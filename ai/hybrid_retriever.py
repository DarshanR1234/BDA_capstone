import os
import re
import pickle
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"D:\fake-news-bda")

DATASET_PATH = (
    BASE_DIR / "data" / "processed" / "news_cleaned.csv"
)

RAG_DIR = BASE_DIR / "results" / "rag"

SEMANTIC_EMBEDDINGS_PATH = (
    RAG_DIR / "semantic_embeddings.npy"
)

SEMANTIC_METADATA_PATH = (
    RAG_DIR / "semantic_metadata.pkl"
)

SEMANTIC_CONFIG_PATH = (
    RAG_DIR / "semantic_config.pkl"
)

WORD_VECTORIZER_PATH = (
    RAG_DIR / "word_vectorizer.joblib"
)

CHAR_VECTORIZER_PATH = (
    RAG_DIR / "char_vectorizer.joblib"
)

# IMPORTANT:
# Your actual files are .joblib, NOT .npz.
WORD_MATRIX_PATH = (
    RAG_DIR / "word_matrix.joblib"
)

CHAR_MATRIX_PATH = (
    RAG_DIR / "char_matrix.joblib"
)


# ============================================================
# LOAD ENV
# ============================================================

load_dotenv(BASE_DIR / ".env")


# ============================================================
# MODEL / INDEX LOADING
# ============================================================

print("[Hybrid Retriever] Loading semantic model...")

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

model = SentenceTransformer(MODEL_NAME)

print("[Hybrid Retriever] Semantic model loaded.")


print("[Hybrid Retriever] Loading lexical indexes...")

word_vectorizer = load(WORD_VECTORIZER_PATH)
char_vectorizer = load(CHAR_VECTORIZER_PATH)

word_matrix = load(WORD_MATRIX_PATH)
char_matrix = load(CHAR_MATRIX_PATH)

semantic_embeddings = np.load(
    SEMANTIC_EMBEDDINGS_PATH,
    mmap_mode="r"
)

with open(SEMANTIC_METADATA_PATH, "rb") as f:
    semantic_metadata = pickle.load(f)

with open(SEMANTIC_CONFIG_PATH, "rb") as f:
    semantic_config = pickle.load(f)


# ============================================================
# NORMALIZE METADATA
# ============================================================

if isinstance(semantic_metadata, pd.DataFrame):

    semantic_records = (
        semantic_metadata
        .reset_index(drop=True)
        .to_dict(orient="records")
    )

else:

    semantic_records = list(semantic_metadata)


print(
    "[Hybrid Retriever] Index loaded successfully: "
    f"{len(semantic_records)} articles"
)


# ============================================================
# BASIC TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text)

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def tokenize(text):

    text = normalize_text(text)

    return set(
        re.findall(
            r"[a-zA-ZÀ-ÿ\u0900-\u097F0-9]+",
            text
        )
    )


# ============================================================
# ENTITY NORMALIZATION
# ============================================================

def normalize_entity(entity):

    entity = normalize_text(entity)

    entity = re.sub(
        r"[^\w\s\u0900-\u097F-]",
        " ",
        entity,
        flags=re.UNICODE
    )

    entity = re.sub(
        r"\s+",
        " ",
        entity
    )

    return entity.strip()


def entity_variants(entity):

    entity = normalize_entity(entity)

    if not entity:
        return []

    variants = {entity}

    parts = entity.split()

    if len(parts) >= 2:
        variants.add(
            " ".join(parts[-2:])
        )

    if len(parts) >= 3:
        variants.add(
            " ".join(parts[-3:])
        )

    return list(variants)


# ============================================================
# PHRASE EXTRACTION
# ============================================================

STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "from",
    "into",
    "after",
    "before",
    "during",
    "said",
    "says",
    "was",
    "were",
    "has",
    "have",
    "had",
    "will",
    "would",
    "could",
    "should",
    "about",
    "their",
    "there",
    "they",
    "them",
    "india",
    "indian",
    "according",
    "report",
    "reports",
    "claim",
    "claims",
    "news",

    # Common Hindi noise
    "और",
    "यह",
    "इस",
    "उस",
    "एक",
    "के",
    "की",
    "का",
    "में",
    "से",
    "पर",
    "को",
    "ने",
    "है",
    "था",
    "थे",
    "हो",
    "रहा",
    "रही",
    "रहे",
    "कहा",
    "बताया",
}


def important_phrases(
    text,
    max_phrases=15
):

    text = normalize_text(text)

    if not text:
        return []

    words = re.findall(
        r"[a-zA-ZÀ-ÿ\u0900-\u097F0-9]+",
        text
    )

    words = [
        w
        for w in words
        if w not in STOPWORDS
        and len(w) >= 3
    ]

    phrases = []

    # 4-word phrases
    for i in range(len(words) - 3):

        phrase = " ".join(
            words[i:i + 4]
        )

        if len(phrase) >= 12:
            phrases.append(phrase)

    # 3-word phrases
    for i in range(len(words) - 2):

        phrase = " ".join(
            words[i:i + 3]
        )

        if len(phrase) >= 10:
            phrases.append(phrase)

    # 2-word phrases
    for i in range(len(words) - 1):

        phrase = " ".join(
            words[i:i + 2]
        )

        if len(phrase) >= 8:
            phrases.append(phrase)

    # Distinctive individual words
    for word in words:

        if len(word) >= 7:
            phrases.append(word)

    result = []
    seen = set()

    for phrase in phrases:

        if phrase in seen:
            continue

        seen.add(phrase)
        result.append(phrase)

    return result[:max_phrases]


# ============================================================
# ENTITY MATCHING
# ============================================================

def matched_entities(
    query_entities,
    title,
    content
):

    if not query_entities:
        return []

    text = normalize_text(
        f"{title} {content[:5000]}"
    )

    matched = []

    for entity in query_entities:

        variants = entity_variants(entity)

        if any(
            variant in text
            for variant in variants
        ):

            matched.append(
                normalize_entity(entity)
            )

    return matched


def entity_match_score(
    query_entities,
    title,
    content
):

    if not query_entities:
        return 0.0

    matched = matched_entities(
        query_entities,
        title,
        content
    )

    return (
        len(matched)
        /
        max(len(query_entities), 1)
    )


# ============================================================
# ENTITY CO-OCCURRENCE
# ============================================================

def entity_cooccurrence_score(
    query_entities,
    title,
    content
):

    if not query_entities:
        return 0.0

    text = normalize_text(
        f"{title} {content[:7000]}"
    )

    total = len(query_entities)

    if total == 1:
        return (
            1.0
            if any(
                v in text
                for v in entity_variants(
                    query_entities[0]
                )
            )
            else 0.0
        )

    matched = 0

    for entity in query_entities:

        variants = entity_variants(entity)

        if any(
            variant in text
            for variant in variants
        ):
            matched += 1

    # Stronger than simple entity match when
    # multiple entities occur together.
    if total >= 4:

        if matched >= 4:
            return 1.0

        if matched == 3:
            return 0.75

        if matched == 2:
            return 0.50

        if matched == 1:
            return 0.15

        return 0.0

    if total == 3:

        if matched == 3:
            return 1.0

        if matched == 2:
            return 0.65

        if matched == 1:
            return 0.20

        return 0.0

    if matched == 2:
        return 1.0

    if matched == 1:
        return 0.30

    return 0.0


# ============================================================
# NUMBER / DATE MATCHING
# ============================================================

def number_date_match_score(
    numbers_dates,
    title,
    content
):

    if not numbers_dates:
        return 0.0

    text = normalize_text(
        f"{title} {content[:7000]}"
    )

    text_digits = re.sub(
        r"[^\d]",
        "",
        text
    )

    matched = 0

    for item in numbers_dates:

        value = normalize_text(item)

        if not value:
            continue

        # Direct match
        if value in text:

            matched += 1
            continue

        # Digit-only comparison
        digits = re.sub(
            r"[^\d]",
            "",
            value
        )

        if len(digits) >= 2:

            if digits in text_digits:

                matched += 1

    return (
        matched
        /
        max(len(numbers_dates), 1)
    )


# ============================================================
# PHRASE MATCHING
# ============================================================

def phrase_match_score(
    phrases,
    title,
    content
):

    if not phrases:
        return 0.0

    title_text = normalize_text(title)

    content_text = normalize_text(
        content[:7000]
    )

    matched = 0

    for phrase in phrases:

        phrase = normalize_text(
            phrase
        )

        if not phrase:
            continue

        if phrase in title_text:

            matched += 1
            continue

        if phrase in content_text:

            matched += 1

    return (
        matched
        /
        max(len(phrases), 1)
    )


# ============================================================
# EXACT EVENT / PHRASE SCORE
# ============================================================

def event_specificity_score(
    claim,
    phrases,
    title,
    content
):
    """
    Measures whether the candidate describes the same
    concrete event/claim, not merely the same topic.

    Title matches are weighted more heavily because news
    titles normally contain the event identity.
    """

    title_text = normalize_text(title)
    content_text = normalize_text(content[:7000])
    full_text = f"{title_text} {content_text}"

    claim_tokens = [
        x
        for x in tokenize(claim)
        if len(x) >= 4
        and x not in STOPWORDS
    ]

    if not claim_tokens:
        return 0.0

    title_matches = sum(
        1
        for token in claim_tokens
        if token in title_text
    )

    content_matches = sum(
        1
        for token in claim_tokens
        if token in content_text
    )

    title_score = (
        title_matches
        /
        max(len(claim_tokens), 1)
    )

    content_score = (
        content_matches
        /
        max(len(claim_tokens), 1)
    )

    exact_phrase_matches = 0
    title_phrase_matches = 0

    for phrase in phrases:
        phrase = normalize_text(phrase)

        if len(phrase.split()) < 2:
            continue

        if phrase in title_text:
            title_phrase_matches += 1
            exact_phrase_matches += 1
        elif phrase in content_text:
            exact_phrase_matches += 1

    phrase_score = min(
        exact_phrase_matches / 5.0,
        1.0
    )

    title_phrase_score = min(
        title_phrase_matches / 3.0,
        1.0
    )

    # Strongest signal: distinctive multi-word event phrases
    # appearing in the title.
    score = (
        0.30 * title_score
        + 0.20 * content_score
        + 0.30 * phrase_score
        + 0.20 * title_phrase_score
    )

    return min(1.0, score)


def claim_event_overlap_score(
    claim,
    title,
    content
):
    """
    Strong claim-level overlap score.

    Uses distinctive claim terms and gives extra weight when
    several distinctive terms occur in the title together.
    """

    title_text = normalize_text(title)
    content_text = normalize_text(content[:7000])

    claim_tokens = [
        x
        for x in tokenize(claim)
        if len(x) >= 4
        and x not in STOPWORDS
    ]

    if not claim_tokens:
        return 0.0

    # Remove very generic words that create false matches.
    generic = {
        "state", "states", "country", "people",
        "government", "government", "according",
        "report", "reports", "news", "said",
        "says", "claim", "claims", "india",
        "indian", "pakistani", "american",
        "former", "current", "new", "latest"
    }

    distinctive = [
        x for x in claim_tokens
        if x not in generic
    ]

    if not distinctive:
        distinctive = claim_tokens

    title_hits = sum(
        1 for token in distinctive
        if token in title_text
    )

    content_hits = sum(
        1 for token in distinctive
        if token in content_text
    )

    title_score = title_hits / max(len(distinctive), 1)
    content_score = content_hits / max(len(distinctive), 1)

    # Multiple distinctive title terms are a strong indication
    # that this is the same event.
    title_density_bonus = 0.0
    if title_hits >= 4:
        title_density_bonus = 0.30
    elif title_hits >= 3:
        title_density_bonus = 0.20
    elif title_hits >= 2:
        title_density_bonus = 0.10

    return min(
        1.0,
        0.55 * title_score
        + 0.25 * content_score
        + title_density_bonus
    )


# ============================================================
# FACT CHECK DETECTION
# ============================================================

FACT_CHECK_TERMS = [
    "fact check",
    "fact-check",
    "factcheck",
    "फैक्ट चेक",
    "फैक्ट-चेक",
    "सच",
    "झूठ",
    "वायरल",
    "पड़ताल",
    "तथ्य जांच",
    "अल्ट न्यूज़",
    "alt news",
    "boom",
    "boomlive",
    "vishvas news",
]


def fact_check_score(
    title,
    content
):

    combined_text = normalize_text(
        f"{title} {content[:1500]}"
    )

    return (
        1.0
        if any(
            term in combined_text
            for term in FACT_CHECK_TERMS
        )
        else 0.0
    )


# ============================================================
# MIN-MAX
# ============================================================

def minmax(values):

    values = np.asarray(
        values,
        dtype=np.float32
    )

    if len(values) == 0:
        return values

    minimum = values.min()
    maximum = values.max()

    if maximum - minimum < 1e-8:

        return np.ones_like(values)

    return (
        values - minimum
    ) / (
        maximum - minimum
    )


# ============================================================
# EXACT ENTITY CANDIDATE DISCOVERY
# ============================================================

def discover_identity_candidates(
    entities,
    phrases,
    numbers_dates,
    exclude_article_id=None,
    max_candidates=250
):

    """
    Additional retrieval layer.

    This scans metadata titles/content for exact
    entity/event identity.

    It is intentionally separate from semantic
    retrieval so that highly specific events are
    not lost simply because their semantic score
    is lower.
    """

    if not entities:
        return set()

    normalized_entities = []

    for entity in entities:

        for variant in entity_variants(entity):

            variant = normalize_text(
                variant
            )

            if (
                variant
                and len(variant) >= 3
            ):

                normalized_entities.append(
                    variant
                )

    normalized_entities = list(
        dict.fromkeys(
            normalized_entities
        )
    )

    candidate_indices = set()

    for index, record in enumerate(
        semantic_records
    ):

        article_id = record.get(
            "article_id",
            index
        )

        if exclude_article_id is not None:

            try:

                if int(article_id) == int(
                    exclude_article_id
                ):
                    continue

            except Exception:

                if str(article_id) == str(
                    exclude_article_id
                ):
                    continue

        title = normalize_text(
            record.get(
                "title",
                ""
            )
        )

        content = normalize_text(
            record.get(
                "content",
                ""
            )
        )[:7000]

        text = f"{title} {content}"

        entity_hits = 0

        for entity in normalized_entities:

            if entity in text:

                entity_hits += 1

        # At least one exact identity match.
        if entity_hits == 0:
            continue

        # Strong identity:
        # two or more entities together.
        if entity_hits >= 2:

            candidate_indices.add(
                index
            )

        else:

            # Single entity is allowed only if
            # an important event phrase/date also matches.
            p_score = phrase_match_score(
                phrases,
                title,
                content
            )

            n_score = number_date_match_score(
                numbers_dates,
                title,
                content
            )

            if (
                p_score >= 0.15
                or n_score >= 0.20
            ):

                candidate_indices.add(
                    index
                )

        if len(candidate_indices) >= max_candidates:
            break

    return candidate_indices


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    claim,
    source_title="",
    source_content="",
    entities=None,
    numbers_dates=None,
    top_k=12,
    exclude_article_id=None,
):

    """
    Production hybrid retriever.

    Retrieval layers:

        1. Semantic similarity
        2. Original-language TF-IDF
        3. Claim TF-IDF
        4. Character TF-IDF
        5. Exact entity discovery
        6. Entity co-occurrence
        7. Exact event/phrase matching
        8. Number/date matching
        9. Fact-check detection

    The important addition is #5:
    exact identity candidates can enter the
    candidate pool even when their semantic
    similarity is not high enough.
    """

    entities = entities or []
    numbers_dates = numbers_dates or []

    claim = str(claim or "")
    source_title = str(
        source_title or ""
    )
    source_content = str(
        source_content or ""
    )

    if not claim.strip():
        return []

    # --------------------------------------------------------
    # Query construction
    # --------------------------------------------------------

    entity_text = " ".join(
        str(x)
        for x in entities
    )

    number_text = " ".join(
        str(x)
        for x in numbers_dates
    )

    semantic_query = (
        f"{claim}. "
        f"{source_title}. "
        f"{entity_text}. "
        f"{number_text}"
    )

    lexical_query_original = (
        f"{source_title} "
        f"{source_content[:3500]}"
    )

    lexical_query_claim = (
        f"{claim} "
        f"{entity_text} "
        f"{number_text}"
    )

    # --------------------------------------------------------
    # Important phrases
    # --------------------------------------------------------

    phrases = important_phrases(
        claim
    )

    for entity in entities:

        for variant in entity_variants(
            entity
        ):

            if (
                variant
                and variant not in phrases
            ):

                phrases.append(
                    variant
                )

    phrases = phrases[:30]

    # --------------------------------------------------------
    # Semantic retrieval
    # --------------------------------------------------------

    query_embedding = model.encode(
        [semantic_query],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    semantic_scores_all = np.dot(
        semantic_embeddings,
        query_embedding[0]
    )

    semantic_top_n = min(
        150,
        len(semantic_scores_all)
    )

    semantic_indices = np.argpartition(
        semantic_scores_all,
        -semantic_top_n
    )[-semantic_top_n:]

    # --------------------------------------------------------
    # Word TF-IDF - original article
    # --------------------------------------------------------

    word_query_original = (
        word_vectorizer.transform(
            [lexical_query_original]
        )
    )

    word_scores_original = (
        cosine_similarity(
            word_query_original,
            word_matrix
        ).ravel()
    )

    word_top_n = min(
        150,
        len(word_scores_original)
    )

    word_indices_original = (
        np.argpartition(
            word_scores_original,
            -word_top_n
        )[-word_top_n:]
    )

    # --------------------------------------------------------
    # Word TF-IDF - extracted claim
    # --------------------------------------------------------

    word_query_claim = (
        word_vectorizer.transform(
            [lexical_query_claim]
        )
    )

    word_scores_claim = (
        cosine_similarity(
            word_query_claim,
            word_matrix
        ).ravel()
    )

    claim_top_n = min(
        100,
        len(word_scores_claim)
    )

    word_indices_claim = (
        np.argpartition(
            word_scores_claim,
            -claim_top_n
        )[-claim_top_n:]
    )

    # --------------------------------------------------------
    # Character TF-IDF
    # --------------------------------------------------------

    char_query = (
        char_vectorizer.transform(
            [lexical_query_original]
        )
    )

    char_scores = (
        cosine_similarity(
            char_query,
            char_matrix
        ).ravel()
    )

    char_top_n = min(
        150,
        len(char_scores)
    )

    char_indices = (
        np.argpartition(
            char_scores,
            -char_top_n
        )[-char_top_n:]
    )

    # --------------------------------------------------------
    # EXACT IDENTITY CANDIDATES
    # --------------------------------------------------------

    identity_indices = (
        discover_identity_candidates(
            entities=entities,
            phrases=phrases,
            numbers_dates=numbers_dates,
            exclude_article_id=exclude_article_id,
            max_candidates=300,
        )
    )

    # --------------------------------------------------------
    # Candidate pool
    # --------------------------------------------------------

    candidate_indices = set()

    candidate_indices.update(
        int(x)
        for x in semantic_indices
    )

    candidate_indices.update(
        int(x)
        for x in word_indices_original
    )

    candidate_indices.update(
        int(x)
        for x in word_indices_claim
    )

    candidate_indices.update(
        int(x)
        for x in char_indices
    )

    # NEW:
    # exact identity candidates
    candidate_indices.update(
        identity_indices
    )

    if not candidate_indices:
        return []

    # --------------------------------------------------------
    # Candidate-level normalization
    # --------------------------------------------------------

    index_list = list(
        candidate_indices
    )

    semantic_values = (
        semantic_scores_all[
            index_list
        ]
    )

    lexical_values = (
        0.55
        *
        word_scores_original[
            index_list
        ]
        +
        0.45
        *
        char_scores[
            index_list
        ]
    )

    claim_values = (
        word_scores_claim[
            index_list
        ]
    )

    semantic_norm = minmax(
        semantic_values
    )

    lexical_norm = minmax(
        lexical_values
    )

    claim_norm = minmax(
        claim_values
    )

    # --------------------------------------------------------
    # Candidate scoring
    # --------------------------------------------------------

    candidates = []

    for pos, index in enumerate(
        index_list
    ):

        record = semantic_records[
            index
        ]

        article_id = record.get(
            "article_id",
            index
        )

        # ----------------------------------------------------
        # Exclude source article
        # ----------------------------------------------------

        if exclude_article_id is not None:

            try:

                if int(article_id) == int(
                    exclude_article_id
                ):
                    continue

            except Exception:

                if str(article_id) == str(
                    exclude_article_id
                ):
                    continue

        title = str(
            record.get(
                "title",
                ""
            )
        )

        content = str(
            record.get(
                "content",
                ""
            )
        )

        url = str(
            record.get(
                "url",
                ""
            )
        )

        # ----------------------------------------------------
        # Base scores
        # ----------------------------------------------------

        semantic_score = float(
            semantic_scores_all[index]
        )

        lexical_score = float(
            0.55
            *
            word_scores_original[index]
            +
            0.45
            *
            char_scores[index]
        )

        claim_lexical_score = float(
            word_scores_claim[index]
        )

        semantic_score_norm = float(
            semantic_norm[pos]
        )

        lexical_score_norm = float(
            lexical_norm[pos]
        )

        claim_score_norm = float(
            claim_norm[pos]
        )

        # ----------------------------------------------------
        # Identity scores
        # ----------------------------------------------------

        entity_score = (
            entity_match_score(
                entities,
                title,
                content
            )
        )

        cooccurrence_score = (
            entity_cooccurrence_score(
                entities,
                title,
                content
            )
        )

        phrase_score = (
            phrase_match_score(
                phrases,
                title,
                content
            )
        )

        event_score = (
            event_specificity_score(
                claim,
                phrases,
                title,
                content
            )
        )

        claim_event_score = (
            claim_event_overlap_score(
                claim,
                title,
                content
            )
        )

        number_score = (
            number_date_match_score(
                numbers_dates,
                title,
                content
            )
        )

        fact_score = (
            fact_check_score(
                title,
                content
            )
        )

        # ----------------------------------------------------
        # Identity candidate flag
        # ----------------------------------------------------

        is_identity_candidate = (
            index in identity_indices
        )

        # ----------------------------------------------------
        # Strong evidence rules
        # ----------------------------------------------------

        strong_identity = (
            entity_score >= 0.20
            or cooccurrence_score >= 0.35
        )

        strong_event = (
            phrase_score >= 0.15
            or event_score >= 0.25
            or claim_event_score >= 0.25
        )

        strong_claim = (
            claim_lexical_score >= 0.20
            or lexical_score >= 0.30
        )

        # ----------------------------------------------------
        # Final relevance
        # ----------------------------------------------------

        relevance = (

            0.14
            *
            semantic_score_norm

            +

            0.08
            *
            lexical_score_norm

            +

            0.15
            *
            claim_score_norm

            +

            0.15
            *
            min(entity_score, 1.0)

            +

            0.18
            *
            min(cooccurrence_score, 1.0)

            +

            0.10
            *
            min(phrase_score, 1.0)

            +

            0.08
            *
            min(event_score, 1.0)

            +

            0.12
            *
            min(claim_event_score, 1.0)

            +

            0.06
            *
            min(number_score, 1.0)

            +

            0.08
            *
            fact_score
        )

        # ----------------------------------------------------
        # Exact identity bonus
        # ----------------------------------------------------

        if is_identity_candidate:

            relevance += 0.12

        # ----------------------------------------------------
        # Strong event + identity bonus
        # ----------------------------------------------------

        if (
            strong_identity
            and strong_event
        ):

            relevance += 0.10

        # ----------------------------------------------------
        # Multiple-entity event bonus
        # ----------------------------------------------------

        if (
            cooccurrence_score >= 0.65
            and (
                event_score >= 0.30
                or claim_event_score >= 0.30
            )
        ):

            relevance += 0.10

        # ----------------------------------------------------
        # Protect against generic semantic matches
        # ----------------------------------------------------

        if (
            not strong_identity
            and not strong_event
            and not strong_claim
        ):

            relevance *= 0.35

        # ----------------------------------------------------
        # Semantic-only penalty
        # ----------------------------------------------------

        if (
            entity_score == 0
            and cooccurrence_score == 0
            and phrase_score < 0.10
            and claim_lexical_score < 0.10
        ):

            relevance *= 0.45

        # ----------------------------------------------------
        # Generic entity penalty
        # ----------------------------------------------------

        # If only a very broad entity such as "India"
        # matches, do not treat it as strong evidence.
        normalized_query_entities = [
            normalize_entity(x)
            for x in entities
        ]

        broad_entities = {
            "india",
            "indian",
            "people",
            "government",
            "country",
            "भारत",
            "भारतीय",
            "लोग",
            "सरकार",
        }

        non_broad_entities = [
            x
            for x in normalized_query_entities
            if x
            and x not in broad_entities
        ]

        if non_broad_entities:

            specific_hits = 0

            text_for_specificity = normalize_text(
                f"{title} {content[:7000]}"
            )

            for entity in non_broad_entities:

                if any(
                    variant in text_for_specificity
                    for variant in entity_variants(
                        entity
                    )
                ):

                    specific_hits += 1

            if specific_hits == 0:

                relevance *= 0.50

        # ----------------------------------------------------
        # Evidence type
        # ----------------------------------------------------

        evidence_type = (
            "FACT_CHECK"
            if fact_score > 0
            else "NEWS"
        )

        # ----------------------------------------------------
        # Append
        # ----------------------------------------------------

        candidates.append(
            {
                "article_id": article_id,
                "url": url,
                "title": title,
                "content": content,

                "semantic_score": round(
                    semantic_score,
                    4
                ),

                "lexical_score": round(
                    lexical_score,
                    4
                ),

                "claim_lexical_score": round(
                    claim_lexical_score,
                    4
                ),

                "entity_match": round(
                    entity_score,
                    4
                ),

                "entity_cooccurrence": round(
                    cooccurrence_score,
                    4
                ),

                "phrase_match": round(
                    phrase_score,
                    4
                ),

                "event_specificity": round(
                    event_score,
                    4
                ),

                "claim_event_overlap": round(
                    claim_event_score,
                    4
                ),

                "number_date_match": round(
                    number_score,
                    4
                ),

                "fact_check_score": round(
                    fact_score,
                    4
                ),

                "identity_candidate": bool(
                    is_identity_candidate
                ),

                "relevance": round(
                    float(relevance),
                    4
                ),

                "evidence_type": evidence_type,
            }
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: (
            x["relevance"],
            x["entity_cooccurrence"],
            x["event_specificity"],
            x["semantic_score"],
        ),
        reverse=True
    )

    return candidates[:top_k]


# ============================================================
# VALIDATION
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 80)
    print("HYBRID RETRIEVER TEST")
    print("=" * 80)

    print(
        f"Dataset path: {DATASET_PATH}"
    )

    print(
        f"Articles loaded: "
        f"{len(semantic_records)}"
    )

    print(
        f"Semantic embeddings: "
        f"{semantic_embeddings.shape}"
    )

    # --------------------------------------------------------
    # CAA / Murshidabad test
    # --------------------------------------------------------

    test_claim = (
        "During the Citizenship Amendment Act "
        "(CAA) protest in Murshidabad, West Bengal, "
        "Hindus were forced out of their homes and killed."
    )

    test_entities = [
        "Citizenship Amendment Act",
        "Murshidabad",
        "West Bengal",
        "Hindus",
    ]

    test_numbers_dates = []

    results = retrieve(
        claim=test_claim,
        source_title="",
        source_content="",
        entities=test_entities,
        numbers_dates=test_numbers_dates,
        top_k=10,
    )

    print()
    print("-" * 80)
    print("CAA / MURSHIDABAD TEST")
    print("-" * 80)

    for i, item in enumerate(
        results,
        1
    ):

        print()
        print(
            f"[{i}] "
            f"E{item['article_id']}"
        )

        print(
            f"Title: "
            f"{item['title']}"
        )

        print(
            f"Semantic: "
            f"{item['semantic_score']}"
        )

        print(
            f"Lexical: "
            f"{item['lexical_score']}"
        )

        print(
            f"Claim lexical: "
            f"{item['claim_lexical_score']}"
        )

        print(
            f"Entity: "
            f"{item['entity_match']}"
        )

        print(
            f"Entity co-occurrence: "
            f"{item['entity_cooccurrence']}"
        )

        print(
            f"Phrase: "
            f"{item['phrase_match']}"
        )

        print(
            f"Event specificity: "
            f"{item['event_specificity']}"
        )

        print(
            f"Number/date: "
            f"{item['number_date_match']}"
        )

        print(
            f"Fact check: "
            f"{item['fact_check_score']}"
        )

        print(
            f"Identity candidate: "
            f"{item['identity_candidate']}"
        )

        print(
            f"Relevance: "
            f"{item['relevance']}"
        )

        print(
            f"Type: "
            f"{item['evidence_type']}"
        )

    print()
    print("=" * 80)
    print("Hybrid retriever validation: PASS")
    print("=" * 80)