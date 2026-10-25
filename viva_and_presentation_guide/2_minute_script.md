# 🎙️ 2-Minute Presentation Script (Cheat Sheet for Viva)
*Keep this page in front of you during your presentation!*

---

### Step 1: The Introduction (30 seconds)
> "Good morning, respected external examiner and faculty.
> Our project is **Veritas AI**, an **Evidence-Grounded Fake News Detection System** built on **Big Data Analytics (BDA)**.
> 
> Most AI models today guess 50/50 when they see breaking news because they are black-box classifiers that only look at writing style.
> Our system does not guess. We index a Big Data repository of **67,587 verified news articles and fact-checks** (~3.7 GB indexes) using **Apache Spark** and **Hadoop HDFS**. When a user enters a claim, our system searches the Big Data repository, pulls the exact proof, and provides a transparent verdict with cited evidence."

---

### Step 2: Explaining the BDA Implementation (30 seconds)
> "In terms of Big Data Analytics:
> 1. **Hadoop HDFS** provides partitioned, distributed storage for our 67,587-article dataset.
> 2. **Apache Spark (PySpark)** performs distributed ETL, schema enforcement, and statistical profiling across parallel partitions, generating analytics reports on class balance (45,845 Real vs 21,742 Fake) and article lengths.
> 3. **Big Data Vector Indexing**: We precomputed a 347 MB dense embedding matrix and 3.6 GB sparse TF-IDF matrices, enabling sub-second retrieval across 67,587 articles."

---

### Step 3: Explaining the AI Pipeline (30 seconds)
> "When a query is submitted:
> 1. **Claim Extraction**: Llama-3.3-70B extracts the factual claim and named entities.
> 2. **Dual-Stage Hybrid Search**: Combines multilingual semantic vectors with character/word TF-IDF to get the top 150 candidates.
> 3. **Multi-Signal Reranker**: Balances semantic similarity, lexical overlap, and entity matching to find the top 3-5 articles.
> 4. **Grounded Reasoning**: Llama-3.3-70B outputs **AUTHENTIC NEWS**, **FAKE NEWS**, or a safety fallback **UNCERTAIN** when evidence is missing."

---

### Step 4: Live Demo (30 seconds)
> "Let me demonstrate our live React dashboard running on FastAPI:
> 1. **Fake News**: Click 'CAA Bengal Violence' -> Shows red badge **FAKE NEWS** (96% confidence) citing Fact-Check #57100.
> 2. **Real News**: Click 'Misbah-ul-Haq COVID' -> Shows green badge **AUTHENTIC NEWS** (95% confidence) citing official news #34373.
> 3. **Uncertain Rumor**: Click 'Modi Passing Rumor' -> Shows amber badge **UNCERTAIN** (90% confidence), proving our model never hallucinates when evidence is absent."
