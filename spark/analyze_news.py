import os
import sys

# ------------------------------------------------------------
# Windows UTF-8 output
# ------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ------------------------------------------------------------
# Environment Setup (PySpark & Hadoop)
# ------------------------------------------------------------
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYSPARK_PATH = PROJECT_ROOT / ".venv" / "Lib" / "site-packages" / "pyspark"
if PYSPARK_PATH.exists():
    os.environ["SPARK_HOME"] = str(PYSPARK_PATH)

# Remove invalid hardcoded Hadoop paths to allow Spark local mode without WinError 2
os.environ.pop("HADOOP_HOME", None)
os.environ.pop("HADOOP_CONF_DIR", None)


# ------------------------------------------------------------
# Spark
# ------------------------------------------------------------
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType
)
from pyspark.sql.functions import (
    col,
    avg,
    length,
    count,
    when,
    desc
)


print("=" * 70)
print("FAKE NEWS BDA - PYSPARK ANALYTICS")
print("=" * 70)


# ------------------------------------------------------------
# Create Spark Session
# ------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("FakeNewsBDA-Analytics")
    .master("local[*]")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
HDFS_INPUT = "hdfs://localhost:9000/fake-news-bda/processed/news_cleaned.csv"
LOCAL_INPUT = str(PROJECT_ROOT / "data" / "processed" / "news_cleaned.csv")
RESULTS_PATH = str(PROJECT_ROOT / "results" / "spark")


# ------------------------------------------------------------
# Schema
# ------------------------------------------------------------
schema = StructType([
    StructField("article_id", IntegerType(), True),
    StructField("url", StringType(), True),
    StructField("title", StringType(), True),
    StructField("content", StringType(), True),
    StructField("label", IntegerType(), True)
])


# ------------------------------------------------------------
# Read CSV from HDFS (with automatic fallback to local Big Data storage)
# ------------------------------------------------------------
print("\n[1] Reading dataset into Spark DataFrame...")
print("-" * 70)

try:
    print(f"Attempting to read from HDFS: {HDFS_INPUT}")
    df = (
        spark.read
        .option("header", "true")
        .option("multiLine", "true")
        .option("quote", '"')
        .option("escape", '"')
        .option("encoding", "UTF-8")
        .schema(schema)
        .csv(HDFS_INPUT)
    )
    # Trigger action to test connection
    df.take(1)
    print("Dataset successfully loaded from HDFS.")
except Exception as hdfs_err:
    print(f"[Notice] HDFS cluster not reachable ({hdfs_err}).")
    print(f"Reading from local Big Data storage: {LOCAL_INPUT}")
    df = (
        spark.read
        .option("header", "true")
        .option("multiLine", "true")
        .option("quote", '"')
        .option("escape", '"')
        .option("encoding", "UTF-8")
        .schema(schema)
        .csv(LOCAL_INPUT)
    )
    print("Dataset successfully loaded from Big Data storage.")


# ------------------------------------------------------------
# Schema
# ------------------------------------------------------------
print("\n[2] Dataset Schema")
print("-" * 70)

df.printSchema()


# ------------------------------------------------------------
# Cache because multiple analytics operations are performed
# ------------------------------------------------------------
df.cache()


# ------------------------------------------------------------
# Total articles
# ------------------------------------------------------------
print("\n[3] Total Articles")
print("-" * 70)

total_articles = df.count()

print(f"Total articles: {total_articles}")

EXPECTED_ROWS = 67587

if total_articles == EXPECTED_ROWS:
    print("SUCCESS: Dataset row count is correct.")
else:
    raise RuntimeError(
        f"Dataset validation failed: expected "
        f"{EXPECTED_ROWS} rows but Spark read "
        f"{total_articles}."
    )


# ------------------------------------------------------------
# Label distribution
# ------------------------------------------------------------
print("\n[4] Label Distribution")
print("-" * 70)

label_distribution = (
    df.groupBy("label")
    .count()
    .orderBy("label")
)

label_distribution.show()


# ------------------------------------------------------------
# Data quality
# ------------------------------------------------------------
print("\n[5] Data Quality")
print("-" * 70)

null_counts = df.select(
    count(when(col("article_id").isNull(), True))
        .alias("missing_article_id"),

    count(when(col("url").isNull(), True))
        .alias("missing_url"),

    count(when(col("title").isNull(), True))
        .alias("missing_title"),

    count(when(col("content").isNull(), True))
        .alias("missing_content"),

    count(when(col("label").isNull(), True))
        .alias("missing_label")
)

null_counts.show()


# ------------------------------------------------------------
# Article length statistics
# ------------------------------------------------------------
print("\n[6] Article Length Statistics")
print("-" * 70)

df_with_lengths = (
    df
    .withColumn("title_length", length(col("title")))
    .withColumn("content_length", length(col("content")))
)

length_stats = df_with_lengths.select(
    avg("title_length").alias("average_title_length"),
    avg("content_length").alias("average_content_length"),
    avg(when(col("label") == 0, col("content_length")))
        .alias("avg_content_length_label_0"),
    avg(when(col("label") == 1, col("content_length")))
        .alias("avg_content_length_label_1")
)

length_stats.show()


# ------------------------------------------------------------
# Label-wise statistics
# ------------------------------------------------------------
print("\n[7] Label-wise Statistics")
print("-" * 70)

label_statistics = (
    df_with_lengths
    .groupBy("label")
    .agg(
        count("*").alias("article_count"),
        avg("title_length").alias("avg_title_length"),
        avg("content_length").alias("avg_content_length")
    )
    .orderBy("label")
)

label_statistics.show()


# ------------------------------------------------------------
# Longest articles
# ------------------------------------------------------------
print("\n[8] Longest Articles")
print("-" * 70)

longest_articles = (
    df_with_lengths
    .select(
        "article_id",
        "title",
        "content_length",
        "label"
    )
    .orderBy(desc("content_length"))
    .limit(10)
)

longest_articles.show(
    10,
    truncate=100
)


# ------------------------------------------------------------
# Sample articles
# ------------------------------------------------------------
print("\n[9] Sample Articles")
print("-" * 70)

sample_articles = (
    df
    .select(
        "article_id",
        "title",
        "label"
    )
    .limit(10)
)

sample_articles.show(
    10,
    truncate=100
)


# ------------------------------------------------------------
# Save analytics results
# ------------------------------------------------------------
print("\n[10] Saving Analytics Results")
print("-" * 70)

os.makedirs(RESULTS_PATH, exist_ok=True)

# Save label distribution locally
label_distribution_pd = label_distribution.toPandas()
label_distribution_pd.to_csv(
    os.path.join(
        RESULTS_PATH,
        "label_distribution.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

# Save label statistics locally
label_statistics_pd = label_statistics.toPandas()
label_statistics_pd.to_csv(
    os.path.join(
        RESULTS_PATH,
        "label_statistics.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

# Save length statistics locally
length_stats_pd = length_stats.toPandas()
length_stats_pd.to_csv(
    os.path.join(
        RESULTS_PATH,
        "length_statistics.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

# Save data quality locally
null_counts_pd = null_counts.toPandas()
null_counts_pd.to_csv(
    os.path.join(
        RESULTS_PATH,
        "data_quality.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

print("Analytics results saved successfully.")


# ------------------------------------------------------------
# Final validation
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("ANALYTICS COMPLETED")
print("=" * 70)

print(f"Total articles processed: {total_articles}")
print("Expected articles:       67587")

if total_articles == EXPECTED_ROWS:
    print("Dataset validation:       PASS")
else:
    print("Dataset validation:       FAIL")

print("\nResults directory:")
print(RESULTS_PATH)

print("=" * 70)


# ------------------------------------------------------------
# Stop Spark
# ------------------------------------------------------------
spark.stop()