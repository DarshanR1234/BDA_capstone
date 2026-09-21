import sys
import os

# Windows UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["HADOOP_HOME"] = r"C:\hadoop\hadoop-3.3.6"
os.environ["HADOOP_CONF_DIR"] = r"C:\hadoop\hadoop-3.3.6\etc\hadoop"

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType
)

spark = (
    SparkSession.builder
    .appName("FakeNewsBDA-HDFS-Test")
    .master("local[*]")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

schema = StructType([
    StructField("article_id", IntegerType(), True),
    StructField("url", StringType(), True),
    StructField("title", StringType(), True),
    StructField("content", StringType(), True),
    StructField("label", IntegerType(), True)
])

hdfs_path = (
    "hdfs://localhost:9000/"
    "fake-news-bda/processed/news_cleaned.csv"
)

print("\nReading HDFS CSV...")

df = (
    spark.read
    .option("header", True)
    .option("multiLine", True)
    .option("quote", '"')
    .option("escape", '"')
    .option("encoding", "UTF-8")
    .schema(schema)
    .csv(hdfs_path)
)

print("CSV loaded.")

print("\nCounting rows...")
total = df.count()

print(f"Total rows: {total}")

print("\nLabel counts:")
df.groupBy("label").count().orderBy("label").show()

print("\nNull counts:")
from pyspark.sql.functions import col, count, when

df.select(
    count(when(col("article_id").isNull(), True)).alias("null_article_id"),
    count(when(col("title").isNull(), True)).alias("null_title"),
    count(when(col("content").isNull(), True)).alias("null_content"),
    count(when(col("label").isNull(), True)).alias("null_label")
).show()

print("\nSample:")
df.select("article_id", "title", "label").show(3, truncate=80)

if total == 67587:
    print("\nSUCCESS: Spark read exactly 67,587 articles.")
else:
    print(f"\nPROBLEM: Spark read {total} articles instead of 67,587.")

spark.stop()