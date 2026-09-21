from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("FakeNewsBDA-Test")
    .master("local[*]")
    .getOrCreate()
)

print("\n==============================")
print("Spark is working!")
print("Spark version:", spark.version)
print("==============================\n")

data = [
    ("Article 1", "REAL"),
    ("Article 2", "FAKE"),
    ("Article 3", "REAL"),
]

df = spark.createDataFrame(data, ["title", "label"])

df.show()

spark.stop()