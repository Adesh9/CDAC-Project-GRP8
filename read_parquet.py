from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("ReadParquet") \
    .getOrCreate()

df = spark.read.parquet(
    "file:///home/talentum/project/parquet_output/reviews_nlp_compare"
)

df.select(
    "clean_summary",
    "lemma_summary",
    "clean_text",
    "lemma_text",
    "score"
).show(5, truncate=False)

spark.stop()

