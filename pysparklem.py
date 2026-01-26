from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import spacy

# =========================
# Spark Session
# =========================
spark = SparkSession.builder \
    .appName("HiveToPySpark_Lemmatization_Compare_Final") \
    .enableHiveSupport() \
    .getOrCreate()

# =========================
# Read from Hive
# =========================
df = spark.sql("""
    SELECT clean_summary, clean_text, score
    FROM review_db2.reviews_hive_cleaned
""")

# IMPORTANT: control partition size
df = df.repartition(4)

# =========================
# spaCy Lemmatization (BATCHED)
# =========================
def lemmatize_partition(iterator):
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    summaries = []
    texts = []
    scores = []

    for row in iterator:
        summaries.append(row.clean_summary)
        texts.append(row.clean_text)
        scores.append(row.score)

    summary_docs = nlp.pipe(summaries, batch_size=50)
    text_docs = nlp.pipe(texts, batch_size=50)

    for clean_sum, clean_txt, sdoc, tdoc, score in zip(
        summaries, texts, summary_docs, text_docs, scores
    ):
        yield (
            clean_sum,
            " ".join(t.lemma_ for t in sdoc if not t.is_stop),
            clean_txt,
            " ".join(t.lemma_ for t in tdoc if not t.is_stop),
            score
        )

# =========================
# Output Schema
# =========================
schema = StructType([
    StructField("clean_summary", StringType(), True),
    StructField("lemma_summary", StringType(), True),
    StructField("clean_text", StringType(), True),
    StructField("lemma_text", StringType(), True),
    StructField("score", IntegerType(), True)
])

# =========================
# Apply transformation
# =========================
df_final = spark.createDataFrame(
    df.rdd.mapPartitions(lemmatize_partition),
    schema
)

# =========================
# Write Parquet
# =========================
df_final.write \
    .mode("overwrite") \
    .parquet("file:///home/talentum/project/parquet_output/reviews_nlp_compare")

spark.stop()

