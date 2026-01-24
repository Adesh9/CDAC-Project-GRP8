#!/usr/bin/python

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.ml import Pipeline

# ----------------------------
# Spark session
# ----------------------------
spark = SparkSession.builder \
    .appName("HiveToPySparkNLP") \
    .enableHiveSupport() \
    .getOrCreate()

# ----------------------------
# Read Hive table
# ----------------------------
df = spark.sql("""
    SELECT clean_text, score
    FROM review_db2.reviews_hive_cleaned
""")

df.show(5)
df.printSchema()

# ----------------------------
# 1. Tokenization
# ----------------------------
tokenizer = Tokenizer(
    inputCol="clean_text",
    outputCol="tokens"
)

# ----------------------------
# 2. Stopword Removal
# ----------------------------
stopword_remover = StopWordsRemover(
    inputCol="tokens",
    outputCol="filtered_tokens"
)

# ----------------------------
# 3. Lemmatization (rule-based UDF)
# ----------------------------
def simple_lemmatizer(words):
    if words is None:
        return ""
    lemmas = []
    for w in words:
        if w.endswith("ing"):
            lemmas.append(w[:-3])
        elif w.endswith("ed"):
            lemmas.append(w[:-2])
        elif w.endswith("s") and len(w) > 3:
            lemmas.append(w[:-1])
        else:
            lemmas.append(w)
    return " ".join(lemmas)

lemmatize_udf = udf(simple_lemmatizer, StringType())

# ----------------------------
# Pipeline
# ----------------------------
pipeline = Pipeline(stages=[
    tokenizer,
    stopword_remover
])

model = pipeline.fit(df)
df_processed = model.transform(df)

# Apply lemmatization
df_final = df_processed.withColumn(
    "final_text",
    lemmatize_udf(col("filtered_tokens"))
).select("final_text", "score")

# ----------------------------
# Verify output
# ----------------------------
df_final.show(5, truncate=False)
df_final.printSchema()

# ----------------------------
# Write Parquet (Local FS)
# ----------------------------
df_final.write \
    .mode("overwrite") \
    .parquet("file:///home/talentum/project/parquet_output/reviews_hive_cleaned_nlp")

spark.stop()

