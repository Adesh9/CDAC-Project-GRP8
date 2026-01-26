CREATE DATABASE IF NOT EXISTS review_db2;
USE review_db2;

DROP TABLE IF EXISTS reviews_raw;

CREATE EXTERNAL TABLE reviews_raw (
    id STRING,
    productid STRING,
    userid STRING,
    profilename STRING,
    helpfulnessnumerator INT,
    helpfulnessdenominator INT,
    score INT,
    time BIGINT,
    summary STRING,
    text STRING
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/user/talentum/data'
TBLPROPERTIES ("skip.header.line.count"="1");

-- ===============================
-- CLEAN SUMMARY + TEXT IN HIVE
-- ===============================
DROP TABLE IF EXISTS reviews_hive_cleaned;

CREATE TABLE reviews_hive_cleaned AS
SELECT
    score,

    -- Clean SUMMARY
    trim(
        regexp_replace(
            lower(summary),
            '<[^>]*>|[^a-z ]',
            ' '
        )
    ) AS clean_summary,

    -- Clean TEXT
    trim(
        regexp_replace(
            lower(text),
            '<[^>]*>|[^a-z ]',
            ' '
        )
    ) AS clean_text

FROM reviews_raw
WHERE summary IS NOT NULL
  AND text IS NOT NULL;

