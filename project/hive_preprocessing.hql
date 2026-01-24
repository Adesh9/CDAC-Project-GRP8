-- =========================
-- Step 0: Database
-- =========================
CREATE DATABASE IF NOT EXISTS review_db2;
USE review_db2;

-- =========================
-- Step 1: Source table (CSV)
-- =========================
DROP TABLE IF EXISTS reviews2;

CREATE TABLE reviews2 (
    Id STRING,
    ProductId STRING,
    UserId STRING,
    ProfileName STRING,
    HelpfulnessNumerator INT,
    HelpfulnessDenominator INT,
    Score INT,
    Time BIGINT,
    Summary STRING,
    Text STRING
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
TBLPROPERTIES ("skip.header.line.count"="1");

-- Load CSV from HDFS
LOAD DATA INPATH '/user/talentum/data/reviews.csv'
OVERWRITE INTO TABLE reviews2;

-- =========================
-- Step 2: Cleaned table
-- =========================
DROP TABLE IF EXISTS reviews_hive_cleaned;

CREATE TABLE reviews_hive_cleaned (
    clean_text STRING,
    score INT
)
STORED AS ORC;

INSERT OVERWRITE TABLE reviews_hive_cleaned
SELECT
    LOWER(
        regexp_replace(
            regexp_replace(Text, '<[^>]*>', ''),
            '[^a-zA-Z ]',
            ''
        )
    ) AS clean_text,
    Score
FROM reviews2
WHERE Summary IS NOT NULL
  AND Text IS NOT NULL;

