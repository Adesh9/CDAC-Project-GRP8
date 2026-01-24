#!/bin/bash

# -----------------------------
# Variables (easy to modify)
# -----------------------------
CSV_FILE="Reviews.csv"
HDFS_DIR="/user/talentum/data/"
DB_NAME="review_db"
TABLE_NAME="reviews"

# -----------------------------
# Step 1: Create HDFS directory
# -----------------------------
echo "Creating HDFS directory..."
hdfs dfs -mkdir -p $HDFS_DIR

# -----------------------------
# Step 2: Upload CSV to HDFS
# -----------------------------
echo "Uploading CSV file to HDFS..."
hdfs dfs -put -f $CSV_FILE $HDFS_DIR/

# -----------------------------
# Step 3: Create Hive database and table
# -----------------------------
echo "Creating Hive database and table..."

hive -e "
CREATE DATABASE IF NOT EXISTS $DB_NAME;
USE $DB_NAME;

CREATE EXTERNAL TABLE IF NOT EXISTS $TABLE_NAME (
	Id INT,
	ProductId STRING,
	UserId STRING,
	ProfileName STRING,
	HelpfulnessNumerator INT,
	HelpfulnessDenominator INT,
	Score INT ,
	Time INT,
	Summary STRING,
	Text STRING 
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE;
"
# -----------------------------
# Step 4: Load data into Hive table
# -----------------------------
echo "Loading data into Hive table..."

hive -e "
USE $DB_NAME;
LOAD DATA INPATH '/user/talentum/data/Reviews.csv'
OVERWRITE INTO TABLE review_db.reviews;
"

# -----------------------------
# Step 5: Verify inserted data
# -----------------------------
echo "Verifying data in Hive table..."

hive -e "
USE $DB_NAME;
SELECT * FROM $TABLE_NAME LIMIT 5;
"

echo "✅ Hive table created and data inserted successfully."
