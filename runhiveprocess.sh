#!/bin/bash

hdfs dfs -mkdir -p /user/talentum/data
hdfs dfs -put Reviews.csv /user/talentum/data/reviews.csv


echo "Starting Hive preprocessing..."

hive -f hive_preprocessing.hql
if [ $? -ne 0 ]; then
    echo "Hive preprocessing failed"
    exit 1
fi

echo "Hive preprocessing completed successfully"
echo "Output table: review_db2.reviews_hive_cleaned"
echo "Ready for PySpark NLP processing"

# Prevent Jupyter auto-launch
unset PYSPARK_DRIVER_PYTHON
unset PYSPARK_DRIVER_PYTHON_OPTS

spark-submit pysparklem.py

