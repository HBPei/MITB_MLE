import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col, regexp_replace
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark,  filename, filepath):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    filename = os.path.splitext(filename)[0]
    partition_name = filename + "_" + snapshot_date_str.replace('-','_') + '.csv'
    # filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": IntegerType(),
        "SSN":  StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType()
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

   
    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_silver_table_main(bronze_lms_directory, silver_attributes_daily_directory, spark, filename):

    filepath = bronze_lms_directory + filename
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())
    

    column_type_map = {
    "Customer_ID": StringType(),
    "Name": StringType(),
    "Age": IntegerType(),
    "SSN": StringType(),
    "Occupation": StringType(),
    "snapshot_date": DateType()
}

    # Columns requiring the same cleaning
    columns_to_clean = ["Age", "Occupation"]

    # Clean columns
    for column in columns_to_clean:
        df = df.withColumn(
            column,
            regexp_replace(col(column), r"[_]", "")
        )

    # Cast columns
    for column, new_type in column_type_map.items():
        df = df.withColumn(
            column,
            col(column).cast(new_type)
        )

    filename = "silver_" + os.path.splitext(filename)[0] + '.parquet'
    filepath = silver_attributes_daily_directory + filename
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    return df