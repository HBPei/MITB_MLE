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

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_table(snapshot_date_str, bronze_clickstream_directory, silver_clickstream_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "feature_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_clickstream_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    fe_list = [f'fe_{i}' for i in range(1, 21)]
    fe_dict = {fe: IntegerType() for fe in fe_list}
    column_type_map = {
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
        **fe_dict
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # save silver table - IRL connect to database to write
    partition_name = "silver_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_clickstream_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df



def process_silver_table_main(bronze_clickstream_directory, silver_clickstream_directory, spark, filename):
    # prepare arguments
    
    # connect to bronze table
    filepath = bronze_clickstream_directory + filename
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    fe_list = [f'fe_{i}' for i in range(1, 21)]
    fe_dict = {fe: IntegerType() for fe in fe_list}
    column_type_map = {
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
        **fe_dict
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # save silver table - IRL connect to database to write
    partition_name = "silver_clickstream_main_"+ '.parquet'
    filepath = silver_clickstream_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df