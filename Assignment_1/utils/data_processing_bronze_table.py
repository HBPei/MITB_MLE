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


def process_bronze_table(snapshot_date_str, bronze_lms_directory, spark, filename, filepath):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    

    df = spark.read.csv(filepath, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + 'row count:', df.count())
    
    # save bronze table to datamart - IRL connect to database to write
    filename = os.path.splitext(filename)[0]
    partition_name = filename + "_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df


def process_bronze_table_main(bronze_lms_directory, spark, filename, filepath):
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    filename = os.path.splitext(filename)[0] + "_main"+ '.csv'
    filepath = bronze_lms_directory + filename
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)
