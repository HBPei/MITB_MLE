import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table
import utils.data_processing_attribute_silver_table
import utils.data_processing_financials_silver_table
import utils.data_processing_clickstream_silver_table
import utils.data_processing_lms_silver_table
import utils.data_processing_lms_gold_table
import utils.data_processing_silver_tables


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)

# create bronze datalake

backend = 'data'
filepaths = []
filenames = []
  
for filename in os.listdir(backend):
    if filename.endswith(".csv"):
        filepath = os.path.join(backend, filename)
        filepaths.append(filepath)
        filenames.append(os.path.splitext(filename)[0])

print(f'list of filepaths is {filepaths}')
print(f'list of filename is {filenames}')

for filename,filepath in zip(filenames, filepaths):
    bronze_data_directory = f"datamart/bronze/{filename}/"
    if not os.path.exists(bronze_data_directory):
        os.makedirs(bronze_data_directory)
    for date_str in dates_str_lst:
        utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_data_directory, spark, filename, filepath)


for filename,filepath in zip(filenames, filepaths):
    bronze_data_directory = f"datamart/bronze/{filename}/"
    if not os.path.exists(bronze_data_directory):
        os.makedirs(bronze_data_directory, exist_ok = True)
    utils.data_processing_bronze_table.process_bronze_table_main(bronze_data_directory, spark, filename, filepath)



# create silver datalake

## setting up file directory config:
bronze_directory = "datamart/bronze/"

attribute_filename = "features_attributes/features_attributes_main.csv"
financial_filename = "features_financials/features_financials_main.csv"
loan_filename = "lms_loan_daily/lms_loan_daily_main.csv"
clickstream_filename = "feature_clickstream/feature_clickstream_main.csv"

silver_attributes_daily_directory = "datamart/silver/features_attributes/"
silver_financials_daily_directory = "datamart/silver/features_financials/"
silver_loan_daily_directory = "datamart/silver/lms_loan_daily/"
silver_clickstream_directory ="datamart/silver/feature_clickstream/"

os.makedirs(bronze_directory, exist_ok = True)
os.makedirs(silver_attributes_daily_directory, exist_ok = True)
os.makedirs(silver_financials_daily_directory, exist_ok = True)
os.makedirs(silver_loan_daily_directory, exist_ok = True)
os.makedirs(silver_clickstream_directory, exist_ok = True)

data_start_date_str = "2023-01-01"
data_end_date_str = "2024-12-01"

utils.data_processing_silver_tables.process_silver_table_main(bronze_directory,silver_attributes_daily_directory,silver_loan_daily_directory,
                                                        silver_financials_daily_directory,silver_clickstream_directory,
                                                        spark,attribute_filename,financial_filename,loan_filename,clickstream_filename, data_start_date_str,data_end_date_str )


# create gold datalake
gold_label_store_directory = "datamart/gold/label_store/"

if not os.path.exists(gold_label_store_directory):
    os.makedirs(gold_label_store_directory)

utils.data_processing_lms_gold_table.process_labels_gold_table_main(silver_loan_daily_directory, gold_label_store_directory, spark, dpd = 30, mob = 6)

# run gold backfill
for date_str in dates_str_lst:
    utils.data_processing_lms_gold_table.process_labels_gold_table(date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd = 30, mob = 6)


folder_path = gold_label_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",df.count())

# Gold Features Preparation

silver_directory = "datamart/silver/"
os.makedirs(silver_directory, exist_ok = True)

gold_directory = "datamart/gold/"
os.makedirs(gold_directory, exist_ok = True)

silver_merged_processed_directory = "datamart/silver/merged/"
os.makedirs(silver_merged_processed_directory, exist_ok = True)

gold_features_main_directory = "datamart/gold/feature_store/"
os.makedirs(gold_features_main_directory, exist_ok = True)

silver_attribute_filename = "features_attributes/silver_features_attributes_main.parquet"
silver_financial_filename = "features_financials/silver_features_financials_main.parquet"
gold_loan_filename = "datamart/gold/label_store/gold_label_store_main_.parquet"
silver_clickstream_filename = "feature_clickstream/silver_clickstream_main_.parquet"

train_end_date = '2024-09-01'
unseen_start_date = '2024-10-01'

utils.data_processing_lms_gold_table.process_features_gold_table_main(silver_directory, spark, 
                            silver_attribute_filename, silver_financial_filename, gold_loan_filename, silver_clickstream_filename,gold_features_main_directory,
                            train_end_date, unseen_start_date,data_start_date_str,data_end_date_str ) 