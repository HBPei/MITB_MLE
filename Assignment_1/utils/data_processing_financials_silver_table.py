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

from pyspark.sql.functions import col, regexp_replace, regexp_extract, coalesce, lit
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

def process_silver_table_main(bronze_lms_directory, silver_financials_daily_directory, spark, filename):

    filepath = bronze_lms_directory + filename
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())
    

    column_type_map = {
    "Customer_ID": StringType(),
    "Annual_Income": FloatType(),
    "Monthly_Inhand_Salary": FloatType(),
    "Num_Bank_Accounts": IntegerType(),
    "Num_Credit_Card": IntegerType(),
    "Interest_Rate": IntegerType(),
    "Num_of_Loan": IntegerType(),
    "Type_of_Loan": StringType(),
    "Delay_from_due_date": IntegerType(),
    "Num_of_Delayed_Payment": IntegerType(),
    "Changed_Credit_Limit": FloatType(),
    "Num_Credit_Inquiries": IntegerType(),
    "Credit_Mix": StringType(),
    "Outstanding_Debt": FloatType(),
    "Credit_Utilization_Ratio": FloatType(),
    "Credit_History_Age": StringType(),
    "Payment_of_Min_Amount": StringType(),
    "Total_EMI_per_month": FloatType(),
    "Amount_invested_monthly": FloatType(),
    "Payment_Behaviour": StringType(),
    "Monthly_Balance": FloatType(),
    "snapshot_date": DateType(),

}

    # Columns requiring the same cleaning
    columns_to_clean = ["Annual_Income", "Num_of_Loan","Num_of_Delayed_Payment","Outstanding_Debt","Amount_invested_monthly","Monthly_Balance"]

    # Clean columns
    for column in columns_to_clean:
        df = df.withColumn(
            column,
            regexp_replace(col(column), r"[_]", "")
        )

    df = df.withColumn(
            'Credit_Mix',
            regexp_replace(col('Credit_Mix'), r"[_]", "Not Defined")
        )

    # replacing "and" with "" in Type_of_Loan column
    df = df.withColumn(
    "Type_of_Loan_Cleaned",
    regexp_replace(col("Type_of_Loan"), r",?\s+and\s+", "")
)

    # replacing multiple spaces with a single space in Type_of_Loan column, replacing null values with "Not Specified"

    df = df.withColumn(
        "Type_of_Loan_Cleaned",
        coalesce(
            regexp_replace(col("Type_of_Loan"), r"\s*,\s*", ","),
            lit("Not Specified")
        )
    )


    # Cast columns
    for column, new_type in column_type_map.items():
        df = df.withColumn(
            column,
            col(column).cast(new_type)
        )

    # convert Credit_History_Age from "X Years and Y Months"
    # into a numeric value in years
    df = df.withColumn(
        "Credit_History_Age_Years",
        (
            regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Years?", 1).cast("double")
            +
            regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Months?", 1).cast("double") / 12
        )
    )


    

    filename = "silver_" + os.path.splitext(filename)[0] + '.parquet'
    filepath = silver_financials_daily_directory + filename
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    return df