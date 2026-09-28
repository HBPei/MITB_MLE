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

from pyspark.sql.functions import col, regexp_replace, regexp_extract, coalesce, lit, trim, when
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from pyspark.sql.window import Window

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

def process_silver_table_main(bronze_lms_directory, silver_attributes_daily_directory, 
                              silver_loan_daily_directory, silver_financials_daily_directory, silver_clickstream_directory, spark, 
                              attribute_filename, financial_filename, loan_filename, clickstream_filename,start_date_str, end_date_str ):

    attribute_main_filepath = bronze_lms_directory + attribute_filename
    attribute_df = spark.read.csv(attribute_main_filepath, header=True, inferSchema=True)
    print('loaded from:', attribute_main_filepath, 'row count:', attribute_df.count())

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
        attribute_df = attribute_df.withColumn(
            column,
            regexp_replace(col(column), r"[_]", "")
        )

    # Cast columns
    for column, new_type in column_type_map.items():
        attribute_df = attribute_df.withColumn(
            column,
            col(column).cast(new_type)
        )

    silver_attribute_filename = "silver_" + os.path.splitext(os.path.basename(attribute_filename))[0] + '.parquet'
    silver_attribute_filepath = silver_attributes_daily_directory + silver_attribute_filename
    attribute_df.write.mode("overwrite").parquet(silver_attribute_filepath)
    print('saved to:', silver_attribute_filepath)


    for date_str in dates_str_lst:
            snapshot_date_str = datetime.strptime(date_str, "%Y-%m-%d")
            df = attribute_df.filter(col('snapshot_date') == snapshot_date_str)
            # save bronze table to datamart - IRL connect to database to write
            silver_attribute_partition_filename = os.path.splitext(os.path.basename(attribute_filename))[0] + "_" + date_str.replace('-','_') + '.parquet'
            silver_attribute_partition_filepath = silver_attributes_daily_directory + silver_attribute_partition_filename
            df.write.mode("overwrite").parquet(silver_attribute_partition_filepath)
            print('saved to:', silver_attribute_partition_filepath)

    ### Financial df processing

    financial_main_filepath = bronze_lms_directory + financial_filename
    financial_df = spark.read.csv(financial_main_filepath, header=True, inferSchema=True)
    print('loaded from:', financial_main_filepath, 'row count:', financial_df.count())

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
        financial_df = financial_df.withColumn(column,regexp_replace(col(column), r"[_]", ""))

    financial_df = financial_df.withColumn(
            'Credit_Mix',regexp_replace(col('Credit_Mix'), r"[_]", "Not Defined"))

    # # replacing "and" with "" in Type_of_Loan column
    # financial_df = financial_df.withColumn(
    # "Type_of_Loan_Cleaned", regexp_replace(col("Type_of_Loan"), r",?\s+and\s+", ""))

    # # replacing multiple spaces with a single space in Type_of_Loan column, replacing null values with "Not Specified"

    # financial_df = financial_df.withColumn(
    # "Type_of_Loan_Cleaned", coalesce(regexp_replace(col("Type_of_Loan"), r"\s*,\s*", ","), lit("Not Specified")))

    # financial_df = financial_df.withColumn(
    # "Type_of_Loan_Cleaned",F.trim(F.col("Type_of_Loan_Cleaned")))


    financial_df = financial_df.withColumn("Type_of_Loan_Cleaned",coalesce(col("Type_of_Loan"),lit("Not Specified")))

    # Remove "and"
    financial_df = financial_df.withColumn("Type_of_Loan_Cleaned",regexp_replace(col("Type_of_Loan_Cleaned"),r"\s+and\s+",""))

    # Remove spaces around commas
    financial_df = financial_df.withColumn("Type_of_Loan_Cleaned",regexp_replace(col("Type_of_Loan_Cleaned"),r"\s*,\s*",","))

    # Strip leading/trailing whitespace
    financial_df = financial_df.withColumn("Type_of_Loan_Cleaned",trim(col("Type_of_Loan_Cleaned"))
    )

    # calculate number of loans based on Type_of_Loan_Cleaned column

    financial_df = financial_df.withColumn(
    "Num_of_Loan_Cleaned", F.size(F.split(F.col("Type_of_Loan_Cleaned"),",")))

    # create new columns for each loan type based on Type_of_Loan_Cleaned column (one hot encoding)
    loan_types = (
    financial_df.select(F.explode(F.split(F.col("Type_of_Loan_Cleaned"),",")).alias("Loan_Type"))
    .distinct().rdd.flatMap(lambda x: x).collect())

    for loan_type in loan_types:

        column_name = ("Loan_"+ loan_type.replace(" ", "_").replace("-", "_"))

        financial_df = financial_df.withColumn(
            column_name,F.when(F.array_contains(F.split(F.col("Type_of_Loan_Cleaned"),","),loan_type),1).otherwise(0))

    # One-hot encode Credit_Mix

    credit_mix_values = (financial_df.select("Credit_Mix").distinct().rdd.flatMap(lambda x: x).collect())

    for value in credit_mix_values:

        column_name = ("Credit_Mix_"+ value.replace(" ", "_"))

        financial_df = financial_df.withColumn(column_name,F.when(F.col("Credit_Mix") == value,1).otherwise(0))

    # One-hot encode Payment_of_Min_Amount

    payment_values = (financial_df.select("Payment_of_Min_Amount").distinct().rdd.flatMap(lambda x: x).collect())

    for value in payment_values:

        column_name = ("Payment_of_Min_Amount_"+ value.replace(" ", "_"))

        financial_df = financial_df.withColumn(column_name,F.when(F.col("Payment_of_Min_Amount") == value,1).otherwise(0))

    # convert Credit_History_Age from "X Years and Y Months" into a numeric value in years
    financial_df = financial_df.withColumn(
        "Credit_History_Age_Years",
        (
            regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Years?", 1).cast("double")
            +
            regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Months?", 1).cast("double") / 12
        )
    )
     # imputing !@9#%8 in Payment_Behaviour column with "High_spent_Small_value_payments"
    financial_df = financial_df.withColumn(
    'Payment_Behaviour_Imputed',when(col('Payment_Behaviour') == '!@9#%8','High_spent_Small_value_payments').otherwise(col('Payment_Behaviour')))


    # Cast columns
    for column, new_type in column_type_map.items():
        financial_df = financial_df.withColumn(
            column,
            col(column).cast(new_type)
        )
    
    silver_financial_filename = "silver_" + os.path.splitext(os.path.basename(financial_filename))[0] + '.parquet'
    silver_financial_filepath = silver_financials_daily_directory + silver_financial_filename
    financial_df.write.mode("overwrite").parquet(silver_financial_filepath)
    print('saved to:', silver_financial_filepath)


    for date_str in dates_str_lst:
        snapshot_date_str = datetime.strptime(date_str, "%Y-%m-%d")
        df = financial_df.filter(col('snapshot_date') == snapshot_date_str)
        # save bronze table to datamart - IRL connect to database to write
        silver_financials_partition_filename = os.path.splitext(os.path.basename(financial_filename))[0] + "_" + date_str.replace('-','_') + '.parquet'
        silver_partition_filepath = silver_financials_daily_directory + silver_financials_partition_filename
        df.write.mode("overwrite").parquet(silver_partition_filepath)
        print('saved to:', silver_partition_filepath)

    ## loan df processing

    # connect to bronze table
    loan_main_filepath = bronze_lms_directory + loan_filename
    loan_df = spark.read.csv(loan_main_filepath, header=True, inferSchema=True)
    print('loaded from:', loan_main_filepath, 'row count:', loan_df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        loan_df = loan_df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    loan_df = loan_df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    loan_df = loan_df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    loan_df = loan_df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    loan_df = loan_df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    silver_loan_filename = "silver_loan_daily_main_"  + '.parquet'
    silver_loan_filepath = silver_loan_daily_directory + silver_loan_filename
    loan_df.write.mode("overwrite").parquet(silver_loan_filepath)
    print('saved to:', silver_loan_filepath)

    # connect to bronze clickstream table
    clickstream_filepath = bronze_lms_directory + clickstream_filename
    clickstream_df = spark.read.csv(clickstream_filepath, header=True, inferSchema=True)
    print('loaded from:', clickstream_filepath, 'row count:', clickstream_df.count())

    
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
        clickstream_df = clickstream_df.withColumn(column, col(column).cast(new_type))

    # save silver table - IRL connect to database to write
    silver_clickstream_name = "silver_clickstream_main_"+ '.parquet'
    silver_clickstream_filepath = silver_clickstream_directory + silver_clickstream_name
    clickstream_df.write.mode("overwrite").parquet(silver_clickstream_filepath)
    print('saved to:', silver_clickstream_filepath)

    for date_str in dates_str_lst:
        snapshot_date_str = datetime.strptime(date_str, "%Y-%m-%d")
        df = clickstream_df.filter(col('snapshot_date') == snapshot_date_str)
        # save bronze table to datamart - IRL connect to database to write
        silver_clickstreamm_partition_filename = os.path.splitext(os.path.basename(silver_clickstream_name))[0] + "_" + date_str.replace('-','_') + '.parquet'
        silver_partition_clickstream_filepath = silver_clickstream_directory + silver_clickstreamm_partition_filename
        df.write.mode("overwrite").parquet(silver_partition_clickstream_filepath)
        print('saved to:', silver_partition_clickstream_filepath)
        

def augment_silver_table_main(silver_lms_directory, silver_merged_processed_directory, spark, 
                            silver_attribute_filename, silver_financial_filename, gold_loan_filename, silver_clickstream_filename,gold_features_main_directory,
                            train_end_date, unseen_start_date,data_start_date_str,data_end_date_str):

    # loading the dataframes from the silver tables
    attribute_main_filepath = silver_lms_directory + silver_attribute_filename
    attribute_df = spark.read.parquet(attribute_main_filepath)
    print('loaded from:', attribute_main_filepath, 'row count:', attribute_df.count())

    financial_main_filepath = silver_lms_directory + silver_financial_filename
    financial_df = spark.read.parquet(financial_main_filepath)
    print('loaded from:', financial_main_filepath, 'row count:', financial_df.count())

    loan_main_filepath = gold_loan_filename
    loan_df = spark.read.parquet(loan_main_filepath)
    print('loaded from:', loan_main_filepath, 'row count:', loan_df.count())

    clickstream_main_filepath = silver_lms_directory + silver_clickstream_filename
    clickstream_df = spark.read.parquet(clickstream_main_filepath)
    print('loaded from:', clickstream_main_filepath, 'row count:', clickstream_df.count())

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
    
    dates_str_lst = generate_first_of_month_dates(data_start_date_str, data_end_date_str)

    # renaming snapshot date columns:

    attribute_df = attribute_df.withColumnRenamed("snapshot_date", "attributes_snapshot_date")

    financial_df = financial_df.withColumnRenamed("snapshot_date", "financials_snapshot_date")

    loan_df = loan_df.withColumnRenamed("snapshot_date", "labels_snapshot_date")

    clickstream_df = clickstream_df.withColumnRenamed("snapshot_date", "clicks_snapshot_date")

    # merging the dataframes on Customer_ID and selecting relevant columns

    label_cs = (
    loan_df.select(
        "loan_id",
        "Customer_ID",
        "label",
        "labels_snapshot_date"
    )
    .alias("l").join(
        clickstream_df.alias("c"),
        on=(
            (F.col("l.Customer_ID") == F.col("c.Customer_ID")) &
            (F.col("l.labels_snapshot_date") == F.col("c.clicks_snapshot_date"))
        ),
        how="left"
    ).select(
        "l.*",
        *[
            F.col(f"c.{col}")
            for col in clickstream_df.columns
            if col not in ["Customer_ID", "clicks_snapshot_date"]
        ]
    )
)

    combined_df = (
        label_cs
        .join(attribute_df.select("Customer_ID","Age","Occupation","attributes_snapshot_date"),
        on="Customer_ID",how="left")
        .join(financial_df,on="Customer_ID",how="left"))
    

    # combined_df = (
    # loan_df.select("loan_id","Customer_ID","label","labels_snapshot_date")
    # .join(attribute_df.select("Customer_ID","Age","Occupation","attributes_snapshot_date"),
    # on="Customer_ID",how="left")
    # .join(financial_df,on="Customer_ID",how="left"))

    # c = combined_df.alias("c")
    # cl = clickstream_df.alias("cl")

    # combined_df = (c.join(cl,
    #         on=(
    #             (F.col("c.Customer_ID") == F.col("cl.Customer_ID")) &
    #             (F.col("c.labels_snapshot_date") == F.col("cl.clicks_snapshot_date"))
    #         ),how="left").select(
    #         "c.*",
    #         *[
    #             F.col(f"cl.{col}")
    #             for col in clickstream_df.columns
    #             if col not in ["Customer_ID", "clicks_snapshot_date"]
    #         ]))


    # filtering the combined dataframe into train and unseen based on the labels_snapshot_date for imputation

    train = combined_df.filter(
        F.col("labels_snapshot_date") <=lit(train_end_date)
    )

    unseen = combined_df.filter(
        F.col("labels_snapshot_date") >= lit(unseen_start_date)
    )

    train_count = train.count()
    unseen_count = unseen.count()

    print("Train:", train_count)
    print("Unseen:", unseen_count)

    print("Train label counts:")

    train_label_counts = (train.groupBy("label").count().withColumn("proportion",col("count") / train_count).orderBy("label"))

    train_label_counts.show()

    print("Unseen label counts:")

    unseen_label_counts = (unseen.groupBy("label").count().withColumn("proportion",col("count") / unseen_count).orderBy("label"))

    unseen_label_counts.show()

    #  Age Imputation

    MIN_BIN_COUNT = 20
    MIN_DISTANCE_GAP = 0.05

    def create_age_bin_column(df, age_col, edges, labels, output_col="Age_Bin"):

        if len(edges) < 2:
            return df.withColumn(output_col, F.lit(None).cast("string"))

        condition = None

        for i, label in enumerate(labels):

            lower = edges[i]
            upper = edges[i + 1]

            if i == 0:
                current_condition = (
                    (F.col(age_col) >= F.lit(lower)) &(F.col(age_col) <= F.lit(upper))
                )
            else:
                current_condition = (
                    (F.col(age_col) > F.lit(lower)) & (F.col(age_col) <= F.lit(upper))
                )

            if condition is None:
                condition = F.when(current_condition, F.lit(label))
            else:
                condition = condition.when(current_condition, F.lit(label))

        return df.withColumn(
            output_col,condition.otherwise(F.lit(None))
        )

    train = train.withColumn(
    "Age_clean",
    F.when(
        F.col("Age").between(18, 100),F.col("Age")).otherwise(F.lit(None)))

    age_history_inconsistent = (F.col("Credit_History_Age_Years").isNotNull() 
                                & F.col("Age_clean").isNotNull()
                                & (F.col("Credit_History_Age_Years")>F.col("Age_clean")))

    # Flag inconsistency BEFORE setting Age_clean to null

    train = train.withColumn("Age_History_Inconsistent",F.when(age_history_inconsistent,1).otherwise(0))

    train = train.withColumn("Age_was_invalid",F.when(F.col("Age_clean").isNull() & F.col("Age").isNotNull(),1).otherwise(0))

    valid_train_df = train.filter(F.col("Age_clean").isNotNull()
                                  & F.col("Annual_Income").isNotNull()
                                  & F.col("Occupation").isNotNull())

    valid_train_df = valid_train_df.withColumn("Log_Annual_Income",F.log1p(F.greatest(F.col("Annual_Income"),F.lit(0))))

    age_percentiles = valid_train_df.approxQuantile("Age_clean",[0.0, 0.25, 0.50, 0.75, 1.0],0.0)

    age_edges = sorted(set(age_percentiles))

    print("Initial Age edges:")
    print(age_edges)

    age_labels = []

    for i in range(len(age_edges) - 1):

        lower = int(age_edges[i])
        upper = int(age_edges[i + 1])

        if i == 0:
            label = f"{lower}-{upper}"
        else:
            label = f"{lower + 1}-{upper}"

        age_labels.append(label)

    valid_train_df = create_age_bin_column(valid_train_df, age_col="Age_clean",edges=age_edges,labels=age_labels,output_col="Age_Bin")

    age_income_lookup = (valid_train_df.groupBy("Occupation","Age_Bin").agg(
        F.percentile_approx("Log_Annual_Income",0.5,10000).alias("log_income_median"),
        F.percentile_approx("Age_clean",0.5,10000).alias("age_median"),
        F.count("Age_clean").alias("count")))

    age_income_lookup = age_income_lookup.filter(F.col("count") >= MIN_BIN_COUNT)

    print("Age income lookup:")
    age_income_lookup.show()

    occupation_median = (valid_train_df.groupBy("Occupation").agg(
        F.percentile_approx("Age_clean",0.5,10000).alias("occupation_age_median")))

    overall_median = (valid_train_df.agg(F.percentile_approx("Age_clean",0.5,10000).alias("overall_age_median")).first()["overall_age_median"])

    print("Overall median age:", overall_median)

    # TRANSFORM AGE FUNCTION

    def transform_age_spark(
        df,
        age_income_lookup,
        occupation_median,
        overall_median,
        age_edges_final=None,
        age_labels_final=None,
        min_distance_gap=0.05
    ):
  
        # Clean Age
  

        df = df.withColumn("Age_clean",F.when(F.col("Age").between(18, 100),F.col("Age")).otherwise(F.lit(None)))


        # Age vs Credit History Age
    

        age_history_inconsistent = (
            F.col("Credit_History_Age_Years").isNotNull()
            & F.col("Age_clean").isNotNull()
            & (F.col("Credit_History_Age_Years") > F.col("Age_clean")))

        df = df.withColumn(
            "Age_History_Inconsistent",
            F.when(
                age_history_inconsistent,
                1
            ).otherwise(0)
        )


        # Invalidate inconsistent Age

        df = df.withColumn(
            "Age_clean",
            F.when(
                age_history_inconsistent,
                F.lit(None)
            ).otherwise(
                F.col("Age_clean")
            )
        )


        # --------------------------------------------------------
        # 3. Age invalid flag
        # --------------------------------------------------------

        df = df.withColumn(
            "Age_was_invalid",
            F.when(
                F.col("Age_clean").isNull()
                &
                F.col("Age").isNotNull(),
                1
            ).otherwise(0)
        )


        # --------------------------------------------------------
        # 4. Create unique row ID
        # --------------------------------------------------------

        df = df.withColumn(
            "_age_row_id",
            F.monotonically_increasing_id()
        )


        # --------------------------------------------------------
        # 5. Log Annual Income
        # --------------------------------------------------------

        df = df.withColumn(
            "_log_income",
            F.when(
                F.col("Annual_Income").isNotNull(),
                F.log1p(
                    F.greatest(
                        F.col("Annual_Income"),
                        F.lit(0)
                    )
                )
            )
        )


        # --------------------------------------------------------
        # 6. Identify rows requiring imputation
        # --------------------------------------------------------

        invalid_df = df.filter(
            F.col("Age_clean").isNull()
        )


        # --------------------------------------------------------
        # 7. Create candidate occupation + Age_Bin combinations
        #
        # Equivalent to:
        #
        # candidates =
        # age_income_lookup[
        #     age_income_lookup['Occupation']
        #     == occupation
        # ]
        # --------------------------------------------------------

        candidates = (
            invalid_df
            .filter(
                F.col("Occupation").isNotNull()
                &
                F.col("Annual_Income").isNotNull()
            )
            .join(
                age_income_lookup,
                on="Occupation",
                how="inner"
            )
        )


        # --------------------------------------------------------
        # 8. Calculate distance between income and group median
        # --------------------------------------------------------

        candidates = candidates.withColumn(
            "_distance",
            F.abs(
                F.col("log_income_median")
                -
                F.col("_log_income")
            )
        )


        # --------------------------------------------------------
        # 9. Rank candidate Age bins by income distance
        # --------------------------------------------------------

        distance_window = (
            Window
            .partitionBy("_age_row_id")
            .orderBy(
                F.col("_distance").asc()
            )
        )


        candidates = candidates.withColumn(
            "_candidate_rank",
            F.row_number().over(distance_window)
        )


        # --------------------------------------------------------
        # 10. Keep best and second-best candidates
        # --------------------------------------------------------

        best_candidates = (
            candidates
            .filter(
                F.col("_candidate_rank") <= 2
            )
            .groupBy("_age_row_id")
            .agg(

                F.max(
                    F.when(
                        F.col("_candidate_rank") == 1,
                        F.col("age_median")
                    )
                ).alias("_best_age"),

                F.max(
                    F.when(
                        F.col("_candidate_rank") == 1,
                        F.col("_distance")
                    )
                ).alias("_best_distance"),

                F.max(
                    F.when(
                        F.col("_candidate_rank") == 2,
                        F.col("_distance")
                    )
                ).alias("_second_distance")
            )
        )


        # --------------------------------------------------------
        # 11. Determine strength of evidence
        #
        # If only one candidate exists:
        #     strong evidence
        #
        # Otherwise:
        #     second distance - best distance >= 0.05
        # --------------------------------------------------------

        best_candidates = best_candidates.withColumn(
            "_strong_income_evidence",
            F.when(
                F.col("_second_distance").isNull(),
                True
            )
            .when(
                (
                    F.col("_second_distance")
                    -
                    F.col("_best_distance")
                )
                >= F.lit(min_distance_gap),
                True
            )
            .otherwise(False)
        )


        # --------------------------------------------------------
        # 12. Join occupation median
        # --------------------------------------------------------

        result = (
            df

            .join(
                best_candidates.select(
                    "_age_row_id",
                    "_best_age",
                    "_strong_income_evidence"
                ),
                on="_age_row_id",
                how="left"
            )

            .join(
                occupation_median,
                on="Occupation",
                how="left"
            )
        )


        # --------------------------------------------------------
        # 13. Impute Age
        #
        # Priority:
        #
        # 1. Original valid Age
        # 2. Occupation + Income + Age_Bin
        # 3. Occupation median
        # 4. Overall median
        # --------------------------------------------------------

        result = result.withColumn(
            "Age_imputed",
            F.when(
                F.col("Age_clean").isNotNull(),
                F.col("Age_clean")
            )
            .when(
                F.col("_strong_income_evidence") == True,
                F.col("_best_age")
            )
            .when(
                F.col("occupation_age_median").isNotNull(),
                F.col("occupation_age_median")
            )
            .otherwise(
                F.lit(overall_median)
            )
        )


        # --------------------------------------------------------
        # 14. Store imputation method
        # --------------------------------------------------------

        result = result.withColumn(
            "Age_imputation_method",
            F.when(
                F.col("Age_clean").isNotNull(),
                F.lit("original")
            )
            .when(
                F.col("_strong_income_evidence") == True,
                F.lit("occupation_income_age_bin")
            )
            .when(
                F.col("occupation_age_median").isNotNull(),
                F.lit("occupation_median")
            )
            .otherwise(
                F.lit("overall_median")
            )
        )


        # --------------------------------------------------------
        # 15. Round Age
        # --------------------------------------------------------

        result = result.withColumn(
            "Age_imputed",
            F.round(
                F.col("Age_imputed")
            ).cast("int")
        )


        # --------------------------------------------------------
        # 16. Apply TRAIN-DERIVED FINAL AGE BINS
        # --------------------------------------------------------

        if (
            age_edges_final is not None
            and
            age_labels_final is not None
        ):

            result = create_age_bin_column(
                result,
                age_col="Age_imputed",
                edges=age_edges_final,
                labels=age_labels_final,
                output_col="Age_Bin"
            )


        # --------------------------------------------------------
        # 17. Drop temporary columns
        # --------------------------------------------------------

        result = result.drop(
            "_age_row_id",
            "_log_income",
            "_best_age",
            "_best_distance",
            "_second_distance",
            "_strong_income_evidence",
            "occupation_age_median"
        )


        return result


    # ============================================================
    # 14. TRANSFORM TRAIN
    #
    # No final Age bins yet.
    # ============================================================

    train_processed = transform_age_spark(
        train,
        age_income_lookup=age_income_lookup,
        occupation_median=occupation_median,
        overall_median=overall_median,
        age_edges_final=None,
        age_labels_final=None,
        min_distance_gap=MIN_DISTANCE_GAP
    )


    # ============================================================
    # 15. DERIVE FINAL AGE BINS FROM IMPUTED TRAIN
    # ============================================================

    age_percentiles_final = train_processed.approxQuantile(
        "Age_imputed",
        [0.0, 0.25, 0.50, 0.75, 1.0],
        0.0
    )

    age_edges_final = sorted(
        set(age_percentiles_final)
    )


    print("Final Age edges:")
    print(age_edges_final)


    # ============================================================
    # 16. CREATE FINAL AGE BIN LABELS
    # ============================================================

    age_labels_final = []

    for i in range(len(age_edges_final) - 1):

        lower = int(age_edges_final[i])
        upper = int(age_edges_final[i + 1])

        if i == 0:

            label = f"{lower}-{upper}"

        else:

            label = f"{lower + 1}-{upper}"

        age_labels_final.append(label)


    print("Final Age labels:")
    print(age_labels_final)

    # ============================================================
# 17. APPLY FINAL AGE BINS TO TRAIN
# ============================================================

    train_processed = create_age_bin_column(
        train_processed,
        age_col="Age_imputed",
        edges=age_edges_final,
        labels=age_labels_final,
        output_col="Age_Bin"
    )

    unseen_processed = transform_age_spark(
        unseen,
        age_income_lookup=age_income_lookup,
        occupation_median=occupation_median,
        overall_median=overall_median,
        age_edges_final=age_edges_final,
        age_labels_final=age_labels_final,
        min_distance_gap=MIN_DISTANCE_GAP
    )

    # CHECK RESULTS


    print("TRAIN")
    print("======")

    print("Total rows:", train_processed.count())

    train_processed.groupBy("Age_imputation_method").count().orderBy("Age_imputation_method").show()

    train_processed.groupBy("Age_Bin").count().orderBy("Age_Bin").show()


    print("UNSEEN")
    print("=======")

    print("Total rows:", unseen_processed.count())

    unseen_processed.groupBy("Age_imputation_method").count().orderBy("Age_imputation_method").show()

    unseen_processed.groupBy("Age_Bin").count().orderBy("Age_Bin").show()



    # SAMPLE OUTPUT


    train_processed.select(
        "Customer_ID",
        "Age",
        "Credit_History_Age_Years",
        "Age_clean",
        "Age_imputed",
        "Age_Bin",
        "Age_was_invalid",
        "Age_History_Inconsistent",
        "Age_imputation_method"
    ).show(20, truncate=False)


    unseen_processed.select(
        "Customer_ID",
        "Age",
        "Credit_History_Age_Years",
        "Age_clean",
        "Age_imputed",
        "Age_Bin",
        "Age_was_invalid",
        "Age_History_Inconsistent",
        "Age_imputation_method"
    ).show(20, truncate=False)


    # Num of bank accounts and credit cards imputation

    # FIT TRAIN-DERIVED ACCOUNT IMPUTATION PARAMETERS


    def fit_account_imputation(
        df,
        raw_col,
        clean_col,
        p95_group_min
    ):
        
        # P95
        p95 = df.approxQuantile(raw_col, [0.95], 0.0)[0]
        
        print(f"{raw_col} P95:", p95)
        print(
            f"Null records for {raw_col}:",
            df.filter(F.col(raw_col).isNull()).count()
        )

        # Clean
        df = df.withColumn(
            clean_col,
            F.when(
                F.col(raw_col).between(0, p95),
                F.col(raw_col)
            )
        )

        # Group median: Occupation + Age_Bin
        group_stats = (
            df.groupBy("Occupation", "Age_Bin")
            .agg(
                F.percentile_approx(
                    clean_col, 0.5, 10000
                ).alias("group_median"),
                F.count(clean_col).alias("count")
            )
            .filter(F.col("count") >= p95_group_min)
            .select(
                "Occupation",
                "Age_Bin",
                F.col("group_median").alias("median")
            )
        )

        # Occupation median
        occupation_median = (
            df.groupBy("Occupation")
            .agg(
                F.percentile_approx(
                    clean_col, 0.5, 10000
                ).alias("occupation_median")
            )
        )

        # Overall median
        overall_median = (
            df.agg(
                F.percentile_approx(
                    clean_col, 0.5, 10000
                ).alias("median")
            )
            .first()["median"]
        )

        return df, p95, group_stats, occupation_median, overall_median

    # TRANSFORM ACCOUNT COUNT
  

    def transform_account_count(
        df,
        raw_col,
        clean_col,
        imputed_col,
        method_col,
        p95,
        group_stats,
        occupation_median,
        overall_median
    ):

        # Clean using TRAIN P95
        df = df.withColumn(
            clean_col,
            F.when(
                F.col(raw_col).between(0, p95),
                F.col(raw_col)
            )
        )

        # Invalid flag
        invalid_col = clean_col.replace(
            "_clean",
            "_Was_Invalid"
        )

        df = df.withColumn(
            invalid_col,
            F.when(
                F.col(clean_col).isNull()
                & F.col(raw_col).isNotNull(),
                1
            ).otherwise(0)
        )

        # Unique row ID for joining lookup results
        df = df.withColumn(
            "_row_id",
            F.monotonically_increasing_id()
        )

        # Occupation + Age_Bin lookup
        df = (
            df.join(
                group_stats.withColumnRenamed(
                    "median",
                    "_group_median"
                ),
                on=["Occupation", "Age_Bin"],
                how="left"
            )
            .join(
                occupation_median.withColumnRenamed(
                    "occupation_median",
                    "_occupation_median"
                ),
                on="Occupation",
                how="left"
            )
        )

        # Imputation hierarchy
        df = df.withColumn(
            imputed_col,
            F.round(
                F.coalesce(
                    F.col(clean_col),
                    F.col("_group_median"),
                    F.col("_occupation_median"),
                    F.lit(overall_median)
                )
            ).cast("int")
        )

        # Method
        df = df.withColumn(
            method_col,
            F.when(
                F.col(clean_col).isNotNull(),
                "original"
            )
            .when(
                F.col("_group_median").isNotNull(),
                "occupation_age_bin_median"
            )
            .when(
                F.col("_occupation_median").isNotNull(),
                "occupation_median"
            )
            .otherwise(
                "overall_median"
            )
        )

        return df.drop(
            "_row_id",
            "_group_median",
            "_occupation_median"
        )



    # FIT BANK ACCOUNT PARAMETERS FROM TRAIN ONLY


    (
        train_processed,
        bank_p95,
        bank_group_stats,
        bank_occupation_median,
        bank_overall_median
    ) = fit_account_imputation(
        train_processed,
        raw_col="Num_Bank_Accounts",
        clean_col="Bank_Accounts_clean",
        p95_group_min=30
    )


    # FIT CREDIT CARD PARAMETERS FROM TRAIN ONLY


    (
        train_processed,
        card_p95,
        card_group_stats,
        card_occupation_median,
        card_overall_median
    ) = fit_account_imputation(
        train_processed,
        raw_col="Num_Credit_Card",
        clean_col="Credit_Cards_clean",
        p95_group_min=10
    )


  
    # IMPUTE TRAIN


    train_processed = transform_account_count(
        train_processed,
        "Num_Bank_Accounts",
        "Bank_Accounts_clean",
        "Bank_Accounts_imputed",
        "Bank_Accounts_imputation_method",
        bank_p95,
        bank_group_stats,
        bank_occupation_median,
        bank_overall_median
    )

    train_processed = transform_account_count(
        train_processed,
        "Num_Credit_Card",
        "Credit_Cards_clean",
        "Credit_Cards_imputed",
        "Credit_Cards_imputation_method",
        card_p95,
        card_group_stats,
        card_occupation_median,
        card_overall_median
    )

  
    # IMPUTE UNSEEN USING TRAIN-DERIVED PARAMETERS

    unseen_processed = transform_account_count(
        unseen_processed,
        "Num_Bank_Accounts",
        "Bank_Accounts_clean",
        "Bank_Accounts_imputed",
        "Bank_Accounts_imputation_method",
        bank_p95,
        bank_group_stats,
        bank_occupation_median,
        bank_overall_median
    )

    unseen_processed = transform_account_count(
        unseen_processed,
        "Num_Credit_Card",
        "Credit_Cards_clean",
        "Credit_Cards_imputed",
        "Credit_Cards_imputation_method",
        card_p95,
        card_group_stats,
        card_occupation_median,
        card_overall_median
    )


  
    # FIT FINAL QUARTILE BINS FROM TRAIN
  

    def fit_account_bins(df, col):

        edges = sorted(
            set(
                df.approxQuantile(
                    col,
                    [0.0, 0.25, 0.50, 0.75, 1.0],
                    0.0
                )
            )
        )

        labels = []

        for i in range(len(edges) - 1):

            lower = int(edges[i])
            upper = int(edges[i + 1])

            labels.append(
                f"{lower}-{upper}"
                if i == 0
                else f"{lower + 1}-{upper}"
            )

        return edges, labels


    bank_account_edges, bank_account_labels = fit_account_bins(
        train_processed,
        "Bank_Accounts_imputed"
    )

    credit_card_edges, credit_card_labels = fit_account_bins(
        train_processed,
        "Credit_Cards_imputed"
    )



    # APPLY SAME TRAIN-DERIVED BINS TO TRAIN + UNSEEN


    train_processed = create_age_bin_column(
        train_processed,
        "Bank_Accounts_imputed",
        bank_account_edges,
        bank_account_labels,
        "Bank_Accounts_Imputed_Bin"
    )

    train_processed = create_age_bin_column(
        train_processed,
        "Credit_Cards_imputed",
        credit_card_edges,
        credit_card_labels,
        "Credit_Cards_Imputed_Bin"
    )

    unseen_processed = create_age_bin_column(
        unseen_processed,
        "Bank_Accounts_imputed",
        bank_account_edges,
        bank_account_labels,
        "Bank_Accounts_Imputed_Bin"
    )

    unseen_processed = create_age_bin_column(
        unseen_processed,
        "Credit_Cards_imputed",
        credit_card_edges,
        credit_card_labels,
        "Credit_Cards_Imputed_Bin"
    )


    # SAMPLE OUTPUT - BANK ACCOUNTS & CREDIT CARDS


    train_processed.select(
        "Customer_ID",
        "Num_Bank_Accounts",
        "Bank_Accounts_clean",
        "Bank_Accounts_imputed",
        "Bank_Accounts_Imputed_Bin",
        "Bank_Accounts_Was_Invalid",
        "Bank_Accounts_imputation_method",
        "Num_Credit_Card",
        "Credit_Cards_clean",
        "Credit_Cards_imputed",
        "Credit_Cards_Imputed_Bin",
        "Credit_Cards_Was_Invalid",
        "Credit_Cards_imputation_method"
    ).show(20, truncate=False)


    unseen_processed.select(
        "Customer_ID",
        "Num_Bank_Accounts",
        "Bank_Accounts_clean",
        "Bank_Accounts_imputed",
        "Bank_Accounts_Imputed_Bin",
        "Bank_Accounts_Was_Invalid",
        "Bank_Accounts_imputation_method",
        "Num_Credit_Card",
        "Credit_Cards_clean",
        "Credit_Cards_imputed",
        "Credit_Cards_Imputed_Bin",
        "Credit_Cards_Was_Invalid",
        "Credit_Cards_imputation_method"
    ).show(20, truncate=False)


    # ============================================================
    # 1. Debt per Credit Account
    # ============================================================
    train_processed_1 = train_processed.copy()
    unseen_processed_1 = unseen_processed.copy()

    train_processed_1 = train_processed_1.withColumn(
        "Debt_per_Credit_Account",
        F.when(
            F.col("Bank_Accounts_clean") + F.col("Credit_Cards_clean") != 0,
            F.col("Outstanding_Debt") /
            (
                F.col("Bank_Accounts_clean")
                +
                F.col("Credit_Cards_clean")
            )
        ).otherwise(None)
    )

    unseen_processed_1 = unseen_processed_1.withColumn(
            "Debt_per_Credit_Account",
            F.when(
                F.col("Bank_Accounts_clean") + F.col("Credit_Cards_clean") != 0,
                F.col("Outstanding_Debt") /
                (
                    F.col("Bank_Accounts_clean")
                    +
                    F.col("Credit_Cards_clean")
                )
            ).otherwise(None)
        )
    


    # ============================================================
    # 2. Delayed Payments per Credit Card
    # ============================================================

    train_processed_1 = train_processed_1.withColumn(
        "Delayed_Payments_per_Card",
        F.when(
            F.col("Credit_Cards_clean") != 0,
            F.col("Num_of_Delayed_Payment") /
            F.col("Credit_Cards_clean")
        ).otherwise(None)
    )

    unseen_processed_1 = unseen_processed_1.withColumn(
            "Delayed_Payments_per_Card",
            F.when(
                F.col("Credit_Cards_clean") != 0,
                F.col("Num_of_Delayed_Payment") /
                F.col("Credit_Cards_clean")
            ).otherwise(None)
        )


    # ============================================================
    # 3. Interest Rate × Debt
    # ============================================================

    train_processed_1 = train_processed_1.withColumn(
        "InterestRate_x_Debt",
        F.col("Interest_Rate") *
        F.col("Outstanding_Debt")
    )

    unseen_processed_1 = unseen_processed_1.withColumn(
            "InterestRate_x_Debt",
            F.col("Interest_Rate") *
            F.col("Outstanding_Debt")
        )

    features_df = train_processed_1.unionByName(unseen_processed_1)

    features_main_name = "gold_feature_store_main_"+ '.parquet'
    features_filepath = gold_features_main_directory + features_main_name
    features_df.write.mode("overwrite").parquet(features_filepath)

    for date_str in dates_str_lst:
        snapshot_date_str = datetime.strptime(date_str, "%Y-%m-%d")
        df = features_df.filter(col('labels_snapshot_date') == snapshot_date_str)
        # save bronze table to datamart - IRL connect to database to write
        gold_feature_partition_filename = os.path.splitext(os.path.basename(features_main_name))[0] + "_" + date_str.replace('-','_') + '.parquet'
        gold_feature_partition_filepath = gold_features_main_directory + gold_feature_partition_filename
        df.write.mode("overwrite").parquet(gold_feature_partition_filepath)
        print('saved to:', gold_feature_partition_filepath)
    



    # save silver table - IRL connect to database to write
    # silver_train_processed_name = "silver_train_processed_"+ '.parquet'
    # silver_unseen_processed_name = "silver_unseen_processed_"+ '.parquet'
    # silver_train_processed_filepath = silver_merged_processed_directory + silver_train_processed_name
    # silver_unseen_processed_filepath = silver_merged_processed_directory + silver_unseen_processed_name         
    # train_processed.write.mode("overwrite").parquet(silver_train_processed_filepath)
    # print('saved to:', silver_train_processed_filepath)
    # unseen_processed.write.mode("overwrite").parquet(silver_unseen_processed_filepath)
    # print('saved to:', silver_unseen_processed_filepath)            