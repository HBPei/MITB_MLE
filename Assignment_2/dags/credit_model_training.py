from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta

from pipeline_config import MODEL_TRAIN_DATE, CHAMPION_EFFECTIVE_FROM, SCRIPTS, XGB_N_ITER

FORCE = ' --force true'   # a training run always rebuilds the stores from the latest full history

# TRAINING PIPELINE - fully automatic:
#   * runs once by itself when the DAG is unpaused ('@once'): builds the stores from all history and deploys
#     the development model (model_train_date = MODEL_TRAIN_DATE)
#   * is triggered again by credit_model_inference when its refresh decision says "train" (scheduled refresh or
#     monitoring alert), with model_train_date = that month and effective_from = the next month.
#
#   label_store   : lms_loan_daily -> check -> bronze -> silver -> gold label store
#   feature_store : attributes / financials / clickstream, each check -> bronze -> silver -> one gold feature
#                   store (train/validation/test/OOT windows from model_train_date; saves the fitted parameters)
#   model_training: train LogReg + train XGBoost (gold-store splits) -> select champion (gates + champion check)

TRAIN = '{{ params.model_train_date }}'
EFFECTIVE = '{{ params.effective_from }}'
DS = ' --snapshotdate "{{ ds }}"'

FEATURE_SOURCES = {'attributes': 'features_attributes', 'financials': 'features_financials',
                   'clickstream': 'feature_clickstream'}


def step(task_id, args):
    return BashOperator(task_id=task_id, bash_command=SCRIPTS + 'python3 data_pipeline.py ' + args + DS)


with DAG(
    'credit_model_training',
    description='build the stores from all history, train LogReg + XGBoost, register the champion',
    schedule_interval='@once',               # first run on unpause; later runs triggered by the inference DAG
    start_date=datetime(2025, 6, 1),
    catchup=False,
    max_active_runs=1,
    default_args={'owner': 'airflow', 'retries': 1, 'retry_delay': timedelta(minutes=5)},
    params={
        'model_train_date': Param(MODEL_TRAIN_DATE, type='string',
                                  description='train date: OOT = 2 months before it, test 2 before that, ...'),
        'effective_from': Param(CHAMPION_EFFECTIVE_FROM, type='string',
                                description='first month the new champion scores'),
    },
) as dag:

    with TaskGroup('label_store', tooltip='lms_loan_daily -> bronze -> silver -> gold label store') as label_store:
        (step('check_source_lms_loan_daily', '--step dep_check --table lms_loan_daily')
         >> step('bronze_lms_loan_daily', '--step bronze --table lms_loan_daily' + FORCE)
         >> step('silver_lms_loan_daily', '--step silver --table lms_loan_daily' + FORCE)
         >> step('gold_label_store', '--step gold_label_store' + FORCE))

    with TaskGroup('feature_store', tooltip='3 sources -> bronze -> silver -> one gold feature store') as feature_store:
        gold_feature_store = step('gold_feature_store',
                                  '--step gold_feature_store --model_train_date ' + TRAIN + FORCE)
        for short, table in FEATURE_SOURCES.items():
            (step(f'check_source_{short}', f'--step dep_check --table {table}')
             >> step(f'bronze_{short}', f'--step bronze --table {table}' + FORCE)
             >> step(f'silver_{short}', f'--step silver --table {table}' + FORCE)
             >> gold_feature_store)

    label_store >> gold_feature_store        # gold features are built on the labelled loans

    with TaskGroup('model_training', tooltip='train both models, then the promotion gate') as model_training:
        train = ' --split_source store --snapshotdate ' + TRAIN
        train_logreg = BashOperator(
            task_id='train_logreg',
            bash_command=SCRIPTS + 'python3 main.py --mode train --algo logreg' + train,
        )
        train_xgboost = BashOperator(
            task_id='train_xgboost',
            bash_command=SCRIPTS + 'python3 main.py --mode train --algo xgb' + train + XGB_N_ITER,
            execution_timeout=timedelta(hours=2),
        )
        select_champion = BashOperator(          # quality gates + beat the current champion -> model registry
            task_id='select_champion',
            bash_command=SCRIPTS + 'python3 main.py --mode select --trigger manual' + train
                         + ' --effective_from ' + EFFECTIVE,
        )
        [train_logreg, train_xgboost] >> select_champion

    feature_store >> model_training
