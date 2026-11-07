from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
import json

from pipeline_config import INFERENCE_START, SCRIPTS, FORCE, TRAINING_PIPELINE, REGISTRY_FILE

# INFERENCE PIPELINE - one run per month ({{ ds }} = the application month), in order (catchup).
#   new_applications : this month's attributes + financials -> bronze -> silver -> inference features
#                      (training-time imputation parameters APPLIED, never re-fitted; no label needed)
#   label_refresh    : rebuild the label store, so labels that matured since last month reach monitoring
#   model_inference  : champion scores (production) + challenger scores (shadow) -> gold predictions table
#   model_monitoring : performance (once labels mature) + stability -> gold monitoring tables -> dashboard PNG
#   governance       : refresh decision per GOVERNANCE.md; "train" automatically triggers credit_model_training
#                      (model_train_date = this month, new champion effective next month).
# wait_for_champion holds each month until a champion exists AND no training run is in progress, so a month is
# always scored with the model that is in force for it.

DS = ' --snapshotdate "{{ ds }}"'
NEW_DATA_TABLES = {'attributes': 'features_attributes', 'financials': 'features_financials'}


def step(task_id, args):
    return BashOperator(task_id=task_id, bash_command=SCRIPTS + 'python3 data_pipeline.py ' + args + DS)


def main_py(task_id, args, **kw):
    return BashOperator(task_id=task_id, bash_command=SCRIPTS + 'python3 main.py ' + args + DS, **kw)


def champion_ready():
    from airflow.models import DagRun
    from airflow.utils.state import DagRunState
    busy = [r for st in (DagRunState.QUEUED, DagRunState.RUNNING)
            for r in DagRun.find(dag_id=TRAINING_PIPELINE, state=st)]
    if busy:
        print(f'waiting: {TRAINING_PIPELINE} run in progress ({[r.run_id for r in busy]})')
        return False
    try:
        with open(REGISTRY_FILE) as f:
            champion = json.load(f).get('champion')
    except FileNotFoundError:
        champion = None
    print(f'champion: {champion}')
    return bool(champion)


def read_decision(ti, **_):
    raw = ti.xcom_pull(task_ids='governance.refresh_decision') or ''
    decision = json.loads(raw.split('DECISION=', 1)[1]) if 'DECISION=' in raw else {'decision': 'skip'}
    print(f'refresh decision: {decision}')
    return 'governance.retrain_recommended' if decision['decision'] == 'train' else 'governance.model_healthy'


with DAG(
    'credit_model_inference',
    description='monthly: score new applications with the champion, monitor, decide on refresh',
    schedule_interval='0 0 1 * *',          # 00:00 on the 1st of every month
    start_date=datetime.strptime(INFERENCE_START, '%Y-%m-%d'),
    catchup=True,                            # missed months are processed in order
    max_active_runs=1,
    default_args={'owner': 'airflow', 'depends_on_past': True, 'retries': 1, 'retry_delay': timedelta(minutes=5)},
) as dag:

    wait_for_champion = PythonSensor(        # frees the worker between checks, so training can run meanwhile
        task_id='wait_for_champion',
        python_callable=champion_ready,
        mode='reschedule',
        poke_interval=120,
        timeout=12 * 3600,
    )

    with TaskGroup('new_applications', tooltip='this month only -> bronze -> silver -> inference features') as new_apps:
        build_features = step('build_inference_features', '--step inference_features' + FORCE)
        for short, table in NEW_DATA_TABLES.items():
            (step(f'bronze_{short}', f'--step bronze_month --table {table}' + FORCE)
             >> step(f'silver_{short}', f'--step silver_month --table {table}' + FORCE)
             >> build_features)

    with TaskGroup('label_refresh', tooltip='matured labels -> label store (for monitoring)') as label_refresh:
        (step('bronze_lms_loan_daily', '--step bronze --table lms_loan_daily --force true')
         >> step('silver_lms_loan_daily', '--step silver --table lms_loan_daily --force true')
         >> step('gold_label_store', '--step gold_label_store --force true'))

    with TaskGroup('model_inference', tooltip='champion = production, challenger = shadow') as model_inference:
        main_py('score_champion', '--mode inference --role champion')
        main_py('score_challenger_shadow', '--mode inference --role challenger')

    with TaskGroup('model_monitoring', tooltip='performance + stability -> gold tables -> dashboard') as monitoring:
        [main_py('monitor_champion', '--mode monitor --role champion'),
         main_py('monitor_challenger', '--mode monitor --role challenger')] >> main_py('build_dashboard',
                                                                                    '--mode report')

    with TaskGroup('governance', tooltip='refresh policy (GOVERNANCE.md)') as governance:
        decision = main_py('refresh_decision', '--mode decide', do_xcom_push=True)
        branch = BranchPythonOperator(task_id='read_decision', python_callable=read_decision)
        retrain = TriggerDagRunOperator(     # automatic refresh: train as of this month, live from next month
            task_id='retrain_recommended',
            trigger_dag_id=TRAINING_PIPELINE,
            conf={'model_train_date': '{{ ds }}',
                  'effective_from': "{{ data_interval_end.strftime('%Y-%m-%d') }}"},
            wait_for_completion=False,       # the next month's wait_for_champion waits for it instead
        )
        decision >> branch >> [retrain, EmptyOperator(task_id='model_healthy')]

    wait_for_champion >> [new_apps, label_refresh]
    new_apps >> model_inference >> monitoring
    label_refresh >> monitoring
    monitoring >> governance
