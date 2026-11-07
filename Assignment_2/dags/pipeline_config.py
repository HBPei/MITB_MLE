"""Settings shared by the training and the inference pipeline (imported by both files in this folder)."""

# The development model: trained once on ALL labelled history, with the notebook's
# train / validation / test / OOT windows counted back from this date.
MODEL_TRAIN_DATE = "2025-06-01"

# "production": the champion goes live on MODEL_TRAIN_DATE and the inference pipeline only scores
#               applications from that month on: new data the model has never seen. Run
#               scripts/generate_synthetic_data.py first to simulate those new months (data/incoming/).
# "backtest":   the champion is made effective from 2023-01-01 and the inference pipeline replays every
#               historical month, to demonstrate scoring, monitoring and the dashboard on the data we have.
#               Months whose labels were in the training window are in-sample: read them as a backtest.
DEPLOYMENT_MODE = "production"

if DEPLOYMENT_MODE == "production":
    CHAMPION_EFFECTIVE_FROM = MODEL_TRAIN_DATE
    INFERENCE_START = MODEL_TRAIN_DATE
else:
    CHAMPION_EFFECTIVE_FROM = "2023-01-01"
    INFERENCE_START = "2023-01-01"

SCRIPTS = 'cd /opt/airflow/scripts && '
FORCE = ' --force "{{ var.value.get(\'rebuild_data\', \'false\') }}"'
XGB_N_ITER = ' --xgb_n_iter "{{ var.value.get(\'xgb_n_iter\', 30) }}"'

TRAINING_PIPELINE = "credit_model_training"
REGISTRY_FILE = "/opt/airflow/scripts/model_bank/model_registry.json"
