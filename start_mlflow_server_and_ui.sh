export MLFLOW_TRACKING_URI=127.0.0.1:8899
export MLFLOW_BACKEND_STORE_URI=/data/digiandomenico/vessel_proj/data/mlflow/backend/
export MLFLOW_ARTIFACT_ROOT=/data/digiandomenico/vessel_proj/data/mlflow/artifact/

mlflow ui --port 5001 &

mlflow server \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --default-artifact-root "$MLFLOW_ARTIFACT_ROOT" \
    --host 0.0.0.0 \
    --port 8899 &
    