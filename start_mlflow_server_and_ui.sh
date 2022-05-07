set -o allexport
source vessel_proj.env
set +o allexport

mlflow ui --port 5001 &

mlflow server \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --default-artifact-root "$MLFLOW_ARTIFACT_ROOT" \
    --host 0.0.0.0 \
    --port 8899 &
    