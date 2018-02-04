set -e
set -x

export BUCKET_NAME=$1
export STAMP="$(date +%Y%m%d_%H%M%S)"
export JOB_NAME="statoil_${STAMP}"
export JOB_DIR="gs://${BUCKET_NAME}/statoil/${STAMP}"
export REGION=europe-west1

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR \
    --runtime-version 1.4 \
    --module-name trainer.nn \
    --package-path ./trainer \
    --region $REGION \
    --scale-tier BASIC_GPU \
    -- \
    --train-file gs://$BUCKET_NAME/data/train.json \
    --test-file gs://$BUCKET_NAME/data/test.json
