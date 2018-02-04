set -e
set -x

mkdir -p tmp

gcloud ml-engine local train \
    --job-dir tmp \
    --module-name trainer.nn \
    --package-path ./trainer \
    -- \
    --train-file data/train.json \
    --test-file data/test.json
