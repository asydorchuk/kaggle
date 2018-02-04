# Statoil/C-CORE Iceberg Classifier Challenge

## Description

Ship or iceberg, can you decide from space?

https://www.kaggle.com/c/statoil-iceberg-classifier-challenge

## Authors
* Andrii Sydorchuk (sydorchuk.andriy * gmail.com)
* Kirill Zhdanovich (kzhdanovich * gmail.com)

## How-to

1. Train single NN (run multiple times to train a few NNs):
  * Locally:
    ```
      ./local.sh
    ```
  * Google CloudML:
    ```
      ./cloudml.sh $YOUR_GCS_BUCKET
    ```
2. Generate features:
  ```
  python trainer/features.py --train-file data/train.json --test-file data/test.json --nn-preds-dir=precalc --features-file=features.csv
  ```
3. Use lgbm to generate final predictions:
  ```
  python trainer/lgbm.py --features-file=features.csv --preds-file=preds.csv
  ```
