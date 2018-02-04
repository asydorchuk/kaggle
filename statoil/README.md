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
## Solution overview

 1. Train 5 NNs on image data (excluding incidence angle). Each model is using CV5 with the same split. NN tuning was done in the following areas: 1) NN architecture; 2) best augmentation parameters; 3) best CV5 seed that gives similar scores on all folds.
 2. Take subset of NNs, with mean prediction that gives the best CV score on train data.
 3. Group mean predictions from the previous step by incidence angle (train and test data together). For each group calculate the following features: mean prediction, median prediction, total number of samples in a group. During this calculation original train labels can be used to boost precision even further, as long as they don't propagate to the sample predictions itself.
 4. Run KNN regressor on predictions from step 2) and incidence angle.
 5. Train LightGBM model on the features from the steps 2), 3) and 4) (only 5 features in total).
 6. Fine tune LightGBM using CV on train data.
 7. Clip final probabilities towards interval (0.001, 0.999), mainly as a precaution.

Additional notes:

 - NN architecture used is from this paper: https://arxiv.org/pdf/1612.00983.pdf
 - It's important to skip incidence angle while training NNs in step 1), so that they can prioritize image features.
 - It's important to calculate group features in step 3) on both train + test data, to avoid overfitting towards train data distribution.
 - it's important to do a proper CV at every step any parameters are tuned in the algorithm.
 - the fact that images with similar incident angle have similar object class on them is exploited by LightGBM in step 6), based on features in step 3).
