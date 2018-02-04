import argparse

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, KFold

import lightgbm


def lgbm_cv(df):
    X = df[df.train][['nn_pred', 'med', 'mea', 'cnt', 'knn_pred']].values
    Y = df[df.train][['is_iceberg']].values.squeeze()
    Z = df[~df.train][['nn_pred', 'med', 'mea', 'cnt', 'knn_pred']]

    # CV for LightGBM.
    kf = StratifiedKFold(n_splits=51, shuffle=True, random_state=27)
    cv = np.zeros(len(X))
    for fold_id, (train_idx, test_idx) in enumerate(kf.split(X, Y)):
        cl = lightgbm.LGBMClassifier(max_depth=3, n_estimators=70, learning_rate=0.1, min_child_samples=40)
        cl.fit(
            X[train_idx], Y[train_idx],
            eval_set=[(X[test_idx], Y[test_idx])],
            eval_metric='logloss', verbose=False
        )
        cv[test_idx] = cl.predict_proba(X[test_idx])[:, 1]
    print('LightGBM CV loss: {}'.format(log_loss(df[df.train].is_iceberg, np.clip(cv, 0.001, 0.999))))

def lgbm(df):
    X = df[df.train][['nn_pred', 'med', 'mea', 'cnt', 'knn_pred']].values
    Y = df[df.train][['is_iceberg']].values.squeeze()
    Z = df[~df.train][['nn_pred', 'med', 'mea', 'cnt', 'knn_pred']]

    cl = lightgbm.LGBMClassifier(max_depth=3, n_estimators=70, learning_rate=0.1, min_child_samples=40)
    cl.fit(X, Y)
    return np.clip(cl.predict_proba(Z)[:,1], 0.001, 0.999)

def main(features_file, preds_file, **args):
    # Load features.
    features = pd.read_csv(features_file, index_col='id')

    # LightGBM CV.
    lgbm_cv(features)

    # LightGBM final.
    predictions = lgbm(features)

    # Save output
    pd.DataFrame(
        data=predictions, index=features[~features.train].index, columns=['is_iceberg']
    ).to_csv(preds_file, header=True)


if __name__ == '__main__':
    # Parse the input arguments for common Cloud ML Engine options
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--features-file',
      help='File with input features')
    parser.add_argument(
      '--preds-file',
      help='File path to save predictions')
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)
