import argparse
from glob import glob
from itertools import chain, combinations

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsRegressor


def load(stream, is_train):
    df = pd.read_json(stream, dtype={'inc_angle': 'str'}).drop(columns=['band_1', 'band_2']).set_index('id')
    df['inc_angle'] = df.inc_angle.replace('na', '0.0')
    df['fake'] = df.inc_angle.apply(lambda x: len(x.split('.')[1]) > 5)
    df['inc_angle'] = df.inc_angle.astype('float64')
    df['train'] = is_train
    return df

def add_nn_preds(both, nn_preds_dir):
    train_nn_preds = np.concatenate([np.load(fname) for fname in glob(nn_preds_dir + '/train_preds_*')])
    test_nn_preds = np.concatenate([np.load(fname) for fname in glob(nn_preds_dir + '/test_preds_*')])

    # Find best subset of NNs.
    assert len(train_nn_preds) < 30, "Too many NNs to find optimal subset with bruteforce"
    seq = range(len(train_nn_preds))
    best_subset = seq
    for subset in chain.from_iterable(combinations(seq, n + 1) for n in seq):
        cand_loss = log_loss(both[both.train].is_iceberg, np.mean(train_nn_preds[subset, :], axis=0))
        best_loss = log_loss(both[both.train].is_iceberg, np.mean(train_nn_preds[best_subset, :], axis=0))
        if cand_loss < best_loss:
            best_subset = subset

    # Calculate mean prediction of the best NN subset.
    both.loc[both.train, 'nn_pred'] = np.mean(train_nn_preds[best_subset, :], axis=0)
    both.loc[~both.train, 'nn_pred'] = np.mean(test_nn_preds[best_subset, :], axis=0)
    both.loc[~both.train, 'is_iceberg'] = np.mean(test_nn_preds[best_subset, :], axis=0)
    print('Best NN subset: {}'.format(best_subset))
    print('Best NN subset CV loss: {}'.format(log_loss(both[both.train].is_iceberg, both[both.train].nn_pred)))
    return both

def group_stats(df, suffix=''):
    best = df.is_iceberg.values
    pred = df.nn_pred.values
    med = np.zeros(len(df))
    mea = np.zeros(len(df))
    cnt = np.zeros(len(df))
    for idx in range(len(df)):
        # Replace true label of a sample with prediction.
        best[idx], pred[idx] = pred[idx], best[idx]
        med[idx] = np.median(best)
        mea[idx] = np.mean(best)
        cnt[idx] = len(best)
        # Swap true label back.
        best[idx], pred[idx] = pred[idx], best[idx]
    df['med' + suffix] = med
    df['mea' + suffix] = mea
    df['cnt' + suffix] = cnt
    return df

def add_knn_preds(both):
    mj_jersey_number = 23
    for key in both.index:
        # Exclude prediction sample and fake data from the training.
        mask = (~both.fake) & (both.index != key)
        cl = KNeighborsRegressor(n_neighbors=mj_jersey_number, weights='distance', algorithm='brute')
        cl.fit(both[mask].inc_angle.values.reshape((-1,1)), both[mask].nn_pred.values)
        both.loc[key, 'knn_pred'] = cl.predict(both.loc[key, 'inc_angle'])
    print('KNN CV loss: {}'.format(log_loss(both[both.train].is_iceberg, both[both.train].knn_pred)))
    return both

def main(train_file, test_file, nn_preds_dir, features_file, **args):
    # Load train & test data.
    both = pd.concat([
        load(train_file, True),
        load(test_file, False),
    ])

    # Add nn predictions.
    both = add_nn_preds(both, nn_preds_dir)

    # Calculate stats for the groups of objects with the same angle.
    both = both.groupby('inc_angle').apply(lambda x: group_stats(x))

    # Use KNN regressor to extract additional information for objects with unique angle.
    both = add_knn_preds(both)

    # Save output.
    both.to_csv(features_file, header=True)


if __name__ == '__main__':
    # Parse the input arguments for common Cloud ML Engine options
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--train-file',
      help='Cloud Storage bucket or local path to training data')
    parser.add_argument(
      '--test-file',
      help='Cloud Storage bucket or local path to test data')
    parser.add_argument(
      '--nn-preds-dir',
      help='Directory with predictions from neural networks')
    parser.add_argument(
      '--features-file',
      help='File path to save features')
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)
