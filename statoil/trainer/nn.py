from __future__ import print_function

import argparse
import pandas as pd
import numpy as np

from keras import layers
from keras import models
from keras import optimizers
from keras import callbacks
from keras.preprocessing import image

from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

# Best seed: 888009397.
# Best seed was taken after running around 10 NN models.
# And picking the one with the best CV score.
SEED=np.random.randint(888009397)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.set_random_seed(SEED)
from tensorflow.python.lib.io import file_io


def merge_bands(band1, band2):
    b1 = np.array(band1).astype(np.float32)
    b2 = np.array(band2).astype(np.float32)
    return [np.stack([b1, b2], -1).reshape(75, 75, 2)]

def load(train_file):
    with file_io.FileIO(train_file, mode='r') as stream:
        df = pd.read_json(stream).set_index('id')
        if 'bands' not in df.columns:
            df['bands'] = df.apply(lambda row: merge_bands(row['band_1'], row['band_2']), axis=1)
            df = df.drop(['band_1', 'band_2'], axis=1)
        return df

def get_callbacks(filepath, patience=10):
    es = callbacks.EarlyStopping('val_loss', patience=patience, mode="min")
    msave = callbacks.ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]

def get_model_vgg9():
    input1 = layers.Input(shape=(75, 75, 2), name='Data1')

    db1 = layers.BatchNormalization(momentum=0.0)(input1)
    db1 = layers.Conv2D(32, (7,7), activation='relu', padding='same')(db1)
    db1 = layers.MaxPooling2D((2, 2))(db1)
    db1 = layers.Dropout(0.2)(db1)
    
    db2 = layers.Conv2D(64, (5,5), activation='relu', padding='same')(db1)
    db2 = layers.MaxPooling2D((2, 2))(db2)
    db2 = layers.Dropout(0.2)(db2)
    
    db3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(db2)
    db3 = layers.MaxPooling2D((2, 2))(db3)
    db3 = layers.Dropout(0.2)(db3)
    db3 = layers.Flatten()(db3)

    fb1 = layers.Dense(128, activation='relu')(db3)
    fb1 = layers.Dropout(0.5)(fb1)
    output = layers.Dense(1, activation='sigmoid')(fb1)
    
    model = models.Model(inputs=[input1], outputs=[output])
    optimizer = optimizers.Adam(lr=1e-4)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
    return model

def train_generator(X, Y):
    batch_size=64
    print ("batch size {}".format(batch_size))
    # All of the below parameters were important to impove performance of the model.
    # Specifically shift range of 0.12 works the best. This is empirical observation
    # and most likely is somehow related to the fact that 0.12 * 75 = 9.0 (pixels).
    rotation_range=10.0
    shift_range=0.12
    vertical_flip=True
    print('rotation_range: {}; shift_range: {}; vertical_flip: {}'.format(rotation_range, shift_range, vertical_flip))
    base_generator = image.ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=shift_range,
        height_shift_range=shift_range,
        vertical_flip=vertical_flip,
    )
    gen = base_generator.flow(X, Y, batch_size=batch_size, seed=SEED)
    while True:
        yield gen.next()

def print_history(h):
    hloss = zip(h.history['loss'], h.history['val_loss'])
    for idx, loss in enumerate(hloss):
        print('Step {} loss: {}, val_loss: {}'.format(idx, loss[0], loss[1]))

def save_preds(directory, filename, arr):
    opath = directory + '/' + filename
    with file_io.FileIO(opath, 'w+') as output_f:
        np.save(output_f, arr)

def train_model(train_file, test_file, job_dir, **args):
    print("SEED %d" % SEED)

    # Train NNs using CV.
    train = load(train_file)
    train_bands = np.stack(train.bands).squeeze()
    labels = train.is_iceberg.values
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold_id, (train_idx, val_idx) in enumerate(kf.split(train_bands, labels)):
        X_train, Y_train = train_bands[train_idx], labels[train_idx]
        X_val, Y_val = train_bands[val_idx], labels[val_idx]
        model = get_model_vgg9()
        history = model.fit_generator(
            train_generator(X_train, Y_train),
            steps_per_epoch=256,
            epochs=100,
            validation_data=(X_val, Y_val),
            callbacks=get_callbacks(filepath='model_weights_{}.hdf5'.format(fold_id), patience=10),
            verbose=0,
        )
        print_history(history)

    # Generate predictions for train and test data using best iteration of CV NNs.
    test = load(test_file)
    test_bands = np.stack(test.bands).squeeze()
    cv_train_preds, cv_test_preds, cv_scores = np.zeros(len(train_bands)), [], []
    for fold_id, (train_idx, val_idx) in enumerate(kf.split(train_bands, labels)):
        m = models.load_model('model_weights_{}.hdf5'.format(fold_id))

        # Take average of predicted probabilities on original and flipped images.
        X_val, Y_val = train_bands[val_idx], labels[val_idx]
        val_orig_predictions = m.predict(X_val, batch_size=64).squeeze()
        val_flip_predictions = m.predict(np.flip(X_val, axis=1), batch_size=64).squeeze()
        cv_train_preds[val_idx] = (val_orig_predictions + val_flip_predictions) * 0.5
        cv_scores.append(log_loss(Y_val, cv_train_preds[val_idx]))

        # Take average of predicted probabilities on original and flipped images.
        test_orig_predictions = m.predict(test_bands).squeeze()
        test_flip_predictions = m.predict(np.flip(test_bands, axis=1)).squeeze()
        cv_test_preds.append((test_orig_predictions + test_flip_predictions) * 0.5)
    # Take mean of test predictions generated by each of the CV models.
    cv_test_preds = np.mean(cv_test_preds, axis=0)

    train_preds_file = 'train_preds_{}'.format(SEED)
    save_preds(job_dir, train_preds_file, cv_train_preds)

    test_preds_file = 'test_preds_{}'.format(SEED)
    save_preds(job_dir, test_preds_file, cv_test_preds)

    print('Best validation scores for each fold: {}'.format(cv_scores))
    print('CV log loss: {}'.format(log_loss(labels, cv_train_preds)))


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
      '--job-dir',
      help='Cloud storage bucket to export the model and store temp files')
    args = parser.parse_args()
    arguments = args.__dict__
    train_model(**arguments)
