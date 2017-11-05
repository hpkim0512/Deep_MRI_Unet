import pprint
import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

import h5py

pp = pprint.PrettyPrinter()


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def read_mat(dataset, is_train):
    """ Import the Version 7.3 MAT-file.
    Yon can save your dataset as Version 7.3 MAT-file in matlab by running following syntax:
    "save(filename, 'input', 'label', '-v7.3')"
    Both of 'input' and 'label' must have the shape as [width, height, num_of_data].

    Args:
      dataset: Path of matlab file.
      is_train: True for training, False for test.
    Returns:
      trX: Numpy arrays of input of Network.
      trY: Numpy arrays of label of Network."""

    with tf.device('/cpu:0'):
        arrays = {}
        f = h5py.File(dataset)

        for k, v in f.items():
            arrays[k] = np.transpose(np.array(np.float32(v)), (0, 2, 1))
        trX = arrays['input']
        if is_train:
            trY = arrays['label']

    if is_train:
        return trX, trY
    else:
        return trX
