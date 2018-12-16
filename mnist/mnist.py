"""
MNIST training with keras.
This is an introduction keras sample code taken from
Keras text
"""

import argparse
import os
from time import gmtime, strftime
from keras.callbacks import TensorBoard

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers
from keras.utils import np_utils


NUM_DIGITS = 10
FLATTEN_SIZE = 28 * 28
VALIDATION_SPLIT = 0.2


def select_optimizer(args):
    if args.optimizer.lower() == 'sgd':
        optimizer = optimizers.SGD()
    elif args.optimizer.lower() == 'adam':
        optimizer = optimizers.Adam()
    return optimizer


def make_tensorboard(root_dir='log'):
    tictoc = strftime('%a_%d_%b_%Y_%H_%M_%S', gmtime())
    log_dir = os.path.join(root_dir, tictoc)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tensor_board = TensorBoard(log_dir=log_dir)
    return tensor_board


def define_model(args):
    model = Sequential()
    model.add(Dense(args.hidden, input_shape=(FLATTEN_SIZE,)))
    model.add(Activation('relu'))
    model.add(Dropout(args.dropout))
    model.add(Dense(args.hidden))
    model.add(Activation('relu'))
    model.add(Dropout(args.dropout))
    model.add(Dense(NUM_DIGITS))
    model.add(Activation('softmax'))
    model.summary()
    return model


def train(args):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, FLATTEN_SIZE).astype(np.float32)
    X_train /= 255.
    X_test = X_test.reshape(-1, FLATTEN_SIZE).astype(np.float32)
    X_test /= 255.
    y_train = np_utils.to_categorical(y_train, NUM_DIGITS)
    y_test = np_utils.to_categorical(y_test, NUM_DIGITS)
    model = define_model(args)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=select_optimizer(args),
        metrics=['accuracy']
    )

    callbacks = [make_tensorboard()]

    model.fit(
        X_train,
        y_train,
        batch_size=args.batchsize,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
        validation_split=VALIDATION_SPLIT
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-E', '--epochs', type=int, default=10)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--hidden', type=int, default=128, help='output size of hidden layer')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--optimizer', type=str, default='sgd')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    np.random.seed(args.seed)
    train(args)
if __name__ == '__main__':
    main()
