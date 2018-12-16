"""
Train MNIST using LeNet a kind of CNN
"""

import os
import random

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import numpy as np


def define_lenet(input_shape, num_classes):
    model = Sequential()
    model.add(
        Conv2D(
            filters=20,
            input_shape=input_shape,
            kernel_size=5,
            padding='same',
            activation='relu',
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(
        Conv2D(
            filters=50,
            kernel_size=5,
            padding='same',
            activation='relu',
        )
    )
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.summary()
    return model


class MNISTDataset():

    def __init__(self):
        self.image_shape = (28, 28, 1)
        self.num_classes = 10

    def preprocess(self, data):
        data = data.astype('float32')
        data /= 255.
        shape = (-1,) + self.image_shape
        data = data.reshape(shape)
        return data

    def to_categorical(self, labels):
        return tensorflow.keras.utils.to_categorical(labels, self.num_classes)

    def load_data(self, preprocess=True):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        if preprocess:
            x_train = self.preprocess(x_train)
            x_test = self.preprocess(x_test)
        y_train = self.to_categorical(y_train)
        y_test = self.to_categorical(y_test)
        return x_train, y_train, x_test, y_test


class Trainer():

    def __init__(self, model, loss, optimizer, log_dir='.'):
        self.model = model
        self.model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=['accuracy']
        )
        self.verbose = 1
        self.log_dir = os.path.join(log_dir, 'log')

    def run(self, x_train, y_train, batch_size, epochs, validation_split):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[TensorBoard(log_dir=self.log_dir)],
            verbose=self.verbose,
        )


def main():
    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
    dataset = MNISTDataset()
    x_train, y_train, x_test, y_test = dataset.load_data(preprocess=True)

    model = define_lenet(
        input_shape=(dataset.image_shape),
        num_classes=10
    )
    model.summary()

    trainer = Trainer(
        model,
        loss='categorical_crossentropy',
        optimizer=Adam()
    )
    # training
    trainer.run(x_train, y_train, batch_size=128, epochs=12, validation_split=0.2)
    # evaluate
    score = model.evalate(x_test, y_test, verbose=0)
    print('loss', score[0])  # loss
    print('accuracy', score[1])  # acc

if __name__ == '__main__':
    main()
