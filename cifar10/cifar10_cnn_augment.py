import os
import random

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten, Dropout
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np


def define_classifier(input_shape, num_classes):
    model = Sequential()
    model.add(
        Conv2D(
            32,
            kernel_size=3,
            input_shape=input_shape,
            padding='same',
            activation='relu',
        )
    )
    model.add(Dropout(0.25))
    model.add(
        Conv2D(
            64,
            kernel_size=3,
            padding='same',
            activation='relu'
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(
        Conv2D(
            64,
            kernel_size=3,
            padding='same',
            activation='relu',
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model


def split_dataset_random(x_train, y_train, validation_split):
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    validation_size = int(x_train.shape[0] * validation_split)
    x_train, x_val = x_train[indices[:-validation_size], :], x_train[indices[-validation_size:], :]
    y_train, y_val = y_train[indices[:-validation_size], :], y_train[indices[-validation_size:], :]
    return (x_train, y_train), (x_val, y_val)


class CIFAR10Dataset():

    def __init__(self):
        self.image_shape = (32, 32, 3)
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
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
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
            metrics=['accuracy'],
        )
        self.save_model_name = 'bestmodel.hdf5'
        self.verbose = 1
        self.log_dir = os.path.join(log_dir, 'log')

    def run(self, x_train, y_train, batch_size, epochs, validation_split):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        image_generator = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False,
        )

        (x_train, y_train), (x_val, y_val) = split_dataset_random(
            x_train, y_train, validation_split=validation_split)
        self.model.fit_generator(
            image_generator.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=x_train.shape[0] // batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=[
                TensorBoard(log_dir=self.log_dir),
                ModelCheckpoint(
                    os.path.join(self.log_dir, self.save_model_name),
                    save_best_only=True,
                )
            ],
            verbose=self.verbose,
            workers=4,
        )


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def main():
    set_random_seed(seed=12345)
    dataset = CIFAR10Dataset()
    x_train, y_train, x_test, y_test = dataset.load_data(preprocess=True)

    model = define_classifier(
        input_shape=(dataset.image_shape),
        num_classes=10
    )
    model.summary()

    trainer = Trainer(
        model,
        loss='categorical_crossentropy',
        optimizer=RMSprop()
    )
    # training
    trainer.run(x_train, y_train, batch_size=128, epochs=15, validation_split=0.2)
    # evaluate
    score = model.evalate(x_test, y_test, verbose=0)
    print('loss', score[0])  # loss
    print('accuracy', score[1])  # acc

if __name__ == '__main__':
    main()
