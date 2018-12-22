"""
train dcgan which generate numbers like MNIST dataset.

We highly depend on `keras-adversarial` library

git clone https://github.com/bstriner/keras-adversarial
cd keras_adversarial
python setup.py install
note that this script is version sensitive with respect to keras
pip install keras==2.1.2

recently https://github.com/ussaema/Vector_Matrix_CapsuleGAN is released.
"""
import argparse
import os

import keras
from keras.layers import Dense, Conv2D, BatchNormalization, UpSampling2D
from keras.layers import Dropout, LeakyReLU, Flatten, Activation
from keras.layers import Input, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import L1L2
from keras.callbacks import TensorBoard

import keras_adversarial
from keras_adversarial.image_grid_callback import ImageGridCallback
from keras_adversarial import gan_targets, simple_gan, normal_latent_sampling
from keras_adversarial import AdversarialModel, AdversarialOptimizerSimultaneous
import numpy as np
import pandas as pd


def load_mnist_image():
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    train_indices = np.arange(len(x_train))
    np.random.shuffle(train_indices)
    x_train = x_train[train_indices]
    x_test = x_test.astype('float32') / 255.
    test_indices = np.arange(len(x_train))
    np.random.shuffle(test_indices)
    x_train = x_train[test_indices]

    return x_train, x_test


def define_generator(latent_dim):
    """
    using Functional API
    """
    nch = 256
    generator_input = Input(shape=[latent_dim])
    h = Dense(nch * 14 * 14)(generator_input)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = Reshape((14, 14, nch))(h)
    h = UpSampling2D(size=(2, 2))(h)
    h = Conv2D(nch // 2, (3, 3), padding='same')(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = Conv2D(nch // 4, (3, 3), padding='same')(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = Conv2D(1, (1, 1), padding='same')(h)
    generator_value = Activation('sigmoid')(h)
    return Model(generator_input, generator_value)


def define_discriminator(input_shape=(1, 28, 28), dropout_rate=0.5):
    # HWC-format
    discriminator_input = Input((input_shape[1], input_shape[2], input_shape[0]), name='input_x')
    nch = 512
    h = Conv2D(
        nch // 2,
        (5, 5),
        strides=(2, 2),
        padding='same',
        activation='relu'
    )(discriminator_input)
    h = LeakyReLU(0.2)(h)
    h = Dropout(dropout_rate)(h)
    h = Conv2D(nch, (5, 5), strides=(2, 2), padding='same', activation='relu')(h)
    h = LeakyReLU(0.2)(h)
    h = Dropout(dropout_rate)(h)
    h = Flatten()(h)
    h = Dense(nch // 2)(h)
    h = LeakyReLU(0.2)(h)
    h = Dropout(dropout_rate)(h)
    discriminator_value = Dense(1, activation='sigmoid')(h)
    return Model(discriminator_input, discriminator_value)


def generator_sampler(latent_dim, generator):
    def fun():
        zsamples = np.random.normal(size=(10 * 10, latent_dim))
        gen = generator.predict(zsamples)
        # BHWC -> BCHW
        gen = gen.transpose(0, 3, 1, 2)
        return gen.reshape((10, 10, 28, 28))

    return fun


def main(args):
    if not os.path.exists(args.result):
        os.mkdir(args.result)
    latent_dim = 100
    input_shape = (1, 28, 28)
    generator = define_generator(latent_dim)
    discriminator = define_discriminator(input_shape)
    gan = simple_gan(generator, discriminator, normal_latent_sampling((latent_dim,)))
    generator.summary()
    discriminator.summary()
    gan.summary()

    modle = AdversarialModel(
        base_model=gan,
        player_params=[generator.trainable_weights, discriminator.trainable_weights],
        player_names=['generator', 'discriminator']
    )

    modle.adversarial_compile(
        adversarial_optimizer=AdversarialOptimizerSimultaneous(),
        player_optimizers=[Adam(1e-4, decay=1e-4), Adam(1e-3, decay=1e-4)],
        loss='binary_crossentropy'
    )

    generator_callbacks = ImageGridCallback(
        os.path.join(args.result, 'epoch-{:03d}.png'),
        generator_sampler(latent_dim, generator)
    )
    x_train, x_test = load_mnist_image()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    y = gan_targets(x_train.shape[0])
    y_test = gan_targets(x_test.shape[0])
    history = modle.fit(
        x=x_train,
        y=y,
        validation_data=(x_test, y_test),
        callbacks=[generator_callbacks],
        epochs=100,
        batch_size=32,
    )
    df = pd.DataFrame(history.history)
    df.to_csv(os.path.join(args.result, 'history.csv'))
    generator.save(os.path.join(args.result, 'generator.h5'))
    discriminator.save(os.path.join(args.result, 'discriminator.h5'))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=str, help='/path/to/save/dir', default='result')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
