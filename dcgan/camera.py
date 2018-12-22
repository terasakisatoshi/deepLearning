import argparse
import math
import os
import time

import cv2
import keras
import numpy as np

EPSILON = 1e-7


def parase_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=str, default='result')
    args = parser.parse_args()
    return args


def restore_generator(args):
    generator_path = os.path.join(args.result, 'generator.h5')
    generator = keras.models.load_model(generator_path)
    return generator


def generate_number_from_camera(args):
    generator = restore_generator(args)
    cap = cv2.VideoCapture(0)
    if cap.isOpened() is False:
        print('Error opening video stream or file')
        exit(1)
    fps_time = 0
    while cap.isOpened():
        ret_val, image = cap.read()
        image = cv2.GaussianBlur(image, (3, 3), 5)
        image = cv2.resize(image, (10, 10))
        # normalize data which take value in interval of [0,1]
        image = 1 / (np.max(image) - np.min(image)) * (image - np.min(image))
        image = np.clip(image, 0.001, 0.999)
        u1, u2 = image[:, :, 0], image[:, :, 1]
        # generate normal distributions using Box–Muller's method
        zsample = np.sqrt(-2 * np.log(u1 + EPSILON)) * np.cos(2 * np.pi * u2)
        number = generator.predict(np.expand_dims(zsample.ravel(), axis=0))
        # (1,28,28,1) -> (28,28)
        number = np.squeeze(number, axis=(0, -1))
        number_image = cv2.resize(
            (255 * number).astype(np.uint8),
            (448, 448)
        )
        number_image = cv2.cvtColor(number_image, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            number_image,
            'FPS: % f' % (1.0 / (time.time() - fps_time)),
            (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('generate MNIST-like number', number_image)
        fps_time = time.time()
        # press Esc to exit
        if cv2.waitKey(1) == 27:
            break


def main():
    args = parase_arguments()
    generate_number_from_camera(args)


if __name__ == '__main__':
    main()
