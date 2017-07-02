import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, 
            one_hot=True)
    X_train = mnist.train.images
    y_train = mnist.train.labels
    X_validation = mnist.validation.images
    y_validation = mnist.validation.labels
    X_test = mnist.test.images
    y_test = mnist.test.labels
    print('X_train: {0} y_train:{1}'.format(
            X_train.shape, y_train.shape))
    print('X_validation: {0} y_validation:{1}'.format(
            X_validation.shape, y_validation.shape))
    print('X_test: {0} y_test:{1}'.format(
            X_test.shape, y_test.shape))
    image_raw = (X_train[1] * 255).astype(int)
    image = image_raw.reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.show()
        
if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)