#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.
This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html
More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

#from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
import string

import lasagne
from parmesan.layers import NormalizeLayer, ScaleAndShiftLayer

# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


# ##################### Build the neural network model #######################

#
conv = lasagne.layers.Conv2DLayer
pool = lasagne.layers.Pool2DLayer
sumlayer = lasagne.layers.ElemwiseSumLayer
nonlin = lasagne.layers.NonlinearityLayer
def convLayer(l, num_filters, filter_size=(1,1), stride=(1,1), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.He, b=lasagne.init.He):
    l = conv(l, filter_size=filter_size, num_filters=num_filters,
		    stride=stride, nonlinearity=None, pad='same')
    l = NormalizeLayer(l)
    l = ScaleAndShiftLayer(l)
    l = nonlin(l, nonlinearity=nonlinearity)
    return l

def bottleneck(l, num_filters, stride=(1,1)):
    l = convLayer(l, num_filters=num_filters, stride=stride)
    l = convLayer(l, num_filters=num_filters, filter_size=(3,3))
    l = convLayer(l, num_filters=num_filters*4)
    return l



def build_model(input_var=None):

    
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    l1 = convLayer(l_in, num_filters=16*4) #Needs a starting layer, l_in doesnt have dimensionality

    l1_a = sumlayer([bottleneck(l1, num_filters=16), l1])
    l1_b = sumlayer([bottleneck(l1_a, num_filters=16), l1_a])
    l1_c = sumlayer([bottleneck(l1_b, num_filters=16), l1_b])
    l1_c_stride = convLayer(l1_c, num_filters=32*4, stride=(2,2)) #should these also be batch norm?
    
    l2_a = sumlayer([bottleneck(l1_c, num_filters=32, stride=(2,2)), l1_c_stride])
    l2_b = sumlayer([bottleneck(l2_a, num_filters=32), l2_a])
    l2_c = sumlayer([bottleneck(l2_b, num_filters=32), l2_b])
    l2_c_stride = convLayer(l2_c, num_filters=64*4, stride=(2,2)) #should these also be batch norm?
    
    l3_a = sumlayer([bottleneck(l2_c, num_filters=64, stride=(2,2)), l2_c_stride])
    l3_b = sumlayer([bottleneck(l3_a, num_filters=64), l3_a])
    l3_c = sumlayer([bottleneck(l3_b, num_filters=64), l3_b])
            
    l_out = lasagne.layers.DenseLayer(
                lasagne.layers.dropout(l3_c, p=.5),
                num_units=10,
                nonlinearity=lasagne.nonlinearities.softmax)

    return l_out


# ############################# Batch iterator ###############################

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################

def main():
    num_epochs=500

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    print("Building model and compiling functions...")
    network = build_model(input_var)

    all_layers = lasagne.layers.get_all_layers(network)
    num_params = lasagne.layers.count_params(network)
    print("  number of parameters: %d" % num_params)
    print("  layer output shapes:")
    for layer in all_layers:
        name = string.ljust(layer.__class__.__name__, 32)
        print("    %s %s" % (name, lasagne.layers.get_output_shape(layer)))

    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    
    print("Building cost functions ...")
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    print("Starting training...")
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        i = 0 #for debug
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            print i
            i += 1            
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
            
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

if __name__ == '__main__':
    main()
