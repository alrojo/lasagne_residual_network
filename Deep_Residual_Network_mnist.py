#!/usr/bin/env python

"""
Lasagne implementation of ILSVRC2015 winner on the mnist dataset
Deep Residual Learning for Image Recognition
http://arxiv.org/abs/1512.03385
"""

from __future__ import print_function

import sys
import os
import time
import string

import numpy as np
import theano
import theano.tensor as T

import lasagne
import lasagne.layers.dnn
import BatchNormLayer
sys.setrecursionlimit(10000)
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

def build_cnn(input_var=None, n=1, num_filters=8):
    # Setting up layers
#    conv = lasagne.layers.Conv2DLayer
    conv = lasagne.layers.dnn.Conv2DDNNLayer # cuDNN
    nonlinearity = lasagne.nonlinearities.rectify
    sumlayer = lasagne.layers.ElemwiseSumLayer
#    scaleandshiftlayer = parmesan.layers.ScaleAndShiftLayer
#    normalizelayer = parmesan.layers.NormalizeLayer
    batchnorm = BatchNormLayer.batch_norm
    # Conv layers must have batchnormalization and
    # Micrsoft PReLU paper style init(might have the wrong one!!)
    def convLayer(l, num_filters, filter_size=(1, 1), stride=(1, 1),
                  nonlinearity=nonlinearity, pad='same', W=lasagne.init.HeNormal(gain='relu')):
        l = conv(l, num_filters=num_filters, filter_size=filter_size,
            stride=stride, nonlinearity=nonlinearity,
            pad=pad, W=W)
        l = batchnorm(l)
        return l
    
    # Bottleneck architecture as descriped in paper
    def bottleneckDeep(l, num_filters, stride=(1, 1), nonlinearity=nonlinearity):
        l = convLayer(
            l, num_filters=num_filters, stride=stride, nonlinearity=nonlinearity)
        l = convLayer(
            l, num_filters=num_filters, filter_size=(3, 3), nonlinearity=nonlinearity)
        l = convLayer(
            l, num_filters=num_filters*4, nonlinearity=nonlinearity)
        return l

    def bottleneckDeep2(l, num_filters, stride=(1, 1), nonlinearity=nonlinearity):
        l = convLayer(
            l, num_filters=num_filters, nonlinearity=nonlinearity)
        l = convLayer(
            l, num_filters=num_filters, filter_size=(3, 3), stride=stride, nonlinearity=nonlinearity)
        l = convLayer(
            l, num_filters=num_filters*4, nonlinearity=nonlinearity)
        return l

    def bottleneckShallow(l, num_filters, stride=(1, 1), nonlinearity=nonlinearity):
        l = convLayer(
            l, num_filters=num_filters*4, filter_size=(3, 3), stride=stride, nonlinearity=nonlinearity)
        l = convLayer(
            l, num_filters=num_filters*4, filter_size=(3, 3), nonlinearity=nonlinearity)
        return l
        
    bottleneck = bottleneckShallow

    # Simply stacks the bottlenecks, makes it easy to model size of architecture with int n   
    def bottlestack(l, n, num_filters):
        for _ in range(n):
            l = sumlayer([bottleneck(l, num_filters=num_filters), l])
        return l

    # Building the network
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    # First layer just a plain convLayer
    l1 = convLayer(
	    l_in, num_filters=num_filters*4, filter_size=(3, 3)) # Filters multiplied by 4 as bottleneck returns such size

    # Stacking bottlenecks and making residuals!

    l1_bottlestack = bottlestack(l1, n=n-1, num_filters=num_filters) #Using the -1 to make it fit with size of the others
    l1_residual = convLayer(l1_bottlestack, num_filters=num_filters*4*2, stride=(2, 2), nonlinearity=None) #Multiplying by 2 because of feature reduction by 2

    l2 = sumlayer([bottleneck(l1_bottlestack, num_filters=num_filters*2, stride=(2, 2)), l1_residual])
    l2_bottlestack = bottlestack(l2, n=n, num_filters=num_filters*2)
    l2_residual = convLayer(l2_bottlestack, num_filters=num_filters*2*2*4, stride=(2, 2), nonlinearity=None)# again, this is now the second reduciton in features

    l3 = sumlayer([bottleneck(l2_bottlestack, num_filters=num_filters*2*2, stride=(2, 2)), l2_residual])
    l3_bottlestack = bottlestack(l3, n=n, num_filters=num_filters*2*2)

    # And, finally, the 10-unit output layer:
    network = lasagne.layers.DenseLayer(
            l3_bottlestack,
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

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
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(n=1, num_filters=8, num_epochs=500):
    assert n>=0
    assert num_filters>0
    assert num_epochs>0
    print("Amount of bottlenecks: %d" % n)
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_cnn(input_var, n, num_filters)
    all_layers = lasagne.layers.get_all_layers(network)
    num_params = lasagne.layers.count_params(network)
    num_conv = 0
    num_nonlin = 0
    num_input = 0
    num_batchnorm = 0
    num_elemsum = 0
    num_dense = 0
    num_unknown = 0
    print("  layer output shapes:")
    for layer in all_layers:
	name = string.ljust(layer.__class__.__name__, 32)
	print("    %s %s" %(name, lasagne.layers.get_output_shape(layer)))
	if "Conv2D" in name:
	    num_conv += 1
	elif "NonlinearityLayer" in name:
	    num_nonlin += 1
	elif "InputLayer" in name:
	    num_input += 1
	elif "BatchNormLayer" in name:
	    num_batchnorm += 1
	elif "ElemwiseSumLayer" in name:
	    num_elemsum += 1
	elif "DenseLayer" in name:
	    num_dense += 1
	else:
	    num_unknown += 1
    print("  no. of InputLayers: %d" % num_input)
    print("  no. of Conv2DLayers: %d" % num_conv)
    print("  no. of BatchNormLayers: %d" % num_batchnorm)
    print("  no. of NonlinearityLayers: %d" % num_nonlin)
    print("  no. of DenseLayers: %d" % num_dense)
    print("  no. of ElemwiseSumLayers: %d" % num_elemsum)
    print("  no. of Unknown Layers: %d" % num_unknown)
    print("  total no. of layers: %d" % len(all_layers))
    print("  no. of parameters: %d" % num_params)
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
    	    inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
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

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a Deep Residual neural network on MNIST using Lasagne.")
        print("Usage: %s [NUM_BOTTLENECKS] [NUM_FILTERS] [EPOCHS]" % sys.argv[0])
        print()
        print("NUM_BOTTLENECKS: Define amount of bottlenecks with integer, e.g. 3")
	print("NUM_FILTERS: Defines the amount of filters in the first layer(doubled at each filter halfing)")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['n'] = int(sys.argv[1])
	if len(sys.argv) > 2:
	    kwargs['num_filters'] = int(sys.argv[2])
        if len(sys.argv) > 3:
            kwargs['num_epochs'] = int(sys.argv[3])
        main(**kwargs)
