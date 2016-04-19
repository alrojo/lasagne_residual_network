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

# helper function for projection_b
def ceildiv(a, b):
    return -(-a // b)

def build_cnn(input_var=None, n=1, num_filters=8, cudnn='no'):
    import lasagne # For some odd reason it can't read the global import, please PR/Issue if you know why
    projection_type = 'B'
    # Setting up layers
    if cudnn == 'yes':
        import lasagne.layers.dnn
        conv = lasagne.layers.dnn.Conv2DDNNLayer # cuDNN
    else:
        conv = lasagne.layers.Conv2DLayer
    nonlin = lasagne.nonlinearities.rectify
    nonlin_layer = lasagne.layers.NonlinearityLayer
    sumlayer = lasagne.layers.ElemwiseSumLayer
    #batchnorm = BatchNormLayer.BatchNormLayer
    batchnorm = lasagne.layers.BatchNormLayer

    # Setting the projection type for when reducing height/width
    # and increasing dimensions.
    # Default is 'B' as B performs slightly better
    # and A requires newer version of lasagne with ExpressionLayer
    projection_type = 'B'
    if projection_type == 'A':
        expression = lasagne.layers.ExpressionLayer
        pad = lasagne.layers.PadLayer

    if projection_type == 'A':
        # option A for projection as described in paper
        # (should perform slightly worse than B)
        def projection(l_inp):
            n_filters = l_inp.output_shape[1]*2
            l = expression(l_inp, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], ceildiv(s[2], 2), ceildiv(s[3], 2)))
            l = pad(l, [n_filters//4,0,0], batch_ndim=1)
            return l

    if projection_type == 'B':
        # option B for projection as described in paper
        def projection(l_inp):
            # twice normal channels when projecting!
            n_filters = l_inp.output_shape[1]*2 
            l = conv(l_inp, num_filters=n_filters, filter_size=(1, 1),
                     stride=(2, 2), nonlinearity=None, pad='same', b=None)
            l = batchnorm(l)
            return l

    # helper function to handle filters/strides when increasing dims
    def filters_increase_dims(l, increase_dims):
        in_num_filters = l.output_shape[1]
        if increase_dims:
            first_stride = (2, 2)
            out_num_filters = in_num_filters*2
        else:
            first_stride = (1, 1)
            out_num_filters = in_num_filters
 
        return out_num_filters, first_stride

    # block as described and used in cifar in the original paper:
    # http://arxiv.org/abs/1512.03385
    def res_block_v1(l_inp, nonlinearity=nonlin, increase_dim=False):
        # first figure filters/strides
        n_filters, first_stride = filters_increase_dims(l_inp, increase_dim)
        # conv -> BN -> nonlin -> conv -> BN -> sum -> nonlin
        l = conv(l_inp, num_filters=n_filters, filter_size=(3, 3),
                 stride=first_stride, nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        l = batchnorm(l)
        l = nonlin_layer(l, nonlinearity=nonlin)
        l = conv(l, num_filters=n_filters, filter_size=(3, 3),
                 stride=(1, 1), nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        l = batchnorm(l)
        if increase_dim:
            # Use projection (A, B) as described in paper
            p = projection(l_inp)
        else:
            # Identity shortcut
            p = l_inp
        l = sumlayer([l, p])
        l = nonlin_layer(l, nonlinearity=nonlin)
        return l

    # block as described in second paper on the subject (by same authors):
    # http://arxiv.org/abs/1603.05027
    def res_block_v2(l_inp, nonlinearity=nonlin, increase_dim=False):
        # first figure filters/strides
        n_filters, first_stride = filters_increase_dims(l_inp, increase_dim)
        # BN -> nonlin -> conv -> BN -> nonlin -> conv -> sum
        l = batchnorm(l_inp)
        l = nonlin_layer(l, nonlinearity=nonlin)
        l = conv(l, num_filters=n_filters, filter_size=(3, 3),
                 stride=first_stride, nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        l = batchnorm(l)
        l = nonlin_layer(l, nonlinearity=nonlin)
        l = conv(l, num_filters=n_filters, filter_size=(3, 3),
                 stride=(1, 1), nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        if increase_dim:
            # Use projection (A, B) as described in paper
            p = projection(l_inp)
        else:
            # Identity shortcut
            p = l_inp
        l = sumlayer([l, p])
        return l

    def bottleneck_block(l_inp, nonlinearity=nonlin, increase_dim=False):
        # first figure filters/strides
        n_filters, first_stride = filters_increase_dims(l_inp, increase_dim)
        # conv -> BN -> nonlin -> conv -> BN -> nonlin -> conv -> BN -> sum
        # -> nonlin
        # first make the bottleneck, scale the filters ..!
        scale = 4 # as per bottleneck architecture used in paper
        scaled_filters = n_filters/scale
        l = conv(l_inp, num_filters=scaled_filters, filter_size=(1, 1),
                 stride=first_stride, nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        l = batchnorm(l)
        l = nonlin_layer(l, nonlinearity=nonlin)
        l = conv(l, num_filters=scaled_filters, filter_size=(3, 3),
                 stride=(1, 1), nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        l = batchnorm(l)
        l = nonlin_layer(l, nonlinearity=nonlin)
        l = conv(l, num_filters=n_filters, filter_size=(1, 1),
                 stride=(1, 1), nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        if increase_dim:
            # Use projection (A, B) as described in paper
            p = projection(l_inp)
        else:
            # Identity shortcut
            p = l_inp
        l = sumlayer([l, p])
        l = nonlin_layer(l, nonlinearity=nonlin)
        return l

    # Bottleneck architecture with more efficiency (the post with Kaiming He's response)
    # https://www.reddit.com/r/MachineLearning/comments/3ywi6x/deep_residual_learning_the_bottleneck/ 
    def bottleneck_block_fast(l_inp, nonlinearity=nonlin, increase_dim=False):
        # first figure filters/strides
        n_filters, last_stride = filters_increase_dims(l_inp, increase_dim)
        # conv -> BN -> nonlin -> conv -> BN -> nonlin -> conv -> BN -> sum
        # -> nonlin
        # first make the bottleneck, scale the filters ..!
        scale = 4 # as per bottleneck architecture used in paper
        scaled_filters = n_filters/scale
        l = conv(l_inp, num_filters=scaled_filters, filter_size=(1, 1),
                 stride=(1, 1), nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        l = batchnorm(l)
        l = nonlin_layer(l, nonlinearity=nonlin)
        l = conv(l, num_filters=scaled_filters, filter_size=(3, 3),
                 stride=(1, 1), nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        l = batchnorm(l)
        l = nonlin_layer(l, nonlinearity=nonlin)
        l = conv(l, num_filters=n_filters, filter_size=(1, 1),
                 stride=last_stride, nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        if increase_dim:
            # Use projection (A, B) as described in paper
            p = projection(l_inp)
        else:
            # Identity shortcut
            p = l_inp
        l = sumlayer([l, p])
        l = nonlin_layer(l, nonlinearity=nonlin)
        return l
       
    res_block = res_block_v1

    # Stacks the residual blocks, makes it easy to model size of architecture with int n   
    def blockstack(l, n, nonlinearity=nonlin):
        for _ in range(n):
            l = res_block(l, nonlinearity=nonlin)
        return l

    # Building the network
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    # First layer! just a plain convLayer
    l1 = conv(l_in, num_filters=num_filters, stride=(1, 1),
              filter_size=(3, 3), nonlinearity=None, pad='same')
    l1 = batchnorm(l1)
    l1 = nonlin_layer(l1, nonlinearity=nonlin)

    # Stacking bottlenecks and increasing dims! (while reducing shape size)
    l1_bs = blockstack(l1, n=n)
    l1_id = res_block(l1_bs, increase_dim=True)

    l2_bs = blockstack(l1_id, n=n)
    l2_id = res_block(l2_bs, increase_dim=True)

    l3_bs = blockstack(l2_id, n=n)

    # And, finally, the 10-unit output layer:
    network = lasagne.layers.DenseLayer(
            l3_bs,
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

def main(n=1, num_filters=8, num_epochs=500, cudnn='no'):
    assert n>=0
    assert num_filters>0
    assert num_epochs>0
    assert cudnn in ['yes', 'no']
    print("Amount of bottlenecks: %d" % n)
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_cnn(input_var, n, num_filters, cudnn)
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

    # several learning rates for low initial learning rates and
    # learning rate anealing (id is epoch)
    learning_rate_schedule = {
    0: 0.0001, # low initial learning rate as described in paper
    2: 0.01,
    100: 0.001,
    150: 0.0001
    }

    learning_rate = theano.shared(np.float32(learning_rate_schedule[0]))

    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=learning_rate, momentum=0.9)

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
        if epoch in learning_rate_schedule:
            lr = np.float32(learning_rate_schedule[epoch])
            print(" setting learning rate to %.7f" % lr)
            learning_rate.set_value(lr)
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
        print("CUDNN: no to not use, yes to use (default: no)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['n'] = int(sys.argv[1])
        if len(sys.argv) > 2:
            kwargs['num_filters'] = int(sys.argv[2])
        if len(sys.argv) > 3:
            kwargs['num_epochs'] = int(sys.argv[3])
        if len(sys.argv) > 4:
            kwargs['cudnn'] = sys.argv[4]
        main(**kwargs)
