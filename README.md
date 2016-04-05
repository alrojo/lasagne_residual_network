# Lasagne implementation of Deep Residual Networks

Recreating Deep Residual Learning for Image Recognition

http://arxiv.org/abs/1512.03385

Recreating Identity Mappings in Deep Residual Networks (only pre-activation)

http://arxiv.org/abs/1603.05027


## Dependancies

Note: CUDA and CuDNN might require root privileges.
- Ubuntu 14.04
- CUDA 6.5 (might work with lower, have not tested lower)
- Follow the lasagne installation lasagne.readthedocs.org/en/latest/user/installation.html
  - Python2.7
  - Numpy
  - Theano (NOT pip install)
  - Lasagne (should only require 0.1 from pip install, but have only tested on 0.2dev)
- CuDNN (only tested with v2)

## CuDNN

The code is setup with CuDNN. If you do not have access to CuDNN then you need to comment out:
* github.com/alrojo/lasagne_residual_network/blob/master/Deep_Residual_Network_mnist.py#L21
* github.com/alrojo/lasagne_residual_network/blob/master/Deep_Residual_Network_mnist.py#L86

And remove comments from
* github.com/alrojo/lasagne_residual_network/blob/master/Deep_Residual_Network_mnist.py#L85

## Set-up and run

The code is based on lasagnes own mnist example: github.com/Lasagne/Lasagne/blob/master/examples/mnist.py

The data is placed in the main folder for ease of use, but if you do not have the data Deep_Residual_Network_mnist.py will automatically download it.

To get an overview of commandline inputs, run

>>python Deep_Residual_Network_mnist.py -h

An example of running with num_filters=8, num_bottlenecks per layer=3 and num_epochs=500

>>python Deep_Residual_Network_mnist.py 8 3 500

## BatchNormLayer

Using a different version of BatchNormLayer (f0k's original PR before turning to cuDNN style) see  github.com/Lasagne/Lasagne/pull/467 for more information.

If you want to use the Lasagne's implementation then change:
github.com/alrojo/lasagne_residual_network/blob/master/Deep_Residual_Network_mnist.py#L22

to

import lasagne.layers.BatchNormLayer

## NOTE

If any of the provided steps does not work for you please let me know and report an issue/PR.
