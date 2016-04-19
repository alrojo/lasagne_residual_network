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

CuDNN is now disabled by default, to enable see below

## Set-up and run

The code is based on lasagne's own mnist example: github.com/Lasagne/Lasagne/blob/master/examples/mnist.py

The data is placed in the main folder for ease of use, but if you do not have the data Deep_Residual_Network_mnist.py will automatically download it.

To get an overview of commandline inputs, run

>>python Deep_Residual_Network_mnist.py -h

An example of running with num_blocks/res_units per layer=3, num_filters=8, num_epochs=500 and CuDNN=no

>>python Deep_Residual_Network_mnist.py 3 8 500 no

## BatchNormLayer

Using lasagnes implementation of BatchNormLayer which is the CuDNNv4 style implementation. See  github.com/Lasagne/Lasagne/pull/467 for more information.

## NOTE

If any of the provided steps does not work for you please let me know and report an issue/PR, thanks!
