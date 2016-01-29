# Lasagne implementation of Deep Residual Networks

Recreating the Deep Residual Learning for Image Recognition

http://arxiv.org/abs/1512.03385

# Dependancies
Note: CUDA and CuDNN might require root privileges.
- Ubuntu 14.04
- CUDA 6.5 (might work with lower, have not tested lower)
- Follow the lasagne installation lasagne.readthedocs.org/en/latest/user/installation.html
  - Python2.7
  - Numpy
  - Theano (NOT pip install)
  - Lasagne (should only require 0.1 from pip install, but have only tested on 0.2dev)
- CuDNN (only tested with v2)
# CuDNN
The code is setup with CuDNN. If you do not have access to CuDNN then you need to comment out:
* github.com/alrojo/lasagne_residual_network/blob/master/Deep_Residual_Network_mnist.py#L21
* github.com/alrojo/lasagne_residual_network/blob/master/Deep_Residual_Network_mnist.py#L86
And comment "in"
* github.com/alrojo/lasagne_residual_network/blob/master/Deep_Residual_Network_mnist.py#L85
# Set-up and run
To get an overview of commandline inputs, run

>>python Deep_Residual_Network_mnist.py -h

An example of running with num_filters=8, num_bottlenecks per layer=3 and num_epochs=500

>>python Deep_Residual_Network_mnist.py 8 3 500

The code is based on lasagnes own mnist example: github.com/Lasagne/Lasagne/blob/master/examples/mnist.py

The data is placed in the main folder for ease of use, but if you do not have the data Deep_Residual_Network_mnist.py will automatically download it.
# BatchNormLayer
Using a different version of BatchNormLayer (f0k's original PR before turning to cuDNN style) see  github.com/Lasagne/Lasagne/pull/467 for more information.
