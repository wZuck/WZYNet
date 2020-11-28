# FileName: dataloader.py (proj1)
# CreateTime: 2020.11.20
# Author: © 王子翊 MF20330086@NJU

from .parameter import Parameter
import unittest

import numpy as np


class Layer(object):
    """
    Layer class. Base class.
    """

    def __init__(self):
        super(Layer, self).__init__()
        self.input = None
        self.output = None

    def forward(self):
        raise NotImplementedError(
            "Layer {} not implemented.".format(self.__class__.__name__))

    def backward(self):
        raise NotImplementedError(
            "Backward in {} not implemented.".format(self.__class__.__name__))

    def __call__(self, *args):
        return self.forward(*args)


class Linear(Layer):
    """
    Linear layer.
    """

    def __init__(self, in_dim, out_dim):
        super(Linear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Do initialization
        self.w = Parameter(np.random.random([in_dim, out_dim]))
        self.b = Parameter(np.zeros([1, out_dim]))

    def forward(self, x):
        self.input = x  # 32 8
        self.output = x@self.w.data+self.b.data   # 32 1 # FIXME:
        return self.output

    def parameters(self):
        return [self.w, self.b]

    def backward(self, gradient):
        self.w.gradient += self.input.T @ gradient
        self.b.gradient += gradient.mean(axis=0)[np.newaxis, :]
        gradient = gradient @ self.w.data.T
        return gradient  # Backward prev_loss to prev_layer.


class Sigmoid(Layer):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        self.input = x
        self.output = 1/(1+np.exp(-x))
        return self.output

    def parameters(self):
        return []

    def backward(self, gradient):
        gradient = gradient * self.output * (1-self.output)
        return gradient


class ModuleSequence(Layer):
    def __init__(self, list_of_layers):
        super(ModuleSequence, self).__init__()
        self.layers = list_of_layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        module_params = []
        for layer in self.layers:
            module_params += layer.parameters()
        return module_params

    def backward(self, gradient):
        for layer in self.layers[::-1]:  # Reverse backward.
            gradient = layer.backward(gradient)
