"""
pygenesis/model/activation.py
"""
import numpy as np

from pygenesis.model.layer import Layer


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))


class Tanh(Activation):
    def __init__(self):
        super().__init__(self.tanh, self.tanh_prime)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_prime(x):
        return 1.0 - np.tanh(x) ** 2
