"""
pygenesis/model/layer.py
"""
from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def forward(self, input_data):
        ...

    @abstractmethod
    def backward(self, output_gradient, learning_rate):
        ...


class Dense(Layer):
    def __init__(self, input_dim, output_dim, seed=None):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weights = np.random.randn(self.input_dim, self.output_dim)
        self.biases = np.zeros((1, output_dim))

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward(self, output_gradient, learning_rate, lambda_):
        input_gradient = np.dot(output_gradient, self.weights.T)
        weights_gradient = np.dot(self.input.T, output_gradient)

        # Update parameters with regularization
        self.weights -= learning_rate * (weights_gradient + 2 * lambda_ * self.weights)
        self.biases -= learning_rate * np.sum(output_gradient, axis=0, keepdims=True)

        return input_gradient

    def get_params(self):
        return self.weights, self.biases

    def set_params(self, weights, biases):
        self.weights = weights
        self.biases = biases
