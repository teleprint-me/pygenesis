"""
tmnn/activations/base.py

This module defines the base Activation layer class for neural networks.
"""
import numpy as np

from tmnn.model.layer import Layer


class Activation(Layer):
    """
    Base class for activation layers in neural networks.

    Attributes:
        activation (function): The activation function used by the layer.
        activation_prime (function): The derivative of the activation function.

    Methods:
        __init__(activation, activation_prime): Constructor method.
        forward(input_data): Perform a forward pass through the layer.
        backward(output_gradient, learning_rate): Perform a backward pass through the layer for gradient computation.
    """

    def __init__(self, activation, activation_prime):
        """
        Initializes the Activation layer.

        Args:
            activation (function): The activation function used by the layer.
            activation_prime (function): The derivative of the activation function.
        """
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input_data):
        """
        Perform a forward pass through the layer.

        Args:
            input_data (numpy.ndarray): The input data to the layer.

        Returns:
            numpy.ndarray: Output data after applying the activation function.
        """
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        """
        Perform a backward pass through the layer for gradient computation.

        Args:
            output_gradient (numpy.ndarray): The gradient of the loss with respect to the layer's output.
            learning_rate (float): The learning rate for gradient descent.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the layer's input data.
        """
        return np.multiply(output_gradient, self.activation_prime(self.input))
