"""
pygenesis/activations/sigmoid.py

This module defines the Sigmoid activation layer class for neural networks.
"""

import numpy as np

from pygenesis.activations.base import Activation


class Sigmoid(Activation):
    """
    Sigmoid activation layer for neural networks.

    This class provides the Sigmoid activation function and its derivative.

    Attributes:
        None

    Methods:
        __init__(): Constructor method.
        sigmoid(x): Compute the Sigmoid activation function.
        sigmoid_prime(x): Compute the derivative of the Sigmoid activation function.
    """

    def __init__(self):
        """
        Initializes the Sigmoid activation layer.
        """
        super().__init__(self.sigmoid, self.sigmoid_prime)

    @staticmethod
    def sigmoid(x):
        """
        Compute the Sigmoid activation function.

        Args:
            x (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The result of applying the Sigmoid activation function to the input.
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_prime(x):
        """
        Compute the derivative of the Sigmoid activation function.

        Args:
            x (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The derivative of the Sigmoid activation function applied to the input.
        """
        s = 1 / (1 + np.exp(-x))
        return s * (1 - s)
