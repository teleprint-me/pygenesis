"""
tmnn/activations/tanh.py

This module defines the Tanh activation layer class for neural networks.
"""
import numpy as np

from tmnn.activations.base import Activation


class Tanh(Activation):
    """
    Tanh activation layer for neural networks.

    This class provides the Tanh activation function and its derivative.

    Attributes:
        None

    Methods:
        __init__(): Constructor method.
        tanh(x): Compute the Tanh activation function.
        tanh_prime(x): Compute the derivative of the Tanh activation function.
    """

    def __init__(self):
        """
        Initializes the Tanh activation layer.
        """
        super().__init__(self.tanh, self.tanh_prime)

    @staticmethod
    def tanh(x):
        """
        Compute the Tanh activation function.

        Args:
            x (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The result of applying the Tanh activation function to the input.
        """
        return np.tanh(x)

    @staticmethod
    def tanh_prime(x):
        """
        Compute the derivative of the Tanh activation function.

        Args:
            x (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The derivative of the Tanh activation function applied to the input.
        """
        return 1.0 - np.tanh(x) ** 2
