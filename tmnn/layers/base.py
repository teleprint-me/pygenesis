"""
tmnn/layers/base.py

This module defines the base layer class for neural network layers.
"""
from typing import Protocol

import numpy as np


class Layer(Protocol):
    """
    Protocol for neural network layers.

    Attributes:
        dtype (numpy.dtype): The data type used for internal computations.
    """

    dtype: np.dtype

    def forward(self, input_data):
        """
        Perform a forward pass through the layer.

        Args:
            input_data: The input data to the layer.

        Returns:
            The output data produced by the layer.
        """
        ...

    def backward(self, output_gradient, learning_rate):
        """
        Perform a backward pass for gradient computation.

        Args:
            output_gradient: The gradient of the loss with respect to the layer's output.
            learning_rate: The learning rate used for gradient descent.

        Returns:
            The gradient of the loss with respect to the layer's input.
        """
        ...
