"""
tmnn/layers/base.py

This module defines the base layer class for neural network layers.
"""
from abc import ABC, abstractmethod


class Layer(ABC):
    """
    Abstract base class for neural network layers.

    Attributes:
        None

    Methods:
        __init__(): Constructor method.
        forward(input_data): Forward pass through the layer.
        backward(output_gradient, learning_rate): Backward pass for gradient computation.
    """

    def __init__(self):
        """
        Initializes the Layer instance.
        """
        ...

    @abstractmethod
    def forward(self, input_data):
        """
        Perform a forward pass through the layer.

        Args:
            input_data: The input data to the layer.

        Returns:
            Output data after applying the layer's transformation.
        """
        ...

    @abstractmethod
    def backward(self, output_gradient, learning_rate):
        """
        Perform a backward pass through the layer for gradient computation.

        Args:
            output_gradient: The gradient of the loss with respect to the layer's output.
            learning_rate: The learning rate for gradient descent.

        Returns:
            The gradient of the loss with respect to the layer's input data.
        """
        ...
