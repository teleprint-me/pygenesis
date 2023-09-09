"""
tmnn/layers/dense.py

This module defines the Dense layer class for neural networks.
"""

import numpy as np

from tmnn.layers.base import Layer


class Dense(Layer):
    """
    Dense layer for neural networks.

    Attributes:
        input_dim (int): The input dimension of the layer.
        output_dim (int): The output dimension of the layer.
        activation_fn (str, optional): The activation function used by the layer. Default is None.
        seed (int, optional): Random seed for weight initialization. Default is None.

    Methods:
        __init__(input_dim, output_dim, activation_fn=None, seed=None): Constructor method.
        _initialize_weights(activation_fn): Initialize layer weights based on the activation function.
        forward(input_data): Perform a forward pass through the layer.
        backward(output_gradient, learning_rate, lambda_): Perform a backward pass through the layer for gradient computation.
        get_params(): Get the current weights and biases of the layer.
        set_params(weights, biases): Set the weights and biases of the layer.
    """

    def __init__(self, input_dim, output_dim, activation_fn=None, seed=None):
        """
        Initializes the Dense layer.

        Args:
            input_dim (int): The input dimension of the layer.
            output_dim (int): The output dimension of the layer.
            activation_fn (str, optional): The activation function used by the layer. Default is None.
            seed (int, optional): Random seed for weight initialization. Default is None.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim

        if seed is not None:
            np.random.seed(seed)

        self._initialize_weights(activation_fn)
        self.biases = np.zeros((1, self.output_dim))

    def _initialize_weights(self, activation_fn):
        """
        Initialize layer weights based on the activation function.

        Args:
            activation_fn (str): The activation function used by the layer.
        """
        if activation_fn == "relu":  # He Initialization - Var(W) = 1/n
            self.weights = np.random.randn(self.input_dim, self.output_dim) * np.sqrt(
                2.0 / self.input_dim
            )
        elif activation_fn in ["tanh", "sigmoid"]:
            # Xavier/Glorot Initialization - Var(W) = 2/n
            self.weights = np.random.randn(self.input_dim, self.output_dim) * np.sqrt(
                1.0 / self.input_dim
            )
        else:  # Fallback to random initialization
            self.weights = np.random.randn(self.input_dim, self.output_dim)

    def forward(self, input_data):
        """
        Perform a forward pass through the layer.

        Args:
            input_data (numpy.ndarray): The input data to the layer.

        Returns:
            numpy.ndarray: Output data after applying the layer's transformation.
        """
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward(self, output_gradient, learning_rate, lambda_):
        """
        Perform a backward pass through the layer for gradient computation.

        Args:
            output_gradient (numpy.ndarray): The gradient of the loss with respect to the layer's output.
            learning_rate (float): The learning rate for gradient descent.
            lambda_ (float): Regularization parameter.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the layer's input data.
        """
        input_gradient = np.dot(output_gradient, self.weights.T)
        weights_gradient = np.dot(self.input.T, output_gradient)

        # Update parameters with regularization
        self.weights -= learning_rate * (weights_gradient + 2 * lambda_ * self.weights)
        self.biases -= learning_rate * np.sum(output_gradient, axis=0, keepdims=True)

        return input_gradient

    def get_params(self):
        """
        Get the current weights and biases of the layer.

        Returns:
            tuple: A tuple containing the layer's current weights and biases.
        """
        return self.weights, self.biases

    def set_params(self, weights, biases):
        """
        Set the weights and biases of the layer.

        Args:
            weights (numpy.ndarray): The new weights for the layer.
            biases (numpy.ndarray): The new biases for the layer.
        """
        self.weights = weights
        self.biases = biases
