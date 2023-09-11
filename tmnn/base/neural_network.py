"""
tmnn/base/neural_network.py

This module defines the base class for all neural network layers.
"""
from typing import Any, Mapping, Optional, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix

from tmnn.base.layer import Layer


class NeuralNetwork(Layer):
    """
    Base class for neural network layers.

    Attributes:
        input_dim (int): The dimensionality of the input data.
        output_dim (int): The dimensionality of the output data.
        dtype (Optional[np.dtype]): The data type of the layer's weights and biases.
        weights (Optional[Union[np.ndarray, csr_matrix]]): The weights of the layer.
        biases (Optional[np.ndarray]): The biases of the layer.
        seed (Optional[int]): The random seed for reproducibility.
        input (Optional[np.ndarray]): The input data to the layer.
        output (Optional[np.ndarray]): The output data from the layer.
    """

    input_dim: int
    output_dim: int
    dtype: Optional[np.dtype]
    weights: Optional[Union[np.ndarray, csr_matrix]]
    biases: Optional[np.ndarray]
    seed: Optional[int]
    input: Optional[np.ndarray]
    output: Optional[np.ndarray]

    @property
    def dimensions(self) -> Tuple[int, int]:
        """
        Get the layer's input and output dimensions as a tuple.

        Returns:
            Tuple[int, int]: A tuple containing input and output dimensions.
        """
        return self.input_dim, self.output_dim

    @property
    def io_tensors(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get the layer's current input and output tensor data.

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            A tuple containing the input and output tensors.
        """
        return self.input, self.output

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Perform a forward pass through the neural network layer.

        Args:
            input_data (np.ndarray): The input data to the layer.

        Returns:
            np.ndarray: The output data from the layer, after applying the activation function.
        """
        ...

    def backward(
        self, output_gradient: np.ndarray, **kwargs: Mapping[str, Any]
    ) -> np.ndarray:
        """
        Perform a backward pass to compute the gradient through the neural network layer.

        Args:
            output_gradient (np.ndarray): The gradient of the loss with respect to the output.
            kwargs (Mapping[str, Any]): Additional keyword arguments for more specialized layers.

        Returns:
            np.ndarray: The gradient of the loss with respect to the input, after applying the derivative of the activation function.
        """
        ...

    def get_params(self) -> Tuple[Union[np.ndarray, csr_matrix], np.ndarray, np.dtype]:
        """
        Get the current weights, biases, and data type of the layer.

        Returns:
            Tuple containing the weights, biases, and dtype.
        """
        return self.weights, self.biases, self.dtype

    def set_params(
        self,
        weights: Union[np.ndarray, csr_matrix],
        biases: np.ndarray,
        dtype: np.dtype = np.float32,
    ) -> None:
        """
        Set the weights, biases, and data type of the layer.

        Args:
            weights (Union[np.ndarray, csr_matrix]): The weights to set.
            biases (np.ndarray): The biases to set.
            dtype (np.dtype, optional): The data type. Defaults to np.float32.
        """
        self.weights = weights
        self.biases = biases
        self.dtype = dtype
