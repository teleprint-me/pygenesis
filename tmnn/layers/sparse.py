"""
tmnn/layers/sparse.py
"""
import numpy as np
from scipy.sparse import csr_matrix

from tmnn.layers.base import Layer


class Sparse(Layer):
    def __init__(self, input_dim, output_dim, sparsity_level=0.9):
        # Initialize weights with some sparse level
        self.weights = csr_matrix(
            np.random.randn(input_dim, output_dim)
            * (np.random.rand(input_dim, output_dim) > sparsity_level),
            dtype=np.float16,
        )
        self.biases = np.zeros(output_dim, dtype=np.float16)

    def forward(self, input_data):
        # Sparse dot product
        self.input = input_data
        self.output = self.weights.dot(self.input) + self.biases
        return self.output

    def backward(self, output_gradient, learning_rate):
        # Gradient computation with sparse matrices
        input_gradient = self.weights.T.dot(output_gradient)

        # Gradient update; convert to dense for updating, then back to sparse
        weight_gradient = self.input.T.dot(csr_matrix(output_gradient))
        self.weights -= learning_rate * weight_gradient
        self.biases -= learning_rate * np.sum(output_gradient, axis=0)

        return input_gradient
