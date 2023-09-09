"""
pygenesis/errors/mse.py

This module defines the Mean Squared Error (MSE) loss function.
"""
import numpy as np

from pygenesis.errors.base import LossFunction


class MSE(LossFunction):
    """
    Mean Squared Error (MSE) loss function.

    This class represents the Mean Squared Error (MSE) loss function used in neural networks.

    Methods:
        __call__(self, y_true, y_pred): Calculate the MSE loss.
        prime(self, y_true, y_pred): Calculate the derivative of the MSE loss.
        regularized(self, y_true, y_pred, weights, lambda_=1e-5): Calculate the regularized MSE loss.

    Attributes:
        None
    """

    def __call__(self, y_true, y_pred):
        """
        Calculate the Mean Squared Error (MSE) loss.

        Args:
            y_true (numpy.ndarray): The true target values.
            y_pred (numpy.ndarray): The predicted values.

        Returns:
            float: The calculated MSE loss.
        """
        return np.mean(np.square(y_true - y_pred))

    def prime(self, y_true, y_pred):
        """
        Calculate the derivative of the Mean Squared Error (MSE) loss.

        Args:
            y_true (numpy.ndarray): The true target values.
            y_pred (numpy.ndarray): The predicted values.

        Returns:
            numpy.ndarray: The derivative of the MSE loss.
        """
        return 2 * (y_pred - y_true) / y_true.size

    def regularized(self, y_true, y_pred, weights, lambda_=1e-5):
        """
        Calculate the regularized Mean Squared Error (MSE) loss with L2 regularization.

        Args:
            y_true (numpy.ndarray): The true target values.
            y_pred (numpy.ndarray): The predicted values.
            weights (numpy.ndarray): The model's weights.
            lambda_ (float, optional): The regularization parameter (default is 1e-5).

        Returns:
            float: The regularized MSE loss.
        """
        mse_loss = np.mean(np.square(y_true - y_pred))
        l2_penalty = lambda_ * np.sum(np.square(weights))
        return mse_loss + l2_penalty
