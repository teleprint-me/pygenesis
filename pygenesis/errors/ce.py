"""
pygenesis/errors/ce.py

This module defines the Cross-Entropy (CE) loss function.
"""
import numpy as np

from pygenesis.errors.base import LossFunction


class CrossEntropy(LossFunction):
    """
    Cross-Entropy loss function for binary and multi-class classification.

    This class represents the Cross-Entropy loss function, which is commonly used
    for both binary and multi-class classification problems.

    Args:
        lambda_ (float, optional): Regularization parameter for L2 regularization. Defaults to 1e-5.

    Methods:
        __call__(self, y_true, y_pred): Calculate the Cross-Entropy loss.
        prime(self, y_true, y_pred): Calculate the derivative of the Cross-Entropy loss.
        regularized(self, y_true, y_pred, weights, lambda_=1e-5): Calculate the regularized Cross-Entropy loss with L2 regularization.
    """

    def __call__(self, y_true, y_pred):
        """
        Calculate the Cross-Entropy loss between y_true and y_pred.

        Args:
            y_true (numpy.ndarray): The true values.
            y_pred (numpy.ndarray): The predicted values.

        Returns:
            float: The Cross-Entropy loss.
        """
        epsilon = 1e-15  # To prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def prime(self, y_true, y_pred):
        """
        Calculate the derivative of the Cross-Entropy loss.

        Args:
            y_true (numpy.ndarray): The true values.
            y_pred (numpy.ndarray): The predicted values.

        Returns:
            numpy.ndarray: The derivative of the Cross-Entropy loss.
        """
        epsilon = 1e-15  # To prevent division by zero
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)

    def regularized(self, y_true, y_pred, weights, lambda_=1e-5):
        """
        Calculate the regularized Cross-Entropy loss with L2 regularization.

        Args:
            y_true (numpy.ndarray): The true values.
            y_pred (numpy.ndarray): The predicted values.
            weights (numpy.ndarray): The weights of the model.
            lambda_ (float, optional): Regularization parameter for L2 regularization. Defaults to 1e-5.

        Returns:
            float: The regularized Cross-Entropy loss.
        """
        cross_entropy_loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        l2_penalty = lambda_ * np.sum(np.square(weights))
        return cross_entropy_loss + l2_penalty
