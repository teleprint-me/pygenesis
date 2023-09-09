"""
pygenesis/errors/base.py

This module defines the base class for all loss functions used in neural networks.
"""
from abc import ABC, abstractmethod


class LossFunction(ABC):
    """
    Base class for all loss functions.

    This class serves as the base for all loss functions used in neural networks.

    Attributes:
        None

    Methods:
        __call__(self, y_true, y_pred): Calculate the loss.
        prime(self, y_true, y_pred): Calculate the derivative of the loss.
        regularized(self, y_true, y_pred, weights, lambda_): Calculate the regularized loss.
    """

    @abstractmethod
    def __call__(self, y_true, y_pred):
        """
        Calculate the loss.

        Args:
            y_true (numpy.ndarray): The true target values.
            y_pred (numpy.ndarray): The predicted values.

        Returns:
            numpy.ndarray: The calculated loss.
        """
        ...

    @abstractmethod
    def prime(self, y_true, y_pred):
        """
        Calculate the derivative of the loss.

        Args:
            y_true (numpy.ndarray): The true target values.
            y_pred (numpy.ndarray): The predicted values.

        Returns:
            numpy.ndarray: The derivative of the loss.
        """
        ...

    @abstractmethod
    def regularized(self, y_true, y_pred, weights, lambda_):
        """
        Calculate the regularized loss.

        Args:
            y_true (numpy.ndarray): The true target values.
            y_pred (numpy.ndarray): The predicted values.
            weights (numpy.ndarray): The model's weights.
            lambda_ (float): The regularization parameter.

        Returns:
            numpy.ndarray: The regularized loss.
        """
        ...
