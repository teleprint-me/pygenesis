"""
tmnn/errors/rmse.py

This module defines the Root Mean Squared Error (RMSE) loss function.
"""
import numpy as np

from tmnn.errors.base import LossFunction


class RMSE(LossFunction):
    """
    Root Mean Squared Error (RMSE) loss function.

    This class represents the RMSE loss function, which calculates the root mean squared
    error between the true values (y_true) and predicted values (y_pred).

    Args:
        lambda_ (float, optional): Regularization parameter for L2 regularization. Defaults to 1e-5.

    Methods:
        __call__(self, y_true, y_pred): Calculate the RMSE loss.
        prime(self, y_true, y_pred): Calculate the derivative of the RMSE loss.
        regularized(self, y_true, y_pred, weights, lambda_=1e-5): Calculate the regularized RMSE loss with L2 regularization.
    """

    def __call__(self, y_true, y_pred):
        """
        Calculate the root mean squared error (RMSE) between y_true and y_pred.

        Args:
            y_true (numpy.ndarray): The true values.
            y_pred (numpy.ndarray): The predicted values.

        Returns:
            float: The RMSE between y_true and y_pred.
        """
        return np.sqrt(np.mean(np.square(y_true - y_pred)))

    def prime(self, y_true, y_pred):
        """
        Calculate the derivative of the RMSE loss.

        Args:
            y_true (numpy.ndarray): The true values.
            y_pred (numpy.ndarray): The predicted values.

        Returns:
            numpy.ndarray: The derivative of the RMSE loss.
        """
        return (y_pred - y_true) / (
            y_true.size * np.sqrt(np.mean(np.square(y_true - y_pred)))
        )

    def regularized(self, y_true, y_pred, weights, lambda_=1e-5):
        """
        Calculate the regularized RMSE loss with L2 regularization.

        Args:
            y_true (numpy.ndarray): The true values.
            y_pred (numpy.ndarray): The predicted values.
            weights (numpy.ndarray): The weights of the model.
            lambda_ (float, optional): Regularization parameter for L2 regularization. Defaults to 1e-5.

        Returns:
            float: The regularized RMSE loss.
        """
        rms_loss = np.sqrt(np.mean(np.square(y_true - y_pred)))
        l2_penalty = lambda_ * np.sum(np.square(weights))
        return rms_loss + l2_penalty
