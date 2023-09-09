"""
pygenesis/model/error.py
"""
import numpy as np


class LossFunction:
    """Base class for all loss functions."""

    def __call__(self, y_true, y_pred):
        """Calculates loss."""
        pass

    def prime(self, y_true, y_pred):
        """Calculates the derivative of the loss."""
        pass


class MSE(LossFunction):
    """Mean Squared Error loss function."""

    def __call__(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def prime(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

    def regularized(self, y_true, y_pred, weights, lambda_=1e-5):
        """Regularized MSE using L2 regularization."""
        mse_loss = np.mean(np.square(y_true - y_pred))
        l2_penalty = lambda_ * np.sum(np.square(weights))
        return mse_loss + l2_penalty


class RMS(LossFunction):
    """Root Mean Squared loss function."""

    def __call__(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_true - y_pred)))

    def prime(self, y_true, y_pred):
        return (y_pred - y_true) / (
            y_true.size * np.sqrt(np.mean(np.square(y_true - y_pred)))
        )

    def regularized(self, y_true, y_pred, weights, lambda_=1e-5):
        """Regularized RMS using L2 regularization."""
        rms_loss = np.sqrt(np.mean(np.square(y_true - y_pred)))
        l2_penalty = lambda_ * np.sum(np.square(weights))
        return rms_loss + l2_penalty
