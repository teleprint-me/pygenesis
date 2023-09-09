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
        # NOTE: 1e-5 is to prevent overfitting models
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
        # NOTE: 1e-5 is to prevent overfitting models
        rms_loss = np.sqrt(np.mean(np.square(y_true - y_pred)))
        l2_penalty = lambda_ * np.sum(np.square(weights))
        return rms_loss + l2_penalty


class CrossEntropy(LossFunction):
    """Cross-Entropy loss function for binary and multi-class classification."""

    def __call__(self, y_true, y_pred):
        epsilon = 1e-15  # To prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def prime(self, y_true, y_pred):
        epsilon = 1e-15  # To prevent division by zero
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)

    def regularized(self, y_true, y_pred, weights, lambda_=1e-5):
        """Regularized Cross-Entropy using L2 regularization."""
        # NOTE: 1e-5 is to prevent overfitting models
        cross_entropy_loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        l2_penalty = lambda_ * np.sum(np.square(weights))
        return cross_entropy_loss + l2_penalty
