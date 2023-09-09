"""
pygenesis/model/error.py
"""
import numpy as np


def mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


def regularized_mse(y_true, y_pred, weights, lambda_):
    mse_loss = np.mean(np.square(y_true - y_pred))
    l2_penalty = lambda_ * np.sum(np.square(weights))
    return mse_loss + l2_penalty


def rms(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


def rms_prime(y_true, y_pred):
    return (y_pred - y_true) / (
        y_true.size * np.sqrt(np.mean(np.square(y_true - y_pred)))
    )


def regularized_rms(y_true, y_pred, weights, lambda_):
    rms_loss = np.sqrt(np.mean(np.square(y_true - y_pred)))
    l2_penalty = lambda_ * np.sum(np.square(weights))
    return rms_loss + l2_penalty
