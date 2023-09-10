"""
tmnn/errors/__init__.py
"""
from tmnn.errors.base import LossFunction
from tmnn.errors.ce import CrossEntropy
from tmnn.errors.mse import MSE
from tmnn.errors.rmse import RMSE

__all__ = [LossFunction, CrossEntropy, MSE, RMSE]
