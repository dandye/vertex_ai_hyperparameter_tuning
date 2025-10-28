"""
Vertex AI Trainer Package for LeNet5 Hyperparameter Optimization

This package contains modules for training LeNet5 models on Vertex AI
with hyperparameter tuning support.
"""

from .model import LeNet5
from .utils import (
    load_fashion_mnist,
    initialize_weights,
    create_optimizer,
    train_one_epoch,
    evaluate_accuracy,
    save_model,
    save_metrics,
    Accumulator,
    accuracy
)

__version__ = '1.0.0'

__all__ = [
    'LeNet5',
    'load_fashion_mnist',
    'initialize_weights',
    'create_optimizer',
    'train_one_epoch',
    'evaluate_accuracy',
    'save_model',
    'save_metrics',
    'Accumulator',
    'accuracy'
]
