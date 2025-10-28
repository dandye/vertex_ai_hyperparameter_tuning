"""
Utility Functions for Vertex AI Training

This module contains helper functions for data loading, metrics calculation,
and artifact management.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import os
import json
from pathlib import Path


class Accumulator:
    """For accumulating sums over n variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):
    """
    Compute the number of correct predictions.

    Args:
        y_hat: Model predictions
        y: True labels

    Returns:
        Number of correct predictions
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = (y_hat.type(y.dtype) == y)
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter, device):
    """
    Compute the accuracy for a model on a dataset.

    Args:
        net: Neural network model
        data_iter: DataLoader for evaluation
        device: torch device (cuda or cpu)

    Returns:
        Accuracy as a float between 0 and 1
    """
    net.eval()
    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            metric.add(accuracy(net(X), y), y.numel())

    return metric[0] / metric[1]


def train_one_epoch(net, train_iter, loss, updater, device):
    """
    Train for one epoch.

    Args:
        net: Neural network model
        train_iter: DataLoader for training
        loss: Loss function
        updater: Optimizer
        device: torch device (cuda or cpu)

    Returns:
        Tuple of (average_loss, accuracy)
    """
    net.train()
    metric = Accumulator(3)

    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)

        updater.zero_grad()
        l.backward()
        updater.step()
        metric.add(float(l) * len(y), accuracy(y_hat, y), y.numel())

    return metric[0] / metric[2], metric[1] / metric[2]


def load_fashion_mnist(data_dir='./data', batch_size=32, num_workers=2):
    """
    Load FashionMNIST dataset.

    Args:
        data_dir: Directory to store/load the dataset
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader

    Returns:
        Tuple of (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    testset = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    train_iter = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test_iter = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_iter, test_iter


def initialize_weights(net, init_method='xavier_normal'):
    """
    Initialize model weights.

    Args:
        net: Neural network model
        init_method: Initialization method ('default', 'xavier_normal',
                     'xavier_uniform', 'kaiming_normal', 'kaiming_uniform')
    """
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            if init_method == 'xavier_normal':
                nn.init.xavier_normal_(m.weight)
            elif init_method == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight)
            elif init_method == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight)
            elif init_method == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight)
            # else: use default initialization

    net.apply(init_weights)


def create_optimizer(net, optimizer_name='adam', lr=0.001):
    """
    Create optimizer for the model.

    Args:
        net: Neural network model
        optimizer_name: Name of optimizer ('sgd', 'adam', 'rmsprop')
        lr: Learning rate

    Returns:
        Optimizer instance
    """
    if optimizer_name.lower() == 'sgd':
        return torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name.lower() == 'adam':
        return torch.optim.Adam(net.parameters(), lr=lr)
    elif optimizer_name.lower() == 'rmsprop':
        return torch.optim.RMSprop(net.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def save_model(net, model_dir, epoch=None):
    """
    Save model checkpoint.

    Args:
        net: Neural network model
        model_dir: Directory to save the model
        epoch: Optional epoch number to include in filename

    Returns:
        Path to saved model
    """
    os.makedirs(model_dir, exist_ok=True)

    if epoch is not None:
        model_path = os.path.join(model_dir, f'model_epoch_{epoch}.pth')
    else:
        model_path = os.path.join(model_dir, 'model.pth')

    torch.save(net.state_dict(), model_path)
    return model_path


def save_metrics(metrics_dict, output_dir):
    """
    Save training metrics to JSON file.

    Args:
        metrics_dict: Dictionary of metrics to save
        output_dir: Directory to save the metrics file

    Returns:
        Path to saved metrics file
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, 'metrics.json')

    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    return metrics_path
