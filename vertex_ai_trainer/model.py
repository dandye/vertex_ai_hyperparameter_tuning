"""
LeNet5 Model Definition for Vertex AI Training

This module contains the parameterized LeNet5 architecture used for
hyperparameter optimization experiments on Vertex AI.
"""

import torch
from torch import nn
from typing import List


class LeNet5(nn.Module):
    """
    Parameterized LeNet5 implementation for FashionMNIST classification.

    Args:
        num_conv_layers: Number of convolutional layers (1, 2, or 3)
        num_fc_layers: Number of fully connected layers (1 or 2)
        conv_channels: List of output channels for each conv layer
        kernel_size: Kernel size for convolution (3 or 5)
        pooling_type: 'avg' for average pooling or 'max' for max pooling
        activation: Activation function ('sigmoid', 'relu', 'leakyrelu', or 'softmax')
        num_classes: Number of output classes
        input_channels: Number of input channels (1 for grayscale, 3 for RGB)
    """
    def __init__(
        self,
        num_conv_layers: int = 2,
        num_fc_layers: int = 2,
        conv_channels: List[int] = [6, 16],
        kernel_size: int = 5,
        pooling_type: str = 'avg',
        activation: str = 'sigmoid',
        num_classes: int = 10,
        input_channels: int = 1
    ):
        super(LeNet5, self).__init__()

        self.num_conv_layers = num_conv_layers
        self.num_fc_layers = num_fc_layers
        self.activation = activation

        # Build activation function
        if activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'leakyrelu':
            self.act = nn.LeakyReLU()
        elif activation == 'softmax':
            # Note: Softmax is typically used only in output layer
            # Using it in hidden layers with dim=1 (across channels)
            self.act = nn.Softmax(dim=1)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build pooling layer
        if pooling_type == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        elif pooling_type == 'max':
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")

        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels

        for i in range(num_conv_layers):
            out_channels = conv_channels[i] if i < len(conv_channels) else conv_channels[-1]
            # Add padding to maintain spatial dimensions for first layer
            # kernel_size=5 needs padding=2, kernel_size=3 needs padding=1
            if i == 0:
                padding = 2 if kernel_size == 5 else 1 if kernel_size == 3 else 0
            else:
                padding = 0
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            )
            in_channels = out_channels

        # Calculate the size after convolutions and pooling
        # For FashionMNIST (28x28 input):
        # After conv1 (with padding=2): 28x28
        # After pool1: 14x14
        # After conv2: 10x10
        # After pool2: 5x5
        self.flatten = nn.Flatten()

        # Compute the flattened size based on the architecture
        if num_conv_layers == 1:
            feature_size = conv_channels[0] * 14 * 14  # After one pooling
        elif num_conv_layers == 2:
            feature_size = conv_channels[1] * 5 * 5  # After two poolings
        else:  # num_conv_layers == 3
            feature_size = conv_channels[2] * 2 * 2  # After three poolings

        # Build fully connected layers
        self.fc_layers = nn.ModuleList()

        if num_fc_layers == 1:
            self.fc_layers.append(nn.Linear(feature_size, num_classes))
        elif num_fc_layers == 2:
            self.fc_layers.append(nn.Linear(feature_size, 120))
            self.fc_layers.append(nn.Linear(120, 84))
            self.fc_layers.append(nn.Linear(84, num_classes))
        else:
            raise ValueError(f"num_fc_layers must be 1 or 2, got {num_fc_layers}")

    def forward(self, x):
        # Convolutional layers with activation and pooling
        for conv in self.conv_layers:
            x = self.pool(self.act(conv(x)))

        # Flatten
        x = self.flatten(x)

        # Fully connected layers
        for i, fc in enumerate(self.fc_layers):
            x = fc(x)
            # Apply activation to all but the last FC layer
            if i < len(self.fc_layers) - 1:
                x = self.act(x)

        return x

    def get_activations(self, x, layer_idx):
        """
        Extract activations from a specific convolutional layer.

        Args:
            x: Input tensor
            layer_idx: Index of the convolutional layer (0, 1, or 2)

        Returns:
            Activation maps from the specified layer
        """
        # Forward pass up to the specified layer
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            x = self.act(x)

            if i == layer_idx:
                return x

            x = self.pool(x)

        return None
