# Dockerfile for Vertex AI Training - LeNet5 Hyperparameter Optimization
#
# This creates a custom training container with all dependencies pre-installed.
# Building a custom container is optional - you can use pre-built Vertex AI
# containers, but custom containers provide better control and faster startup.
#
# To build and push:
#   docker build -t gcr.io/YOUR_PROJECT_ID/lenet5-trainer:latest .
#   docker push gcr.io/YOUR_PROJECT_ID/lenet5-trainer:latest
#
# Or use Artifact Registry:
#   docker build -t us-central1-docker.pkg.dev/YOUR_PROJECT_ID/REPO_NAME/lenet5-trainer:latest .
#   docker push us-central1-docker.pkg.dev/YOUR_PROJECT_ID/REPO_NAME/lenet5-trainer:latest

# Base image: PyTorch with GPU support
# Use PyTorch 2.0+ with CUDA 11.8
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /root

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training code
COPY vertex_ai_trainer/ ./vertex_ai_trainer/
COPY setup.py .

# Install the training package
RUN pip install -e .

# Set Python path
ENV PYTHONPATH=/root:$PYTHONPATH

# The entrypoint will be specified in the Vertex AI job configuration
# Typically: python -m vertex_ai_trainer.task [args]

# For debugging: Uncomment to run bash instead
# ENTRYPOINT ["/bin/bash"]
