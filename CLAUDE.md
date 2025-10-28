# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a hyperparameter tuning system for LeNet5 convolutional neural networks using Google Cloud Vertex AI. The project demonstrates production-grade ML pipeline implementation, migrating from a notebook-based approach to a fully managed hyperparameter tuning pipeline with parallel execution, experiment tracking, and TensorBoard integration.

## Repository Structure

- `vertex_ai_trainer/` - Python package containing training logic
  - `model.py` - LeNet5 architecture with configurable activation functions
  - `utils.py` - Data loading, training loops, and evaluation functions
  - `task.py` - Main entry point for Vertex AI training jobs
- `launch_vertex_hp_tuning.py` - Command-line tool to configure and launch hyperparameter tuning jobs
- `Dockerfile` - Container definition for training environment (PyTorch CUDA-enabled)
- `setup.py` - Package configuration for the training module
- `requirements.txt` - Python dependencies

## Key Commands

### Prerequisites and Setup

```bash
# Set environment variables
export PROJECT_ID=$(gcloud config get-value project)
export BUCKET_NAME="${PROJECT_ID}-lenet5-hp-tuning"
export REGION=us-central1

# Enable required APIs
gcloud services enable aiplatform.googleapis.com storage.googleapis.com

# Create regional GCS bucket (must match region!)
gcloud storage buckets create gs://${BUCKET_NAME} --location=${REGION}

# Authenticate
gcloud auth application-default login
```

### Build and Push Container

```bash
# Build with Cloud Build (recommended)
gcloud builds submit --tag gcr.io/${PROJECT_ID}/lenet5-trainer:latest .

# Alternative: Local Docker build
docker build -t gcr.io/${PROJECT_ID}/lenet5-trainer:latest .
docker push gcr.io/${PROJECT_ID}/lenet5-trainer:latest
```

### Launch Hyperparameter Tuning

```bash
# Debug mode (synchronous, see errors immediately)
python launch_vertex_hp_tuning.py \
    --project ${PROJECT_ID} \
    --bucket ${BUCKET_NAME} \
    --use-custom-container \
    --container-uri gcr.io/${PROJECT_ID}/lenet5-trainer:latest \
    --max-trial-count 3 \
    --max-parallel-trial-count 1 \
    --debug-sync

# Production run (27 trials, 3 parallel)
python launch_vertex_hp_tuning.py \
    --project ${PROJECT_ID} \
    --bucket ${BUCKET_NAME} \
    --use-custom-container \
    --container-uri gcr.io/${PROJECT_ID}/lenet5-trainer:latest \
    --max-trial-count 27 \
    --max-parallel-trial-count 3

# Using pre-built Vertex AI containers (no custom container needed)
python launch_vertex_hp_tuning.py \
    --project ${PROJECT_ID} \
    --bucket ${BUCKET_NAME} \
    --max-trial-count 27 \
    --max-parallel-trial-count 3
```

### Local Development and Testing

```bash
# Install package locally for development
pip install -e .

# Run training locally (outside Vertex AI)
python -m vertex_ai_trainer.task \
    --activation relu \
    --optimizer adam \
    --batch-size 64 \
    --num-epochs 10 \
    --gcs-bucket ${BUCKET_NAME}
```

## Architecture and Key Patterns

### Hyperparameter Search Space

The system tunes three hyperparameters creating 27 combinations:
- **Activation functions**: relu, softmax, leakyrelu
- **Optimizers**: sgd, adam, rmsprop
- **Batch sizes**: 32, 64, 128

### TensorBoard Integration Pattern

The codebase uses a robust local-write-then-sync pattern for TensorBoard:
1. Write logs locally using PyTorch's SummaryWriter
2. Asynchronously sync to GCS using a background thread
3. Explicit final sync before process termination

This pattern ensures reliability while maintaining compatibility with Vertex AI's managed TensorBoard.

### Debugging Strategy

The `--debug-sync` flag is critical for development. It makes job execution synchronous, surfacing errors immediately in the terminal rather than failing silently server-side.

### Container Optimization

- Uses PyTorch official CUDA-enabled base image
- Pre-installs all dependencies in container for faster startup
- Sets `num_workers=0` in DataLoader for container stability

## Important Considerations

### Region Alignment
All resources (GCS bucket, Vertex AI services, compute) must be in the same region. The bucket creation command enforces this with `--location=${REGION}`.

### GPU Configuration
When using GPUs, ensure:
1. Specify both `--accelerator-type` and `--accelerator-count` in the launcher
2. The container has CUDA support (uses pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime base)
3. Verify GPU attachment with the hardware detection logging in task.py

### Error Handling
The codebase includes comprehensive error handling and logging:
- Hardware detection and verification
- Graceful degradation for TensorBoard failures
- Explicit persistence verification for critical data

## Development Workflow

1. Make changes to the training code in `vertex_ai_trainer/`
2. Test locally using direct module execution
3. Build and push container if using custom container
4. Launch with `--debug-sync` flag for initial testing
5. Scale up parallel trials once verified

## Troubleshooting

Common issues and solutions are documented in the README:
- Silent job failures: Use `--debug-sync` flag
- Bucket location mismatch: Ensure bucket region matches Vertex AI region
- No GPU detected: Check accelerator_spec in worker_pool_specs
- TensorBoard data missing: Verify final sync completion