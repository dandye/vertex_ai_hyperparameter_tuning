# Scaling Hyperparameter Optimization with Vertex AI: From Academic Assignment to Production Pipeline

![Image of Vertex AI TensorBoard ](https://github.com/user-attachments/assets/24541f79-eaaa-41e9-bcac-f7893f43ac15)

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Project Scope](#project-scope)
- [Why Vertex AI?](#why-vertex-ai)
- [Architecture Overview](#architecture-overview)
- [Implementation Journey and Key Learnings](#implementation-journey-and-key-learnings)
- [Implementation Details](#implementation-details)
- [Launch Script Command Reference](#launch-script-command-reference)
- [Troubleshooting](#troubleshooting)
- [Monitoring and Results](#monitoring-and-results)
- [Cost Estimation](#cost-estimation)
- [Advanced Usage](#advanced-usage)
- [Results and Performance](#results-and-performance)
- [Key Takeaways](#key-takeaways-and-recommendations)
- [Resources](#resources)
- [Conclusion](#conclusion)

## Introduction

As part of a computer vision course at the University of South Florida, I was tasked with implementing hyperparameter optimization for a LeNet5 convolutional neural network on the FashionMNIST dataset. While the provided Colab notebook worked well for initial experimentation, I saw an opportunity to explore how such workflows scale in production environments.

This post documents the successful migration from a notebook-based approach to a fully managed hyperparameter tuning pipeline on Google Cloud Vertex AI, demonstrating the platform's capabilities for parallel execution, experiment tracking, and seamless TensorBoard integration.

## Prerequisites

Before running the hyperparameter tuning pipeline, ensure you have the following set up:

### 1. GCP Project Setup

You need a Google Cloud Platform project with:
- Billing enabled
- Vertex AI API enabled
- Cloud Storage API enabled
- Artifact Registry enabled (recommended over Container Registry)

### 2. Enable Required APIs

```bash
# Enable all necessary services
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

### 3. Authentication

```bash
# Authenticate with your Google account
gcloud auth login

# Set application default credentials (required for the Python SDK)
gcloud auth application-default login

# Set your default project
gcloud config set project ${PROJECT_ID}
```

### 4. Create Environment Variables

```bash
# Set environment variables for convenience
export PROJECT_ID=$(gcloud config get-value project)
export BUCKET_NAME="${PROJECT_ID}-lenet5-hp-tuning"
export REGION=us-central1
```

### 5. Create a Regional GCS Bucket

**IMPORTANT**: The bucket region MUST match your Vertex AI region for optimal performance and to avoid errors.

```bash
# Create regional bucket
gcloud storage buckets create gs://${BUCKET_NAME} \
    --location=${REGION}
```

### 6. Local Python Environment (Optional)

For local testing before deploying to Vertex AI:

```bash
# Requires Python 3.8 or higher
python --version  # Verify Python >= 3.8

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the trainer package in development mode
pip install -e .

# Verify installation
python -c "import vertex_ai_trainer; print(vertex_ai_trainer.__version__)"
```

## Project Scope

The hyperparameter search space consists of:
- **Activation functions**: relu, softmax, and leakyrelu
- **Optimizers**: sgd, adam, and rmsprop
- **Batch sizes**: 32, 64, and 128

This creates 27 unique combinations to explore. The original notebook approach processed these sequentially, taking approximately 2.25 hours for a complete search.

## Why Vertex AI?

While Colab provides an excellent environment for prototyping, Vertex AI offers several advantages for systematic hyperparameter optimization:

### Enhanced Capabilities

1. **Parallel Execution**: Run multiple trials simultaneously, reducing total experiment time from hours to minutes
2. **Managed TensorBoard**: Persistent visualization and comparison of all trials in one place
3. **Bayesian Optimization**: Intelligent search using Google Vizier to find optimal configurations more efficiently
4. **Experiment Tracking**: Comprehensive logging of parameters, metrics, and artifacts with full lineage
5. **Resource Flexibility**: Easy access to various machine types and GPUs as needed
6. **Production Readiness**: The same code can scale from experiments to production deployments

### The Value Proposition

Vertex AI transforms experimental ML workflows into reproducible, scalable pipelines. For this assignment, it meant reducing experiment time by 3x through parallelization while gaining professional-grade experiment tracking and visualization capabilities.

## Architecture Overview

The final implementation consists of three main components:

### 1. Training Package (`vertex_ai_trainer/`)

A Python package containing the core training logic:

- **`model.py`**: LeNet5 architecture with configurable activation functions, pooling types, and layer counts
- **`utils.py`**: Data loading, weight initialization, training loops, and evaluation functions
- **`task.py`**: Main entry point that orchestrates training, logging, and artifact management

The package design separates concerns: model architecture, training utilities, and execution logic are cleanly decoupled, making the code testable and maintainable.

### 2. Custom Docker Container

A containerized training environment built on PyTorch's official CUDA-enabled base image:

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /root

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY vertex_ai_trainer/ ./vertex_ai_trainer/
COPY setup.py .

RUN pip install -e .
```

The container bundles all dependencies, ensuring consistent execution across trials. Using a custom container provides better control than pre-built Vertex AI containers and faster startup times since dependencies are pre-installed.

### 3. Job Launcher (`launch_vertex_hp_tuning.py`)

A command-line tool that configures and launches hyperparameter tuning jobs:

```python
hp_job = aiplatform.HyperparameterTuningJob(
    display_name=job_name,
    custom_job=custom_job,
    metric_spec=metric_spec,
    parameter_spec=parameter_spec,
    max_trial_count=args.max_trial_count,
    parallel_trial_count=args.max_parallel_trial_count,
    search_algorithm='bayesian',
)
```

The launcher handles project initialization, TensorBoard instance creation, service account configuration, and job monitoring.

## Implementation Journey and Key Learnings

Migrating from a notebook to Vertex AI provided valuable insights into production ML systems. Here are the key learnings from the implementation process:

### Learning 1: Implementing Robust Error Handling

When working with distributed systems, asynchronous job submission can make debugging challenging. I discovered that adding a synchronous debugging mode significantly improved the development experience:

```python
# Add debug flag for synchronous execution
run_kwargs = {
    'sync': args.debug_sync,  # Block until completion when debugging
    'create_request_timeout': 120.0,
}
hp_job.run(**run_kwargs)
```

**Best Practice**: Always implement both synchronous and asynchronous execution modes. Synchronous mode enables immediate error visibility during development, while asynchronous mode provides better resource utilization in production.

### Learning 2: Understanding Resource Locality

Vertex AI requires resources to be in the same region for optimal performance and compliance. This is an important consideration when setting up cloud infrastructure:

```bash
# Create a region-specific bucket that matches your Vertex AI region
gcloud storage buckets create gs://${BUCKET_NAME} \
    --location=${REGION}  # Must match Vertex AI region
```

**Best Practice**: Always align resource locations with your primary compute region. This ensures low latency, reduces egress costs, and complies with data residency requirements. Vertex AI's region requirements help enforce these best practices.

### Learning 3: Building Reliable TensorBoard Integration

Integrating TensorBoard with Vertex AI's managed service required careful consideration of how PyTorch's SummaryWriter interacts with Google Cloud Storage. I developed a robust pattern that separates concerns:

```python
import os
import time
import threading
import logging
from pathlib import Path
from google.cloud import storage

def sync_tensorboard_to_gcs(local_dir, gcs_dir, interval=30):
    """Reliable pattern for TensorBoard GCS integration with error handling"""

    def sync():
        # Parse GCS path components
        gcs_parts = gcs_dir[5:].split('/', 1)
        bucket_name = gcs_parts[0]
        prefix = gcs_parts[1] if len(gcs_parts) > 1 else ''

        client = storage.Client()
        bucket = client.bucket(bucket_name)

        while True:
            try:
                # Sync all local files to GCS
                for file_path in Path(local_dir).rglob('*'):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(local_dir)
                        blob_name = os.path.join(prefix, str(relative_path))
                        blob = bucket.blob(blob_name)
                        blob.upload_from_filename(str(file_path))
            except Exception as e:
                logging.warning(f"TensorBoard sync error: {e}")
                # Continue syncing after error

            time.sleep(interval)

    thread = threading.Thread(target=sync, daemon=True)
    thread.start()
```

This approach leverages the reliability of the Google Cloud Storage Python client while maintaining compatibility with PyTorch's SummaryWriter.

**Best Practice**: Decouple logging from storage. Write logs locally for speed and reliability, then sync to cloud storage asynchronously. This pattern works across different ML frameworks and storage backends.

### Learning 4: Optimizing for Containerized Environments

Containerized environments have different resource characteristics than local development. PyTorch's DataLoader multiprocessing, which works well locally, benefits from adjustment in containers:

```python
train_iter, test_iter = load_fashion_mnist(
    batch_size=args.batch_size,
    num_workers=0  # Optimal for containerized environments
)
```

Setting `num_workers=0` ensures stable execution by avoiding potential file descriptor limits and shared memory constraints in container runtimes.

**Best Practice**: Profile your workloads in the target environment. Container orchestration systems have specific resource limits and sharing models that may differ from local development. Adjust parallelism settings accordingly.

### Learning 5: Ensuring Data Persistence

To guarantee that hyperparameter metadata appears in TensorBoard's HParams dashboard, I implemented a robust finalization pattern:

```python
import time
from pathlib import Path
from google.cloud import storage

# Write hyperparameters and metrics
tb_writer.add_hparams(hparams, metrics)
tb_writer.flush()
tb_writer.close()

# Ensure all data is persisted before exit
if gcs_tensorboard_dir:
    time.sleep(5)  # Allow flush to complete - consider making this configurable

    # Parse GCS path for final sync
    gcs_parts = gcs_tensorboard_dir[5:].split('/', 1)
    bucket_name = gcs_parts[0]
    prefix = gcs_parts[1] if len(gcs_parts) > 1 else ''

    # Perform final synchronous upload
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for file_path in Path(local_dir).rglob('*'):
        if file_path.is_file():
            relative_path = file_path.relative_to(local_dir)
            blob_name = f"{prefix}/{relative_path}" if prefix else str(relative_path)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(file_path))
```

This pattern ensures critical metadata is never lost, even during rapid job completion.

**Best Practice**: Always implement graceful shutdown procedures for distributed systems. Explicitly flush buffers and verify critical data persistence before process termination.

### Learning 6: Verifying Accelerator Utilization

To ensure optimal resource utilization, I added comprehensive GPU detection and verification:

```python
# Detailed GPU detection
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

Proper GPU utilization requires correct configuration at multiple levels:

```python
worker_pool_specs = [{
    'machine_spec': {
        'machine_type': 'n1-standard-4',
        'accelerator_type': 'NVIDIA_TESLA_V100',
        'accelerator_count': 1
    },
    'replica_count': 1,
    'container_spec': {...}
}]
```

**Best Practice**: Implement comprehensive hardware detection and logging. Verify accelerator attachment, driver compatibility, and actual utilization through performance metrics rather than configuration alone.

## Implementation Details

### Hyperparameter Search Space

The final implementation tunes three hyperparameters:

```python
parameter_spec = {
    'activation': hpt.CategoricalParameterSpec(
        values=['softmax', 'relu', 'leakyrelu']
    ),
    'optimizer': hpt.CategoricalParameterSpec(
        values=['sgd', 'adam', 'rmsprop']
    ),
    'batch_size': hpt.DiscreteParameterSpec(
        values=[32, 64, 128],
        scale='linear'
    ),
}
```

This creates 27 possible combinations (3 × 3 × 3). Using Bayesian optimization, Vertex AI can find near-optimal configurations with fewer than 27 trials by building a probabilistic model of the hyperparameter space.

### TensorBoard Integration

Each trial logs comprehensive metrics to TensorBoard:

- **Scalars**: Training loss, training accuracy, test accuracy, epoch time
- **Model graph**: Network architecture visualization
- **Hyperparameters**: Searchable table comparing all trials

The managed TensorBoard instance persists across job runs, enabling longitudinal experiment tracking.

### Reporting Metrics to Vertex AI

The training script reports the final metric to Vertex AI's hyperparameter tuning service:

```python
import hypertune

hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag='test_accuracy',
    metric_value=best_test_acc,
    global_step=args.num_epochs
)
```

This allows Vertex AI to compare trials and identify the best configuration.

## Quick Start

### Build and Push Container

```bash
# Option 1: Using Artifact Registry (Recommended)
# First, create an Artifact Registry repository if not exists
gcloud artifacts repositories create ml-containers \
    --repository-format=docker \
    --location=${REGION} \
    --description="ML training containers"

# Build with Cloud Build
gcloud builds submit --tag ${REGION}-docker.pkg.dev/${PROJECT_ID}/ml-containers/lenet5-trainer:latest .

# Option 2: Using Container Registry (being deprecated)
# gcloud builds submit --tag gcr.io/${PROJECT_ID}/lenet5-trainer:latest .
```

### Launch Training

#### Option 1: Start with CPU and Debug Mode (Recommended First Step)

```bash
# Pre-flight validation
echo "Project: ${PROJECT_ID}"
echo "Bucket: gs://${BUCKET_NAME}"
echo "Region: ${REGION}"

# Verify bucket exists
gcloud storage ls gs://${BUCKET_NAME} || echo "ERROR: Bucket not found!"

# First attempt: CPU only with debug mode to see errors immediately
python launch_vertex_hp_tuning.py \
    --project ${PROJECT_ID} \
    --bucket ${BUCKET_NAME} \
    --use-custom-container \
    --container-uri ${REGION}-docker.pkg.dev/${PROJECT_ID}/ml-containers/lenet5-trainer:latest \
    --max-trial-count 3 \
    --max-parallel-trial-count 1 \
    --debug-sync  # SEE ERRORS IMMEDIATELY!
```

#### Option 2: Scale Up with Parallelization

```bash
# Production run: All 27 trials with 3 parallel executions
python launch_vertex_hp_tuning.py \
    --project ${PROJECT_ID} \
    --bucket ${BUCKET_NAME} \
    --use-custom-container \
    --container-uri ${REGION}-docker.pkg.dev/${PROJECT_ID}/ml-containers/lenet5-trainer:latest \
    --max-trial-count 27 \
    --max-parallel-trial-count 3
```

#### Option 3: Use Pre-built Vertex AI Containers

```bash
# Skip custom container build by using Vertex AI's pre-built images
python launch_vertex_hp_tuning.py \
    --project ${PROJECT_ID} \
    --bucket ${BUCKET_NAME} \
    --max-trial-count 27 \
    --max-parallel-trial-count 3
```

#### Option 4: Enable GPU Acceleration

```bash
# Use GPU for faster training
python launch_vertex_hp_tuning.py \
    --project ${PROJECT_ID} \
    --bucket ${BUCKET_NAME} \
    --machine-type n1-standard-4 \
    --accelerator-type NVIDIA_TESLA_T4 \
    --accelerator-count 1 \
    --max-trial-count 27 \
    --max-parallel-trial-count 3
```

**Available GPU types:**
- `NVIDIA_TESLA_K80` - Budget option, older generation
- `NVIDIA_TESLA_P4` - Good performance for inference
- `NVIDIA_TESLA_T4` - **Recommended** for most workloads
- `NVIDIA_TESLA_V100` - High performance training
- `NVIDIA_TESLA_A100` - Best performance, highest cost

### Launch Script Command Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--project` | **Required** | GCP Project ID |
| `--bucket` | **Required** | GCS bucket name (without gs:// prefix) |
| `--region` | `us-central1` | GCP region for resources |
| `--job-name` | Auto-generated | Custom job name with timestamp |
| `--experiment-name` | `lenet5-hp-tuning` | Vertex AI Experiment name |
| `--tensorboard-name` | Auto-create | Existing TensorBoard instance name |
| `--machine-type` | `n1-standard-4` | Machine type (n1-standard-4, n1-highmem-8, etc.) |
| `--accelerator-type` | None | GPU type (see list above) |
| `--accelerator-count` | 1 | Number of GPUs per trial |
| `--use-custom-container` | False | Use custom Docker container |
| `--container-uri` | Auto-select | Custom container URI (gcr.io/PROJECT/...) |
| `--num-epochs` | 10 | Training epochs per trial |
| `--max-trial-count` | 27 | Maximum number of hyperparameter trials |
| `--max-parallel-trial-count` | 3 | Number of parallel trials |
| `--debug-sync` | False | Synchronous execution for debugging |
| `--enable-model-registry` | False | Auto-register best model |

### Local Testing

Test the training script locally before launching on Vertex AI:

```bash
# Run a quick local test
python -m vertex_ai_trainer.task \
    --activation relu \
    --optimizer adam \
    --batch-size 64 \
    --num-epochs 2
```

**Expected Output:**
```
Starting training with hyperparameters:
  Activation: relu
  Optimizer: adam
  Batch size: 64
  Epochs: 2

CUDA available: False
Device: cpu

Downloading FashionMNIST dataset...
Dataset loaded: 60000 training samples, 10000 test samples

Epoch 1/2
Train Loss: 0.432, Train Acc: 0.841
Test Acc: 0.854

Epoch 2/2
Train Loss: 0.298, Train Acc: 0.891
Test Acc: 0.875

Training completed successfully!
Final test accuracy: 0.875
```

## Troubleshooting

### Quick Reference Table

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| "Job accepted but never appeared" | Silent validation failure | Add `--debug-sync` flag |
| "Bucket location mismatch" | Multi-region vs regional bucket | Create bucket with `--location=${REGION}` |
| "No filesystem for prefix gs" | TensorBoard GCS issue | Use local write + sync pattern |
| "Bad file descriptor" | DataLoader multiprocessing | Set `num_workers=0` |
| "No hparams data" | Daemon thread died early | Add explicit final sync |
| "No GPU detected" | Missing accelerator spec | Check worker_pool_specs config |
| "Container build timeout" | apt-get network issues | Remove apt-get from Dockerfile |

### Detailed Troubleshooting Guide

#### 1. Permission Denied Errors

**Error Message:**
```
ERROR: Permission denied on GCS bucket
ERROR: 403 Forbidden
```

**Solution:**
Ensure your service account has proper IAM roles:

```bash
# Add AI Platform User role
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:YOUR_SERVICE_ACCOUNT@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Add Storage Admin role
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:YOUR_SERVICE_ACCOUNT@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/storage.admin"
```

Or use your default compute service account:
```bash
export PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
export SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/aiplatform.user"
```

#### 2. Container Not Found

**Error Message:**
```
ERROR: Failed to pull image gcr.io/PROJECT/lenet5-trainer:latest
ERROR: Container image not found
```

**Solution:**
- Verify the container was pushed successfully:
  ```bash
  gcloud container images list --repository=gcr.io/${PROJECT_ID}
  ```
- Rebuild and push:
  ```bash
  gcloud builds submit --tag gcr.io/${PROJECT_ID}/lenet5-trainer:latest .
  ```
- Check Container Registry permissions:
  ```bash
  gcloud projects add-iam-policy-binding ${PROJECT_ID} \
      --member="serviceAccount:${SERVICE_ACCOUNT}" \
      --role="roles/containerregistry.ServiceAgent"
  ```

#### 3. Resource Quota Exceeded

**Error Message:**
```
ERROR: Quota exceeded for resource 'NVIDIA_T4_GPUS' in region 'us-central1'
ERROR: Insufficient regional quota
```

**Solution:**
- Request quota increase in [GCP Console → IAM & Admin → Quotas](https://console.cloud.google.com/iam-admin/quotas)
- Use a different GPU type or region with available quota
- Reduce `--max-parallel-trial-count` to use fewer resources simultaneously
- Switch to CPU-only for testing: remove `--accelerator-type` flag

#### 4. Training Script Failures

**Error Message:**
```
ERROR: Training script failed with exit code 1
ERROR: Module not found
```

**Solution:**
- Check detailed logs in Vertex AI Console
- Test locally first:
  ```bash
  python -m vertex_ai_trainer.task \
      --activation relu \
      --optimizer adam \
      --batch-size 32 \
      --num-epochs 1
  ```
- Verify all dependencies are in requirements.txt
- Check that setup.py installs correctly:
  ```bash
  pip install -e .
  python -c "import vertex_ai_trainer; print(vertex_ai_trainer.__version__)"
  ```

#### 5. Job Accepted But Never Runs

**Error Message:**
Job shows as "Accepted" but never progresses to "Running"

**Solution:**
This is usually a silent validation failure. Use `--debug-sync` flag:
```bash
python launch_vertex_hp_tuning.py \
    --project ${PROJECT_ID} \
    --bucket ${BUCKET_NAME} \
    --debug-sync \
    --max-trial-count 1
```

The synchronous mode will show the actual error message immediately.

#### 6. TensorBoard Data Not Appearing

**Error Message:**
TensorBoard shows no data or missing hyperparameter comparisons

**Solution:**
- Ensure the TensorBoard sync pattern is working (check task.py implementation)
- Verify GCS bucket has the tensorboard logs:
  ```bash
  gcloud storage ls gs://${BUCKET_NAME}/tensorboard/
  ```
- Check that the TensorBoard instance is pointed at the correct GCS path
- Wait a few minutes - TensorBoard may take time to process new data

### Job Recovery and Cleanup

#### Recovering from Partial Failures

If your hyperparameter tuning job partially fails:

```bash
# List recent jobs
gcloud ai hp-tuning-jobs list --region=${REGION} --limit=5

# Get detailed status of a specific job
gcloud ai hp-tuning-jobs describe JOB_ID --region=${REGION}

# Cancel a stuck job
gcloud ai hp-tuning-jobs cancel JOB_ID --region=${REGION}
```

#### Resuming from a Specific Point

To continue from where a failed job left off:

```python
# In launch_vertex_hp_tuning.py, add logic to skip completed trials
completed_trials = ['relu_adam_32', 'softmax_sgd_64']  # From previous run

# Modify parameter_spec to exclude completed combinations
remaining_params = {
    'activation': hpt.CategoricalParameterSpec(
        values=[v for v in ['relu', 'softmax', 'leakyrelu']
                if f"{v}_{optimizer}_{batch}" not in completed_trials]
    ),
    # ... continue for other parameters
}
```

#### Cleaning Up Resources

After experiments, clean up to avoid ongoing charges:

```bash
# Delete TensorBoard instances
gcloud ai tensorboards list --region=${REGION}
gcloud ai tensorboards delete TENSORBOARD_ID --region=${REGION}

# Clean up GCS artifacts (be careful!)
# List first to verify
gcloud storage ls gs://${BUCKET_NAME}/lenet5_hp_tuning/

# Delete old job artifacts (keep best models!)
gcloud storage rm -r gs://${BUCKET_NAME}/lenet5_hp_tuning/OLD_JOB_ID/

# Delete unused container images
gcloud artifacts docker images list --repository=ml-containers --location=${REGION}
gcloud artifacts docker images delete ${REGION}-docker.pkg.dev/${PROJECT_ID}/ml-containers/lenet5-trainer:OLD_TAG
```

## Monitoring and Results

### Vertex AI Console

Monitor your hyperparameter tuning job in real-time via the Vertex AI Console:

**Direct Link:**
```
https://console.cloud.google.com/vertex-ai/training/hyperparameter-tuning-jobs?project=${PROJECT_ID}
```

You can view:
- Trial progress and current status
- Real-time metrics for each trial
- Best trial identification
- Resource utilization
- Logs for each trial

### Vertex AI TensorBoard

Access the managed TensorBoard instance for detailed visualization:

1. Navigate to [Vertex AI Console → TensorBoard](https://console.cloud.google.com/vertex-ai/experiments/tensorboard-instances)
2. Click on your TensorBoard instance
3. View comprehensive visualizations:
   - Training curves (loss, accuracy over time)
   - Model architecture graphs
   - Hyperparameter comparison tables
   - Parallel coordinates plots

### Programmatic Access

Query job status and results using the Python SDK:

```python
import os
from google.cloud import aiplatform

# Initialize the SDK
project_id = os.environ.get('PROJECT_ID', 'your-project-id')
aiplatform.init(project=project_id, location='us-central1')

# Get the most recent hyperparameter tuning job
jobs = aiplatform.HyperparameterTuningJob.list()
job = jobs[0]  # Most recent job

# Check job status
print(f"Job name: {job.display_name}")
print(f"Job state: {job.state}")
print(f"Created: {job.create_time}")

# Get all trials
trials = job.trials
print(f"Total trials: {len(trials)}")

# Find the best trial
best_trial = max(trials, key=lambda t: t.final_measurement.metrics[0].value)
print(f"\nBest Trial:")
print(f"  Accuracy: {best_trial.final_measurement.metrics[0].value:.4f}")
print(f"  Hyperparameters: {best_trial.parameters}")
print(f"  Trial ID: {best_trial.id}")
```

### Download Artifacts

Download trained models and metrics from GCS:

```bash
# List all artifacts for your job
gcloud storage ls gs://${BUCKET_NAME}/lenet5_hp_tuning/

# Download artifacts for a specific trial
gcloud storage cp -r gs://${BUCKET_NAME}/lenet5_hp_tuning/JOB_NAME/TRIAL_ID/ ./local_directory/

# Download best model (after identifying best trial)
gcloud storage cp gs://${BUCKET_NAME}/lenet5_hp_tuning/JOB_NAME/BEST_TRIAL_ID/model.pth ./best_model.pth
```

## Cost Estimation

Understanding costs is important for planning your hyperparameter tuning experiments.

### Pricing Factors

1. **Compute Time**: Charged per node-hour based on machine type
2. **Machine Specifications**: Higher specs = higher hourly cost
3. **GPU Accelerators**: Significantly increase compute costs
4. **Storage**: GCS storage for artifacts and logs
5. **TensorBoard**: Managed TensorBoard instance fees
6. **Network**: Minimal egress costs if resources are in same region

### Example Cost Estimates (us-central1, approximate)

| Configuration | Cost per Hour | Est. Cost for 27 Trials (10 epochs each, ~30 min per trial) |
|---------------|---------------|-------------------------------------------------------------|
| **n1-standard-4** (CPU only) | $0.19 | ~$2-3 (with 3 parallel trials) |
| **n1-standard-4 + T4 GPU** | $0.54 | ~$5-7 (with 3 parallel trials) |
| **n1-standard-8 + V100 GPU** | $2.95 | ~$25-35 (with 3 parallel trials) |
| **n1-standard-8 + A100 GPU** | $4.50 | ~$40-50 (with 3 parallel trials) |

**Note**: These are approximate costs as of late 2025. Use the [GCP Pricing Calculator](https://cloud.google.com/products/calculator) for current estimates.

### Cost Breakdown Example

For a typical run with **27 trials, 3 parallel, CPU-only** (n1-standard-4):

```
Compute: 27 trials × 0.5 hours × $0.19/hour = ~$2.57
Storage: ~$0.10 (for models, logs, and TensorBoard data)
TensorBoard: ~$0.30/month (prorated for usage)
---------------------------------------------------------
Total: ~$3.00
```

### Cost Optimization Tips

1. **Start Small**: Use `--max-trial-count 3` for initial validation
2. **CPU First**: Test with CPU-only before enabling expensive GPUs
3. **Parallel Tuning**: Balance speed vs. cost
   - More parallel trials = faster completion but higher instantaneous cost
   - Fewer parallel trials = lower concurrent cost but longer total time
4. **Region Selection**: Some regions have lower pricing
5. **Early Stopping**: Implement early stopping to terminate poor trials quickly
6. **Use Efficient Search**: Bayesian optimization finds good results with fewer trials than grid search
7. **Monitor Actively**: Stop jobs early if results aren't promising

## Advanced Usage

### Custom Hyperparameter Search Space

To modify the hyperparameters being tuned, edit the `parameter_spec` dictionary in `launch_vertex_hp_tuning.py`:

```python
from google.cloud.aiplatform import hyperparameter_tuning as hpt

# Add continuous parameters
parameter_spec = {
    'learning_rate': hpt.DoubleParameterSpec(
        min=0.0001,
        max=0.01,
        scale='log'  # Logarithmic scale for learning rate
    ),
    'dropout': hpt.DoubleParameterSpec(
        min=0.1,
        max=0.5,
        scale='linear'
    ),
    # Keep existing categorical parameters
    'activation': hpt.CategoricalParameterSpec(
        values=['relu', 'leakyrelu']
    ),
}
```

### Model Deployment to Vertex AI Endpoints

Deploy your best model for online predictions:

```python
import os
from google.cloud import aiplatform

# Initialize
project_id = os.environ.get('PROJECT_ID')
bucket_name = os.environ.get('BUCKET_NAME')
aiplatform.init(project=project_id, location='us-central1')

# Create an endpoint
endpoint = aiplatform.Endpoint.create(
    display_name="lenet5-fashion-mnist-endpoint",
    description="LeNet5 model for FashionMNIST classification"
)

# Upload the model (replace JOB_NAME and BEST_TRIAL_ID with actual values)
model = aiplatform.Model.upload(
    display_name="lenet5-best-model",
    artifact_uri=f"gs://{bucket_name}/lenet5_hp_tuning/JOB_NAME/BEST_TRIAL_ID/",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-0:latest",
    serving_container_predict_route="/predict",
    serving_container_health_route="/health",
)

# Deploy to endpoint
model.deploy(
    endpoint=endpoint,
    deployed_model_display_name="lenet5-v1",
    machine_type="n1-standard-2",
    min_replica_count=1,
    max_replica_count=3,
    accelerator_type="NVIDIA_TESLA_T4",  # Optional
    accelerator_count=1,  # Optional
)

print(f"Model deployed to: {endpoint.resource_name}")
```

### Batch Prediction

For offline batch predictions on large datasets:

```python
batch_prediction_job = model.batch_predict(
    job_display_name="lenet5-batch-prediction",
    gcs_source=f"gs://{bucket_name}/input_images/*.png",
    gcs_destination_prefix=f"gs://{bucket_name}/predictions/",
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
)

# Monitor progress
batch_prediction_job.wait()
print(f"Predictions saved to: {batch_prediction_job.output_info}")
```

### Comparison: Local vs. Vertex AI Training

| Feature | Local Training | Vertex AI Training |
|---------|----------------|-------------------|
| **Scalability** | Limited by local hardware | Unlimited, on-demand resources |
| **Parallelization** | Sequential or limited | Up to 100+ parallel trials |
| **Initial Cost** | Hardware investment | Pay-per-use, no upfront cost |
| **GPU Access** | Depends on local hardware | On-demand GPUs (K80 to A100) |
| **Monitoring** | Manual setup required | Managed TensorBoard included |
| **Experiment Tracking** | Manual implementation | Automatic with Vertex AI Experiments |
| **Model Registry** | Manual versioning | Integrated model registry |
| **Fault Tolerance** | Manual restart on failure | Automatic retry and recovery |
| **Team Collaboration** | Difficult to share | Easy sharing via console |
| **Reproducibility** | Depends on local setup | Fully reproducible environments |

## Results and Performance

The migration to Vertex AI delivered significant improvements across multiple dimensions:

### Performance Metrics

- **Time to completion**: 45 minutes with 3 parallel trials vs 2.25 hours sequential (3x speedup)
- **Success rate**: 100% completion (27/27 trials) with full reproducibility
- **Cost efficiency**: Approximately $5 for complete CPU experiment, with GPU options available for time-critical workloads
- **Productivity gain**: 3x faster experimentation cycles
- **GPU acceleration**: Automatic GPU provisioning when needed (T4, V100, A100 available)
- **Bayesian optimization**: Potentially find optimal hyperparameters with fewer than 27 trials

### Technical Achievements

1. **Full Reproducibility**: Complete parameter and metric tracking with artifact lineage
2. **Scalable Architecture**: Pipeline can scale from 27 to thousands of trials
3. **Professional Tooling**: Integration with managed TensorBoard for comprehensive visualization
4. **Production Readiness**: Same infrastructure can deploy winning models to production
5. **Infrastructure as Code**: All configuration version-controlled and reproducible

### Key Implementation Lessons

#### Debugging in Production
The most valuable lesson was the importance of synchronous debugging modes. The `--debug-sync` flag proved critical:

```python
run_kwargs = {
    'sync': args.debug_sync,  # Block until job completes
    'create_request_timeout': 120.0,  # Timeout for API calls
}

hp_job.run(**run_kwargs)
```

Without this, jobs would be accepted by the API but fail silently server-side. Synchronous mode surfaces errors immediately in the terminal output.

#### Infrastructure Best Practices
The final implementation demonstrates several IaC best practices:
- **Containerization**: All dependencies packaged together
- **Configuration via arguments**: No hardcoded project IDs or bucket names
- **Reproducible builds**: Docker ensures consistent environments
- **Version control**: Entire training pipeline tracked in Git

### Learning Outcomes

The project provided hands-on experience with:
- Container orchestration and Docker best practices
- Cloud-native ML pipeline design
- Distributed system debugging and monitoring
- Production ML infrastructure patterns
- Cost optimization strategies for cloud resources

## Key Takeaways and Recommendations

For those planning similar migrations, here are the most valuable insights:

### Development Strategy
1. **Incremental Migration**: Start with basic functionality, then add features progressively
2. **Debug Mode First**: Implement synchronous execution for development before enabling parallelism
3. **Monitor Everything**: Add comprehensive logging and metrics from the start
4. **Test at Scale**: Verify behavior with small experiments before launching large searches

### Technical Best Practices
1. **Resource Alignment**: Ensure all resources (buckets, services, compute) are in the same region
2. **Graceful Degradation**: Build fallback mechanisms for external dependencies
3. **Explicit Persistence**: Never rely on implicit behavior for critical data
4. **Hardware Verification**: Always confirm accelerator utilization with performance metrics

### Platform Strengths
Vertex AI excels in several areas:
- **Managed Infrastructure**: Automatic scaling and resource provisioning
- **Integrated Tooling**: Native TensorBoard, experiment tracking, and model registry
- **Enterprise Features**: IAM integration, audit logging, and compliance controls
- **Optimization Algorithms**: State-of-the-art Bayesian optimization with Google Vizier

## Resources

### Official Documentation

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs) - Comprehensive guide to all Vertex AI features
- [Hyperparameter Tuning Overview](https://cloud.google.com/vertex-ai/docs/training/hyperparameter-tuning-overview) - Detailed hyperparameter tuning guide
- [TensorBoard on Vertex AI](https://cloud.google.com/vertex-ai/docs/experiments/tensorboard-overview) - Managed TensorBoard documentation
- [Vertex AI Python SDK](https://cloud.google.com/python/docs/reference/aiplatform/latest) - API reference
- [Google Cloud Pricing Calculator](https://cloud.google.com/products/calculator) - Estimate your costs

### PyTorch Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) - Official PyTorch docs
- [TensorBoard with PyTorch](https://pytorch.org/docs/stable/tensorboard.html) - PyTorch TensorBoard integration
- [PyTorch Docker Images](https://hub.docker.com/r/pytorch/pytorch) - Official container images

### Additional Learning

- [Vertex AI Samples Repository](https://github.com/GoogleCloudPlatform/vertex-ai-samples) - Example notebooks and code
- [ML Engineering for Production (MLOps)](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops) - Coursera specialization
- [Google Cloud Skills Boost](https://www.cloudskillsboost.google/catalog?keywords=vertex%20ai) - Hands-on labs

## Conclusion

This project successfully transformed a notebook-based hyperparameter search into a production-grade ML pipeline on Vertex AI. The resulting system provides:

- **3x faster experimentation** through parallel execution
- **100% reproducibility** with comprehensive tracking
- **Enterprise-scale capabilities** that can grow with project needs
- **Professional development experience** with modern MLOps tools

Vertex AI proved to be a powerful platform for scaling ML workflows, offering the right balance of managed services and flexibility. The learnings from this implementation provide a solid foundation for future production ML projects.

## Implementation Resources

The complete implementation demonstrates production best practices:

### Core Components
- **`vertex_ai_trainer/`**: Modular Python package with clean separation of concerns
- **`launch_vertex_hp_tuning.py`**: Robust job launcher with debugging capabilities
- **`Dockerfile`**: Optimized container configuration for ML workloads
- **`requirements.txt` & `setup.py`**: Comprehensive dependency management

### Key Features
- Synchronous and asynchronous execution modes
- Automatic service account detection
- Reliable TensorBoard integration pattern
- Comprehensive hardware detection and logging

The code follows Google Cloud best practices and can serve as a template for similar migrations from notebook-based development to production ML pipelines.

## Acknowledgments

Special thanks to the Google Cloud Vertex AI team for their comprehensive documentation and the PyTorch community for their excellent container images. This project demonstrates how academic assignments can bridge the gap between educational exercises and real-world ML engineering.
