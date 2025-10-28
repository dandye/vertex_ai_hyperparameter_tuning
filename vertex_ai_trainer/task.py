"""
Vertex AI Training Task for LeNet5 Hyperparameter Optimization

This script serves as the entry point for Vertex AI Training jobs.
It handles hyperparameter tuning, TensorBoard logging, and artifact management.
"""

import argparse
import os
import time
import json
import threading
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from google.cloud import aiplatform

# Import hypertune if available (only needed when running on Vertex AI)
try:
    import hypertune
    HYPERTUNE_AVAILABLE = True
except ImportError:
    HYPERTUNE_AVAILABLE = False
    print("Warning: cloudml-hypertune not available. Hyperparameter tuning metrics will not be reported.")

from .model import LeNet5
from .utils import (
    load_fashion_mnist,
    initialize_weights,
    create_optimizer,
    train_one_epoch,
    evaluate_accuracy,
    save_model,
    save_metrics
)


def sync_tensorboard_to_gcs(local_dir, gcs_dir, interval=30):
    """
    Background thread to sync TensorBoard logs to GCS using Python client.

    Args:
        local_dir: Local directory with TensorBoard logs
        gcs_dir: GCS directory to sync to (gs://bucket/path)
        interval: Sync interval in seconds
    """
    from google.cloud import storage
    import os
    from pathlib import Path

    def sync():
        # Parse GCS path
        if not gcs_dir.startswith('gs://'):
            print(f"Invalid GCS path: {gcs_dir}")
            return

        gcs_parts = gcs_dir[5:].split('/', 1)
        bucket_name = gcs_parts[0]
        prefix = gcs_parts[1] if len(gcs_parts) > 1 else ''

        # Initialize GCS client
        try:
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            print(f"Connected to GCS bucket: {bucket_name}")
        except Exception as e:
            print(f"Failed to connect to GCS: {e}")
            return

        while True:
            try:
                # Walk through local directory and upload files
                local_path = Path(local_dir)
                uploaded = 0

                for file_path in local_path.rglob('*'):
                    if file_path.is_file():
                        # Calculate relative path and GCS blob name
                        relative_path = file_path.relative_to(local_path)
                        blob_name = os.path.join(prefix, str(relative_path))

                        # Upload file to GCS
                        blob = bucket.blob(blob_name)
                        blob.upload_from_filename(str(file_path))
                        uploaded += 1

                if uploaded > 0:
                    print(f"TensorBoard sync: uploaded {uploaded} files to {gcs_dir}")

            except Exception as e:
                print(f"TensorBoard sync error: {e}")

            time.sleep(interval)

    # Start background sync thread
    thread = threading.Thread(target=sync, daemon=True)
    thread.start()
    print(f"Started TensorBoard sync thread (interval: {interval}s)")


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='LeNet5 Training on Vertex AI')

    # Hyperparameters to tune
    parser.add_argument('--activation', type=str, default='relu',
                       choices=['softmax', 'relu', 'leakyrelu'],
                       help='Activation function')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['sgd', 'adam', 'rmsprop'],
                       help='Optimizer')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')

    # Fixed hyperparameters
    parser.add_argument('--initialization', type=str, default='xavier_normal',
                       choices=['default', 'xavier_normal', 'xavier_uniform',
                               'kaiming_normal', 'kaiming_uniform'],
                       help='Weight initialization method')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--num_conv_layers', type=int, default=2,
                       help='Number of convolutional layers')
    parser.add_argument('--num_fc_layers', type=int, default=2,
                       help='Number of fully connected layers')
    parser.add_argument('--kernel_size', type=int, default=5,
                       help='Convolution kernel size')
    parser.add_argument('--pooling_type', type=str, default='avg',
                       choices=['avg', 'max'],
                       help='Pooling type')

    # Vertex AI specific arguments
    parser.add_argument('--model_dir', type=str, default=os.environ.get('AIP_MODEL_DIR', './model'),
                       help='GCS path or local path to save model artifacts')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory to store/load dataset')
    parser.add_argument('--tensorboard_log_dir', type=str,
                       default='./logs',
                       help='TensorBoard log directory (local path - Vertex AI syncs automatically)')
    parser.add_argument('--experiment_name', type=str, default='lenet5-hp-tuning',
                       help='Vertex AI Experiment name')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Vertex AI Experiment run name')

    args = parser.parse_args()

    # Store GCS TensorBoard directory if running on Vertex AI
    # We'll write locally and sync to GCS in the background
    args.gcs_tensorboard_dir = os.environ.get('AIP_TENSORBOARD_LOG_DIR')
    if args.gcs_tensorboard_dir:
        print(f"Will sync TensorBoard logs to: {args.gcs_tensorboard_dir}")
        # Use local directory for actual TensorBoard writes
        args.tensorboard_log_dir = '/tmp/tensorboard'
    else:
        # Running locally, use the specified directory
        args.tensorboard_log_dir = './logs'

    return args


def train(args):
    """
    Main training function.

    Args:
        args: Parsed command-line arguments
    """
    # Setup device with detailed GPU debugging
    print("\n" + "="*80)
    print("GPU DETECTION DEBUG")
    print("="*80)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print(f"PyTorch built with CUDA: {torch.cuda.is_available()}")

    # Check for NVIDIA GPUs
    try:
        import subprocess
        nvidia_smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if nvidia_smi.returncode == 0:
            print("nvidia-smi output:")
            print(nvidia_smi.stdout[:500])  # First 500 chars
        else:
            print("nvidia-smi not found or failed")
    except Exception as e:
        print(f"Could not run nvidia-smi: {e}")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"\n✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("\n✗ Using CPU - GPU not detected")
    print("="*80)

    # Print hyperparameters
    print("\n" + "="*80)
    print("HYPERPARAMETERS")
    print("="*80)
    print(f"Activation: {args.activation}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Initialization: {args.initialization}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Num Epochs: {args.num_epochs}")
    print("="*80 + "\n")

    # Initialize TensorBoard writer with local directory
    os.makedirs(args.tensorboard_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=args.tensorboard_log_dir)
    print(f"TensorBoard logging to local directory: {args.tensorboard_log_dir}")

    # Start background sync to GCS if running on Vertex AI
    if hasattr(args, 'gcs_tensorboard_dir') and args.gcs_tensorboard_dir:
        sync_tensorboard_to_gcs(args.tensorboard_log_dir, args.gcs_tensorboard_dir, interval=30)

    # Load data
    print("Loading FashionMNIST dataset...")
    train_iter, test_iter = load_fashion_mnist(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0  # Use 0 for containerized environments to avoid multiprocessing issues
    )

    # Create model
    print("Creating LeNet5 model...")
    net = LeNet5(
        num_conv_layers=args.num_conv_layers,
        num_fc_layers=args.num_fc_layers,
        conv_channels=[6, 16],
        kernel_size=args.kernel_size,
        pooling_type=args.pooling_type,
        activation=args.activation,
        num_classes=10,
        input_channels=1
    )
    net.to(device)

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # Log model graph to TensorBoard
    sample_input = torch.randn(1, 1, 28, 28).to(device)
    tb_writer.add_graph(net, sample_input)

    # Initialize weights
    print(f"Initializing weights with method: {args.initialization}")
    initialize_weights(net, args.initialization)

    # Create optimizer and loss function
    optimizer = create_optimizer(net, args.optimizer, args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'epochs': []
    }

    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)

    best_test_acc = 0.0
    for epoch in range(args.num_epochs):
        start_time = time.time()

        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            net, train_iter, loss_fn, optimizer, device
        )

        # Evaluate on test set
        test_acc = evaluate_accuracy(net, test_iter, device)

        epoch_time = time.time() - start_time

        # Track best accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['epochs'].append(epoch + 1)

        # Log to TensorBoard
        tb_writer.add_scalar('Loss/train', train_loss, epoch + 1)
        tb_writer.add_scalar('Accuracy/train', train_acc, epoch + 1)
        tb_writer.add_scalar('Accuracy/test', test_acc, epoch + 1)
        tb_writer.add_scalar('Time/epoch', epoch_time, epoch + 1)

        # Print progress
        print(f'Epoch {epoch + 1}/{args.num_epochs}: '
              f'Time={epoch_time:.2f}s, '
              f'Loss={train_loss:.4f}, '
              f'Train Acc={train_acc:.4f}, '
              f'Test Acc={test_acc:.4f}')

    print("="*80)
    print(f"TRAINING COMPLETE - Best Test Accuracy: {best_test_acc:.4f}")
    print("="*80 + "\n")

    # Log hyperparameters and final metrics to TensorBoard
    hparams = {
        'activation': args.activation,
        'optimizer': args.optimizer,
        'initialization': args.initialization,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
    }
    metrics = {
        'final_test_acc': test_acc,
        'final_train_acc': train_acc,
        'final_train_loss': train_loss,
        'best_test_acc': best_test_acc
    }
    tb_writer.add_hparams(hparams, metrics)
    tb_writer.flush()  # Force write to disk
    tb_writer.close()

    # Final sync to GCS before exit (ensures hparams get uploaded)
    if hasattr(args, 'gcs_tensorboard_dir') and args.gcs_tensorboard_dir:
        print("Performing final TensorBoard sync to ensure hparams are uploaded...")
        time.sleep(5)  # Give time for flush to complete

        # Do one final explicit sync
        from google.cloud import storage
        from pathlib import Path

        try:
            # Parse GCS path and upload all files one more time
            gcs_parts = args.gcs_tensorboard_dir[5:].split('/', 1)
            bucket_name = gcs_parts[0]
            prefix = gcs_parts[1] if len(gcs_parts) > 1 else ''

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            local_path = Path(args.tensorboard_log_dir)

            uploaded = 0
            for file_path in local_path.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_path)
                    blob_name = os.path.join(prefix, str(relative_path))
                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(str(file_path))
                    uploaded += 1

            print(f"Final TensorBoard sync complete! Uploaded {uploaded} files")
        except Exception as e:
            print(f"Warning: Final TensorBoard sync failed: {e}")
            print("Hparams might not appear in TensorBoard")

    # Save model
    print(f"Saving model to {args.model_dir}...")
    model_path = save_model(net, args.model_dir)
    print(f"Model saved to: {model_path}")

    # Save metrics
    all_metrics = {
        'hyperparameters': hparams,
        'final_metrics': metrics,
        'history': history,
        'total_parameters': total_params
    }
    metrics_path = save_metrics(all_metrics, args.model_dir)
    print(f"Metrics saved to: {metrics_path}")

    # Report metric to Vertex AI Hyperparameter Tuning service
    # This is used by Vertex AI to determine the best trial
    if HYPERTUNE_AVAILABLE:
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='test_accuracy',
            metric_value=best_test_acc,
            global_step=args.num_epochs
        )
        print(f"Reported test_accuracy={best_test_acc:.4f} to Vertex AI Hyperparameter Tuning")
    else:
        print("Skipping hyperparameter tuning metric reporting (cloudml-hypertune not available)")

    return best_test_acc


def main():
    """Main entry point."""
    args = get_args()

    # Initialize Vertex AI (if running on GCP)
    try:
        # These environment variables are set by Vertex AI
        project = os.environ.get('CLOUD_ML_PROJECT_ID')
        location = os.environ.get('CLOUD_ML_REGION', 'us-central1')

        if project:
            # Initialize with experiment name
            aiplatform.init(
                project=project,
                location=location,
                experiment=args.experiment_name
            )
            print(f"Initialized Vertex AI: project={project}, location={location}")

            # Start Vertex AI Experiment run (no experiment parameter needed)
            if args.run_name is None:
                args.run_name = f"{args.activation}_{args.optimizer}_{args.initialization}_{int(time.time())}"

            aiplatform.start_run(args.run_name)
            print(f"Started experiment run: {args.run_name}")

            # Log parameters
            aiplatform.log_params({
                'activation': args.activation,
                'optimizer': args.optimizer,
                'initialization': args.initialization,
                'learning_rate': args.learning_rate,
                'batch_size': args.batch_size,
                'num_epochs': args.num_epochs,
            })
    except Exception as e:
        print(f"Could not initialize Vertex AI: {e}")
        print("Continuing with local execution...")

    # Run training
    final_acc = train(args)

    # Log final metrics to Vertex AI Experiments
    try:
        if os.environ.get('CLOUD_ML_PROJECT_ID'):
            aiplatform.log_metrics({
                'final_test_accuracy': final_acc
            })
            aiplatform.end_run()
            print("Logged metrics to Vertex AI Experiments")
    except Exception as e:
        print(f"Could not log to Vertex AI Experiments: {e}")

    print("\nTraining job completed successfully!")


if __name__ == '__main__':
    main()
