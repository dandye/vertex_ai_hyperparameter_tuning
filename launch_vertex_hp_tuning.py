"""Launch Vertex AI Hyperparameter Tuning Job for LeNet5

This script configures and launches a hyperparameter tuning job on Vertex AI
with TensorBoard integration, artifact tracking, and model registry support.

Usage:
    python launch_vertex_hp_tuning.py --project YOUR_PROJECT_ID --bucket YOUR_BUCKET

Before running:
1. Set up GCP project and enable APIs (see README.md)
2. Authenticate: gcloud auth application-default login
3. Create a GCS bucket for storing artifacts
4. (Optional) Build custom training container
"""

import argparse
import os
from datetime import datetime
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Launch Vertex AI Hyperparameter Tuning for LeNet5'
    )

    # Required arguments
    parser.add_argument(
        '--project',
        type=str,
        required=True,
        help='GCP Project ID'
    )
    parser.add_argument(
        '--bucket',
        type=str,
        required=True,
        help='GCS bucket name for storing artifacts (without gs:// prefix)'
    )

    # Optional arguments
    parser.add_argument(
        '--region',
        type=str,
        default='us-central1',
        help='GCP region (default: us-central1)'
    )
    parser.add_argument(
        '--job-name',
        type=str,
        default=None,
        help='Custom job name (default: auto-generated with timestamp)'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='lenet5-hp-tuning',
        help='Vertex AI Experiment name (default: lenet5-hp-tuning)'
    )
    parser.add_argument(
        '--tensorboard-name',
        type=str,
        default=None,
        help='Existing TensorBoard instance resource name (default: create new)'
    )
    parser.add_argument(
        '--machine-type',
        type=str,
        default='n1-standard-4',
        help='Machine type (default: n1-standard-4). Options: n1-standard-4, n1-highmem-8, etc.'
    )
    parser.add_argument(
        '--accelerator-type',
        type=str,
        default=None,
        help='Accelerator type (e.g., NVIDIA_TESLA_T4, NVIDIA_TESLA_V100, NVIDIA_TESLA_A100)'
    )
    parser.add_argument(
        '--accelerator-count',
        type=int,
        default=1,
        help='Number of accelerators (default: 1)'
    )
    parser.add_argument(
        '--use-custom-container',
        action='store_true',
        help='Use custom Docker container (must be built and pushed to GCR/AR first)'
    )
    parser.add_argument(
        '--container-uri',
        type=str,
        default=None,
        help='Custom container URI (e.g., gcr.io/PROJECT/lenet5-trainer:latest)'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=10,
        help='Number of training epochs per trial (default: 10)'
    )
    parser.add_argument(
        '--max-trial-count',
        type=int,
        default=9,
        help='Maximum number of trials (default: 9)'
    )
    parser.add_argument(
        '--max-parallel-trial-count',
        type=int,
        default=3,
        help='Maximum parallel trials (default: 3)'
    )
    parser.add_argument(
        '--enable-model-registry',
        action='store_true',
        help='Enable automatic model registration for best trial'
    )
    parser.add_argument(
        '--debug-sync',
        action='store_true',
        help='Run in sync mode to see detailed errors (blocks until job starts)'
    )
    parser.add_argument(
        '--service-account',
        type=str,
        default=None,
        help='Service account email for the training job (default: auto-detect Compute Engine SA)'
    )

    return parser.parse_args()


def create_tensorboard_instance(project, region, display_name):
    """Create a new Vertex AI TensorBoard instance.

    Args:
        project: GCP project ID
        region: GCP region
        display_name: Display name for TensorBoard instance

    Returns:
        TensorBoard resource name
    """
    print("\nCreating TensorBoard instance...")
    tensorboard = aiplatform.Tensorboard.create(
        display_name=display_name,
        project=project,
        location=region,
    )
    print(f"TensorBoard created: {tensorboard.resource_name}")

    # Try to get TensorBoard URL (attribute name varies by SDK version)
    try:
        tb_url = tensorboard.gca_resource.web_app_uri
        print(f"TensorBoard URL: {tb_url}")
    except AttributeError:
        print(f"TensorBoard URL: Check console at https://console.cloud.google.com/vertex-ai/tensorboard?project={project}")

    return tensorboard.resource_name


def launch_hp_tuning_job(args):
    """Launch the hyperparameter tuning job on Vertex AI.

    Args:
        args: Parsed command-line arguments
    """
    # Initialize Vertex AI
    aiplatform.init(
        project=args.project,
        location=args.region,
        staging_bucket=f'gs://{args.bucket}'
    )

    # Generate job name if not provided
    if args.job_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = f"lenet5_hp_tuning_{timestamp}"
    else:
        job_name = args.job_name

    # Set up TensorBoard
    tensorboard_name = None
    if args.tensorboard_name is None:
        tensorboard_name = create_tensorboard_instance(
            args.project,
            args.region,
            f"lenet5-tensorboard-{datetime.now().strftime('%Y%m%d')}"
        )
    elif args.tensorboard_name.lower() not in ['skip', 'none', 'false']:
        # User provided a specific TensorBoard resource name
        tensorboard_name = args.tensorboard_name
        print(f"\nUsing existing TensorBoard: {tensorboard_name}")
    else:
        print("\nSkipping TensorBoard integration")

    # Set up base output directory
    base_output_dir = f"gs://{args.bucket}/lenet5_hp_tuning/{job_name}"

    # Determine container image
    if args.use_custom_container:
        if args.container_uri is None:
            raise ValueError("--container-uri must be provided when using --use-custom-container")
        container_uri = args.container_uri
    else:
        # Use pre-built PyTorch container
        # Use CPU version unless GPU is specified
        if args.accelerator_type:
            container_uri = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-0.py310:latest"
            print(f"Using GPU container (GPU specified: {args.accelerator_type})")
        else:
            container_uri = "us-docker.pkg.dev/vertex-ai/training/pytorch-cpu.2-0.py310:latest"
            print("Using CPU container (no GPU specified)")

    print(f"Container image: {container_uri}")

    # Define worker pool spec
    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": args.machine_type,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": container_uri,
                "command": ["python", "-m", "vertex_ai_trainer.task"],
                "args": [
                    f"--num_epochs={args.num_epochs}",
                    f"--experiment_name={args.experiment_name}",
                    # Note: --model_dir is set automatically by Vertex AI via AIP_MODEL_DIR env var
                ],
            },
        }
    ]

    # Add GPU if specified
    if args.accelerator_type:
        worker_pool_specs[0]["machine_spec"]["accelerator_type"] = args.accelerator_type
        worker_pool_specs[0]["machine_spec"]["accelerator_count"] = args.accelerator_count
        print(f"Using GPU: {args.accelerator_type} x {args.accelerator_count}")

    # Define hyperparameter specs
    # These match the arguments in task.py
    # Tuning: activation (3), optimizer (3), batch_size (3) = 27 combinations
    metric_spec = {'test_accuracy': 'maximize'}

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

    # Get service account for TensorBoard (required when using managed TensorBoard)
    # Use default Compute Engine service account
    # Format: PROJECT_NUMBER-compute@developer.gserviceaccount.com
    service_account = args.service_account

    if service_account:
        print(f"\nUsing provided service account: {service_account}")
    else:
        # Try to auto-detect the Compute Engine service account
        try:
            from google.cloud import resourcemanager_v3

            # Get project number
            client = resourcemanager_v3.ProjectsClient()
            project_name = f"projects/{args.project}"
            project_obj = client.get_project(name=project_name)
            project_number = project_obj.name.split('/')[-1]
            service_account = f"{project_number}-compute@developer.gserviceaccount.com"
            print(f"\nAuto-detected service account: {service_account}")
        except ImportError:
            print("\nNote: google-cloud-resource-manager not installed")
            print("Install it: pip install google-cloud-resource-manager")
            print("Or specify service account with: --service-account YOUR_SA@PROJECT.iam.gserviceaccount.com")
        except Exception as e:
            print(f"\nWarning: Could not auto-detect service account ({e})")
            print("Specify it manually with: --service-account YOUR_SA@PROJECT.iam.gserviceaccount.com")

    # Create custom job
    print("\nConfiguring hyperparameter tuning job...")
    print(f"Job name: {job_name}")
    print(f"Max trials: {args.max_trial_count}")
    print(f"Parallel trials: {args.max_parallel_trial_count}")
    print(f"Base output directory: {base_output_dir}")

    custom_job = aiplatform.CustomJob(
        display_name=job_name,
        worker_pool_specs=worker_pool_specs,
        base_output_dir=base_output_dir,
    )

    # Set service account if we got one
    if service_account:
        custom_job._gca_resource.job_spec.service_account = service_account

    # Create and run hyperparameter tuning job
    print("\nLaunching hyperparameter tuning job...")

    try:
        hp_job = aiplatform.HyperparameterTuningJob(
            display_name=job_name,
            custom_job=custom_job,
            metric_spec=metric_spec,
            parameter_spec=parameter_spec,
            max_trial_count=args.max_trial_count,
            parallel_trial_count=args.max_parallel_trial_count,
            search_algorithm=None,  # Let Vertex AI choose (uses Bayesian optimization)
        )

        # Run the job (this actually submits it to Vertex AI)
        print("Submitting job to Vertex AI...")
        if args.debug_sync:
            print("Note: Running in SYNC mode (will block until job starts or fails)")
        else:
            print("Note: Running in background mode (will return immediately)")
        try:
            # Build run() kwargs
            run_kwargs = {
                'sync': args.debug_sync,  # Use sync mode if debugging
                'create_request_timeout': 120.0,
            }
            # Only add tensorboard if it's not None
            if tensorboard_name is not None:
                run_kwargs['tensorboard'] = tensorboard_name

            hp_job.run(**run_kwargs)
            print("Job submission accepted by Vertex AI API")
        except Exception as run_error:
            print(f"\nError during hp_job.run():")
            print(f"  Type: {type(run_error).__name__}")
            print(f"  Message: {run_error}")

            # Provide helpful error messages for common issues
            if "GPU Accelerator is required" in str(run_error):
                print("\nTip: You're using a GPU container without specifying a GPU.")
                print("     Either add --accelerator-type NVIDIA_TESLA_T4")
                print("     or the script will use CPU container automatically.")

            raise

        # In async mode, verify the job appeared in the system
        # In sync mode, the job either works or raises an exception above
        if not args.debug_sync:
            # Give job time to appear in the system
            import time as time_module
            print("Waiting for job to appear in Vertex AI system...")

            max_attempts = 6
            for attempt in range(max_attempts):
                time_module.sleep(10)
                print(f"  Checking... (attempt {attempt + 1}/{max_attempts})")

                jobs = aiplatform.HyperparameterTuningJob.list(
                    filter=f'display_name="{job_name}"'
                )

                if len(jobs) > 0:
                    created_job = jobs[0]
                    print(f"  ✓ Job found in system!")
                    break
            else:
                # Job never appeared
                print(f"\n  ✗ Job '{job_name}' not found after {max_attempts * 10} seconds")
                print("\nChecking all jobs in project...")
                all_jobs = aiplatform.HyperparameterTuningJob.list()
                print(f"Total jobs in project: {len(all_jobs)}")

                raise RuntimeError(
                    f"Job was accepted by API but never appeared in the system.\n"
                    f"This suggests a server-side validation failure.\n\n"
                    f"Possible causes:\n"
                    f"  1. Container image validation failed\n"
                    f"  2. Worker pool spec is invalid\n"
                    f"  3. Service account lacks permissions\n"
                    f"  4. TensorBoard instance is inaccessible\n\n"
                    f"Try running with --debug-sync to see the actual error:\n"
                    f"  python {os.path.basename(__file__)} --project {args.project} --bucket {args.bucket} --debug-sync"
                )
        else:
            # In sync mode, if we got here, the job started successfully
            created_job = hp_job
            print("Job started successfully in sync mode!")

        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING JOB CREATED SUCCESSFULLY!")
        print("="*80)
        print(f"Job name: {job_name}")

        # Try to show resource name and state
        try:
            print(f"Job resource name: {created_job.resource_name}")
            print(f"Job state: {created_job.state}")
        except Exception:
            print("Job is initializing... details will be available shortly")

    except Exception as e:
        print("\n" + "="*80)
        print("ERROR: Failed to create hyperparameter tuning job!")
        print("="*80)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nPossible causes:")
        print("  1. Insufficient IAM permissions")
        print("  2. Quota limits exceeded")
        print("  3. Invalid container image")
        print("  4. Staging bucket access issues")
        print("\nTroubleshooting:")
        print("  - Check IAM roles: roles/aiplatform.user or roles/aiplatform.admin")
        print("  - Check quotas: https://console.cloud.google.com/iam-admin/quotas")
        print(f"  - Verify bucket access: gsutil ls gs://{args.bucket}")
        print("="*80)
        raise

    print(f"\nMonitor progress:")
    print(f"  Console: https://console.cloud.google.com/vertex-ai/locations/{args.region}/training/hyperparameter-tuning-jobs")
    print(f"  TensorBoard: https://console.cloud.google.com/vertex-ai/experiments/tensorboard?project={args.project}")
    print(f"\nTo check job status programmatically:")
    print(f"  # List recent jobs")
    print(f"  jobs = aiplatform.HyperparameterTuningJob.list()")
    print(f"  latest_job = jobs[0]")
    print(f"  print(latest_job.state)")
    print("="*80)

    # Optionally wait for completion and register best model
    if args.enable_model_registry:
        print("\nWaiting for job completion to register best model...")
        print("(This may take a while. You can Ctrl+C and register later)")
        try:
            created_job.wait()
            print("\nJob completed!")

            # Get best trial
            trials = created_job.trials
            best_trial = max(trials, key=lambda t: t.final_measurement.metrics[0].value)

            print(f"\nBest trial: {best_trial.id}")
            print(f"Best test accuracy: {best_trial.final_measurement.metrics[0].value:.4f}")
            print(f"Best hyperparameters:")
            for param in best_trial.parameters:
                print(f"  {param.parameter_id}: {param.value}")

            # Register model
            model = aiplatform.Model.upload(
                display_name=f"{job_name}_best_model",
                artifact_uri=f"{base_output_dir}/{best_trial.id}",
                serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-0:latest",
            )
            print(f"\nModel registered: {model.resource_name}")

        except KeyboardInterrupt:
            print("\nInterrupted. Job is still running in the background.")
            print("You can register the best model later after job completion.")

    return created_job


def main():
    """Main entry point."""
    args = parse_args()

    print("\n" + "="*80)
    print("VERTEX AI HYPERPARAMETER TUNING - LENET5")
    print("="*80)
    print(f"Project: {args.project}")
    print(f"Region: {args.region}")
    print(f"Bucket: gs://{args.bucket}")
    print(f"Machine type: {args.machine_type}")
    if args.accelerator_type:
        print(f"Accelerator: {args.accelerator_type} x {args.accelerator_count}")
    print("="*80)

    # Launch the job
    hp_job = launch_hp_tuning_job(args)

    print("\nDone!")


if __name__ == '__main__':
    main()
