"""RunPod launcher for parallel autoresearch experiments.

Creates pods on RunPod, each running one experiment. Monitors progress,
collects results, and terminates pods when done.

Usage:
    # Launch a batch of experiments
    python -m autoresearch.runpod_launcher launch --batch autoresearch/batches/batch1.json

    # Check status of running pods
    python -m autoresearch.runpod_launcher status

    # Download results from network volume
    python -m autoresearch.runpod_launcher results

    # Terminate all autoresearch pods
    python -m autoresearch.runpod_launcher cleanup
"""

import argparse
import json
import os
import sys
import time

try:
    import runpod
except ImportError:
    print("RunPod SDK not installed. Run: pip install runpod")
    sys.exit(1)


# Configuration
DEFAULT_GPU_TYPE = "NVIDIA RTX 4090"
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
DEFAULT_CLOUD_TYPE = "COMMUNITY"  # cheaper than SECURE
DEFAULT_CONTAINER_DISK_GB = 20
DEFAULT_VOLUME_GB = 50
POD_NAME_PREFIX = "autoresearch"


def load_api_key():
    """Load RunPod API key from env."""
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("ERROR: Set RUNPOD_API_KEY environment variable")
        sys.exit(1)
    runpod.api_key = key
    return key


def load_batch(batch_file):
    """Load experiment batch definition."""
    with open(batch_file) as f:
        return json.load(f)


def launch_batch(batch_file, network_volume_id, gpu_type=None, cloud_type=None):
    """Launch all experiments in a batch as parallel RunPod pods."""
    api_key = load_api_key()
    batch = load_batch(batch_file)

    gpu = gpu_type or batch.get("gpu_type", DEFAULT_GPU_TYPE)
    cloud = cloud_type or batch.get("cloud_type", DEFAULT_CLOUD_TYPE)
    repo_url = batch.get("repo_url", "")
    repo_branch = batch.get("repo_branch", "main")

    experiments = batch["experiments"]
    print(f"Launching {len(experiments)} experiments on {gpu} ({cloud})...")
    print(f"Network volume: {network_volume_id}")
    print()

    launched_pods = []

    for exp in experiments:
        run_name = exp["run_name"]
        experiment_config = exp["experiment_config"]
        max_epochs = exp.get("max_epochs", 200)
        patience = exp.get("patience", 50)

        pod_name = f"{POD_NAME_PREFIX}-{run_name}"

        env_vars = {
            "EXPERIMENT_CONFIG": experiment_config,
            "RUN_NAME": run_name,
            "MAX_EPOCHS": str(max_epochs),
            "PATIENCE": str(patience),
            "REPO_URL": repo_url,
            "REPO_BRANCH": repo_branch,
            "RUNPOD_API_KEY": api_key,
        }

        try:
            pod = runpod.create_pod(
                name=pod_name,
                image_name=DEFAULT_IMAGE,
                gpu_type_id=gpu,
                cloud_type=cloud,
                gpu_count=1,
                volume_in_gb=DEFAULT_VOLUME_GB,
                container_disk_in_gb=DEFAULT_CONTAINER_DISK_GB,
                network_volume_id=network_volume_id,
                docker_args=f"bash /runpod-volume/project/autoresearch/pod_start.sh",
                env=env_vars,
            )
            pod_id = pod["id"]
            launched_pods.append({"pod_id": pod_id, "run_name": run_name})
            print(f"  Launched: {run_name} -> pod {pod_id}")
        except Exception as e:
            print(f"  FAILED:   {run_name} -> {e}")

    # Save pod tracking info
    tracking_file = batch_file.replace(".json", "_pods.json")
    with open(tracking_file, "w") as f:
        json.dump({"pods": launched_pods, "batch_file": batch_file}, f, indent=2)

    print(f"\nLaunched {len(launched_pods)}/{len(experiments)} pods")
    print(f"Tracking file: {tracking_file}")
    print(f"\nMonitor with: python -m autoresearch.runpod_launcher status --tracking {tracking_file}")


def check_status(tracking_file=None):
    """Check status of all autoresearch pods."""
    load_api_key()
    pods = runpod.get_pods()

    autoresearch_pods = [
        p for p in pods
        if p.get("name", "").startswith(POD_NAME_PREFIX)
    ]

    if not autoresearch_pods:
        print("No autoresearch pods found.")
        return

    print(f"\n{'Pod ID':<15} {'Name':<40} {'Status':<15} {'GPU':<20}")
    print("-" * 90)

    for p in autoresearch_pods:
        runtime = p.get("runtime", {}) or {}
        gpu_info = ""
        gpus = runtime.get("gpus", [])
        if gpus:
            gpu_info = gpus[0].get("gpuUtilPercent", "?")
            gpu_info = f"{gpu_info}% util"

        print(
            f"{p['id']:<15} "
            f"{p.get('name', '???'):<40} "
            f"{p.get('desiredStatus', '???'):<15} "
            f"{gpu_info:<20}"
        )

    running = sum(1 for p in autoresearch_pods if p.get("desiredStatus") == "RUNNING")
    stopped = sum(1 for p in autoresearch_pods if p.get("desiredStatus") in ("EXITED", "STOPPED"))
    print(f"\nRunning: {running} | Stopped: {stopped} | Total: {len(autoresearch_pods)}")


def cleanup():
    """Terminate all stopped autoresearch pods."""
    load_api_key()
    pods = runpod.get_pods()

    autoresearch_pods = [
        p for p in pods
        if p.get("name", "").startswith(POD_NAME_PREFIX)
    ]

    if not autoresearch_pods:
        print("No autoresearch pods to clean up.")
        return

    for p in autoresearch_pods:
        status = p.get("desiredStatus", "")
        if status in ("EXITED", "STOPPED"):
            try:
                runpod.terminate_pod(p["id"])
                print(f"Terminated: {p['id']} ({p.get('name', '???')})")
            except Exception as e:
                print(f"Failed to terminate {p['id']}: {e}")
        else:
            print(f"Skipping {p['id']} ({p.get('name', '???')}) - status: {status}")


def main():
    parser = argparse.ArgumentParser(description="RunPod autoresearch launcher")
    subparsers = parser.add_subparsers(dest="command")

    # Launch
    launch_parser = subparsers.add_parser("launch", help="Launch experiment batch")
    launch_parser.add_argument("--batch", required=True, help="Batch definition JSON")
    launch_parser.add_argument("--volume-id", required=True, help="Network volume ID")
    launch_parser.add_argument("--gpu-type", default=None)
    launch_parser.add_argument("--cloud-type", default=None)

    # Status
    status_parser = subparsers.add_parser("status", help="Check pod status")
    status_parser.add_argument("--tracking", default=None)

    # Cleanup
    subparsers.add_parser("cleanup", help="Terminate stopped pods")

    args = parser.parse_args()

    if args.command == "launch":
        launch_batch(args.batch, args.volume_id, args.gpu_type, args.cloud_type)
    elif args.command == "status":
        check_status(args.tracking)
    elif args.command == "cleanup":
        cleanup()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
