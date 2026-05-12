#!/usr/bin/env python3
"""
Run the complete GAD pipeline with all stages.

Usage:
    python scripts/run_pipeline.py --config configs/pipeline.yaml
    python scripts/run_pipeline.py --config configs/pipeline.yaml --start-stage warmup
"""

import argparse
import os
import subprocess
import sys

import yaml

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="Run the complete GAD pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline.yaml",
        help="Path to the pipeline configuration file",
    )
    parser.add_argument(
        "--start-stage",
        type=str,
        default=None,
        help="Stage to start from (skip earlier stages)",
    )
    parser.add_argument(
        "--end-stage",
        type=str,
        default=None,
        help="Stage to end at (skip later stages)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load pipeline configuration
    with open(args.config, "r") as f:
        pipeline_config = yaml.safe_load(f)

    stages = pipeline_config.get("pipeline", {}).get("stages", [])

    if not stages:
        print("Error: No stages defined in pipeline configuration")
        sys.exit(1)

    print(f"Found {len(stages)} stages in pipeline:")
    for i, stage in enumerate(stages):
        print(f"  {i+1}. {stage['name']}: {stage.get('description', '')}")

    # Determine start and end indices
    start_idx = 0
    end_idx = len(stages)

    if args.start_stage:
        for i, stage in enumerate(stages):
            if stage["name"] == args.start_stage:
                start_idx = i
                break
        else:
            print(f"Error: Start stage '{args.start_stage}' not found")
            sys.exit(1)

    if args.end_stage:
        for i, stage in enumerate(stages):
            if stage["name"] == args.end_stage:
                end_idx = i + 1
                break
        else:
            print(f"Error: End stage '{args.end_stage}' not found")
            sys.exit(1)

    # Run each stage
    for i in range(start_idx, end_idx):
        stage = stages[i]
        stage_name = stage["name"]
        config_path = stage["config"]

        print(f"\n{'='*60}")
        print(f"Running stage {i+1}/{len(stages)}: {stage_name}")
        print(f"{'='*60}")

        # Build command
        cmd = [
            sys.executable,
            "scripts/run_stage.py",
            "--stage", stage_name,
            "--config", config_path,
        ]

        if "checkpoint_input" in stage:
            cmd.extend(["--checkpoint_input", stage["checkpoint_input"]])

        if "checkpoint_output" in stage:
            cmd.extend(["--checkpoint_output", stage["checkpoint_output"]])

        print(f"Command: {' '.join(cmd)}")

        if args.dry_run:
            print("(dry run - skipping execution)")
            continue

        # Run the stage
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        if result.returncode != 0:
            print(f"Error: Stage {stage_name} failed with return code {result.returncode}")
            sys.exit(1)

        print(f"Stage {stage_name} completed successfully!")

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
