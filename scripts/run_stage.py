#!/usr/bin/env python3
"""
Run a single training stage of the GAD pipeline.

Usage:
    python scripts/run_stage.py --stage seqkd --config configs/seqkd.yaml
    python scripts/run_stage.py --stage warmup --config configs/warmup.yaml --checkpoint_input checkpoints/seqkd
    python scripts/run_stage.py --stage gad --config configs/gad.yaml --checkpoint_input checkpoints/warmup
    python scripts/run_stage.py --stage eval --config configs/eval.yaml --checkpoint_input checkpoints/gad
"""

import argparse
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single GAD training stage")
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["seqkd", "warmup", "gad", "eval"],
        help="Training stage to run",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the stage configuration file",
    )
    parser.add_argument(
        "--checkpoint_input",
        type=str,
        default=None,
        help="Path to input checkpoint (for warmup/gad/eval stages)",
    )
    parser.add_argument(
        "--checkpoint_output",
        type=str,
        default=None,
        help="Path to output checkpoint directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Running GAD stage: {args.stage}")
    print(f"Configuration: {args.config}")

    if args.checkpoint_input:
        print(f"Input checkpoint: {args.checkpoint_input}")

    if args.checkpoint_output:
        print(f"Output checkpoint: {args.checkpoint_output}")

    # Import and run the training script
    # Note: This is a placeholder. The actual implementation would integrate
    # with the existing training infrastructure.
    from omegaconf import OmegaConf

    # Load configuration
    config = OmegaConf.load(args.config)

    # Set training stage in config
    if "trainer" not in config:
        config.trainer = {}
    config.trainer.training_stage = args.stage

    # Set checkpoint paths if provided
    if args.checkpoint_input:
        config.trainer.resume_from_path = args.checkpoint_input
        config.trainer.resume_mode = "resume_path"

    if args.checkpoint_output:
        config.trainer.default_local_dir = args.checkpoint_output

    print("Configuration loaded successfully!")
    print(OmegaConf.to_yaml(config))

    # TODO: Integrate with actual training entry point
    # from verl.trainer.main_ppo import main
    # main(config)


if __name__ == "__main__":
    main()
