#!/usr/bin/env python3
"""
Upload robustness testing results to Weights & Biases.

This creates a summary visualization of robustness testing across
different checkpoints and noise levels.
"""

import argparse
import wandb


def upload_robustness_results(project, entity):
    """Upload robustness testing results to wandb."""

    # E5b Robustness Results (matmul + Epoch-Aware AQN)
    # Checkpoint Step 58 (Epoch 1)
    e5b_step58 = {
        "experiment": "E5b",
        "description": "matmul + Epoch-Aware AQN",
        "checkpoint": "step_58",
        "epoch": 1,
        "noise_0pct": 79.00,
        "noise_5pct": 79.00,
        "noise_10pct": 78.00,
        "degradation_5pct": 0.00,
        "degradation_10pct": -1.00,
    }

    # E5b Checkpoint Step 116 (Epoch 2)
    e5b_step116 = {
        "experiment": "E5b",
        "description": "matmul + Epoch-Aware AQN",
        "checkpoint": "step_116",
        "epoch": 2,
        "noise_0pct": 77.00,
        "noise_5pct": 78.00,
        "noise_10pct": 77.50,
        "degradation_5pct": 1.00,
        "degradation_10pct": 0.50,
    }

    # E5d Robustness Results (ALL_OPS + Epoch-Aware AQN)
    # Checkpoint Step 58 (Epoch 1)
    e5d_step58 = {
        "experiment": "E5d",
        "description": "ALL_OPS + Epoch-Aware AQN",
        "checkpoint": "step_58",
        "epoch": 1,
        "noise_0pct": 76.50,
        "noise_5pct": 77.00,
        "noise_10pct": 76.50,
        "degradation_5pct": 0.50,
        "degradation_10pct": 0.00,
    }

    # E5d Checkpoint Step 116 (Epoch 2)
    e5d_step116 = {
        "experiment": "E5d",
        "description": "ALL_OPS + Epoch-Aware AQN",
        "checkpoint": "step_116",
        "epoch": 2,
        "noise_0pct": 74.50,
        "noise_5pct": 74.50,
        "noise_10pct": 74.50,
        "degradation_5pct": 0.00,
        "degradation_10pct": 0.00,
    }

    all_results = [e5b_step58, e5b_step116, e5d_step58, e5d_step116]

    # Upload each result as a separate run
    for result in all_results:
        run_name = f"{result['experiment']}-{result['checkpoint']}"

        run = wandb.init(
            project=project,
            name=run_name,
            entity=entity,
            tags=[
                result["experiment"],
                result["checkpoint"],
                f"epoch{result['epoch']}",
                "robustness",
            ],
            config={
                "experiment": result["experiment"],
                "description": result["description"],
                "checkpoint": result["checkpoint"],
                "epoch": result["epoch"],
            },
            reinit=True,
            settings=wandb.Settings(init_timeout=300),
        )

        # Log accuracy at different noise levels (simple metrics only)
        wandb.log(
            {
                "accuracy/noise_0pct": result["noise_0pct"],
                "accuracy/noise_5pct": result["noise_5pct"],
                "accuracy/noise_10pct": result["noise_10pct"],
                "degradation/noise_5pct": result["degradation_5pct"],
                "degradation/noise_10pct": result["degradation_10pct"],
            }
        )

        wandb.finish()
        print(f"Uploaded {run_name}: {run.url}")

    print(f"\nAll robustness results uploaded!")
    print(f"View project at: https://wandb.ai/{entity}/{project}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload robustness results to Weights & Biases"
    )
    parser.add_argument(
        "--project", default="aqn-robustness", help="W&B project name"
    )
    parser.add_argument("--entity", default="vaai", help="W&B entity (team/user)")

    args = parser.parse_args()

    upload_robustness_results(args.project, args.entity)


if __name__ == "__main__":
    main()
