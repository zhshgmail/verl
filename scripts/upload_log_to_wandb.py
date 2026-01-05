#!/usr/bin/env python3
"""
Parse VERL training logs and upload metrics to Weights & Biases.

Usage:
    python upload_log_to_wandb.py --log /path/to/log.log --run-name "baseline_test2" --project qerl
"""

import argparse
import re
import os
import wandb


def parse_log_line(line):
    """Parse a single log line and extract metrics."""
    # Remove ANSI color codes
    line = re.sub(r'\x1b\[[0-9;]*m', '', line)

    # Check if line contains step info
    if 'step:' not in line:
        return None

    metrics = {}

    # Extract step number - look for "step:N " pattern at start of metrics
    step_match = re.search(r'\bstep:(\d+)\b', line)
    if step_match:
        metrics['step'] = int(step_match.group(1))
    else:
        return None

    # Remove the prefix like "(TaskRunner pid=1210110) "
    line = re.sub(r'\([^)]+\)\s*', '', line)

    # Parse key:value pairs separated by " - "
    # Format: key:value - key:value - ...
    parts = line.split(' - ')

    for part in parts:
        part = part.strip()
        if ':' not in part:
            continue

        # Handle nested keys like val-core/openai/gsm8k/acc/mean@1
        key_value = part.split(':', 1)
        if len(key_value) != 2:
            continue

        key, value = key_value
        key = key.strip()
        value = value.strip()

        # Skip keys that look like log prefixes
        if 'pid=' in key or 'TaskRunner' in key:
            continue

        # Skip the standalone "step" key (we already captured it)
        if key == 'step':
            continue

        # Skip non-numeric values
        try:
            # Handle scientific notation
            value = float(value)
            metrics[key] = value
        except ValueError:
            continue

    return metrics if len(metrics) > 1 else None  # Must have step + at least one metric


def parse_log_file(log_path):
    """Parse entire log file and return list of metrics dicts."""
    all_metrics = []

    with open(log_path, 'r') as f:
        for line in f:
            metrics = parse_log_line(line)
            if metrics and 'step' in metrics:
                all_metrics.append(metrics)

    # Remove duplicates by step (keep last occurrence)
    seen_steps = {}
    for m in all_metrics:
        seen_steps[m['step']] = m

    return sorted(seen_steps.values(), key=lambda x: x['step'])


def upload_to_wandb(metrics_list, project, run_name, entity=None, tags=None, config=None):
    """Upload metrics to Weights & Biases."""

    # Initialize wandb run with extended timeout for slow connections
    run = wandb.init(
        project=project,
        name=run_name,
        entity=entity,
        tags=tags or [],
        config=config or {},
        reinit=True,
        settings=wandb.Settings(init_timeout=300)
    )

    print(f"Uploading {len(metrics_list)} steps to wandb...")

    for metrics in metrics_list:
        step = metrics.pop('step')
        wandb.log(metrics, step=step)
        print(f"  Logged step {step}: {len(metrics)} metrics")

    wandb.finish()
    print(f"Upload complete! View at: {run.url}")
    return run.url


def main():
    parser = argparse.ArgumentParser(description='Upload VERL logs to Weights & Biases')
    parser.add_argument('--log', required=True, help='Path to log file')
    parser.add_argument('--project', default='qerl', help='W&B project name')
    parser.add_argument('--run-name', required=True, help='Name for this run')
    parser.add_argument('--entity', default=None, help='W&B entity (team/user)')
    parser.add_argument('--tags', nargs='*', default=[], help='Tags for the run')
    parser.add_argument('--config', type=str, default=None, help='JSON config string')
    parser.add_argument('--dry-run', action='store_true', help='Parse only, do not upload')

    args = parser.parse_args()

    # Parse log file
    print(f"Parsing log file: {args.log}")
    metrics_list = parse_log_file(args.log)
    print(f"Found {len(metrics_list)} steps with metrics")

    if not metrics_list:
        print("No metrics found in log file!")
        return

    # Show sample
    print("\nSample metrics from first step:")
    for key, value in list(metrics_list[0].items())[:10]:
        print(f"  {key}: {value}")

    if args.dry_run:
        print("\nDry run mode - not uploading to wandb")
        return

    # Parse config if provided
    config = None
    if args.config:
        import json
        config = json.loads(args.config)

    # Upload to wandb
    upload_to_wandb(
        metrics_list,
        project=args.project,
        run_name=args.run_name,
        entity=args.entity,
        tags=args.tags,
        config=config
    )


if __name__ == '__main__':
    main()
