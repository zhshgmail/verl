#!/usr/bin/env python3
"""
Parse VERL training logs and export metrics to TensorBoard format.

Usage:
    python export_log_to_tensorboard.py --log /path/to/log.log --run-name "baseline_test2" --logdir ./tb_logs

    # Batch export:
    python export_log_to_tensorboard.py --batch-dir /tmp/aqn_logs --logdir ./tb_logs

Then view with:
    tensorboard --logdir ./tb_logs
"""

import argparse
import re
import os
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


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

    # Remove the prefix like "(TaskRunner pid=1210110) " at the start only
    # Be careful not to remove np.float64(...) values
    line = re.sub(r'^\s*\([^)]+\)\s*', '', line)
    line = re.sub(r'\[36m\([^)]+\)\[0m\s*', '', line)

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
            # Handle np.float64(...) format
            if value.startswith('np.float64('):
                value = value.replace('np.float64(', '').replace(')', '')
            elif value.startswith('np.int32('):
                value = value.replace('np.int32(', '').replace(')', '')
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


def export_to_tensorboard(metrics_list, logdir, run_name):
    """Export metrics to TensorBoard format."""

    run_dir = os.path.join(logdir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=run_dir)

    print(f"Exporting {len(metrics_list)} steps to TensorBoard at {run_dir}...")

    for metrics in metrics_list:
        step = metrics.pop('step')
        for key, value in metrics.items():
            # Convert metric names to TensorBoard-friendly format
            # Replace / with _ to avoid nested tags
            # Remove @N suffix (e.g., mean@1 -> mean)
            tag = key.replace('/', '_')
            if '@' in tag:
                tag = tag.split('@')[0]
            writer.add_scalar(tag, value, step)

    writer.close()
    print(f"Export complete! Run 'tensorboard --logdir {logdir}' to view.")
    return run_dir


def get_run_name_from_log(log_path):
    """Generate a run name from log filename."""
    filename = Path(log_path).stem
    return filename


def main():
    parser = argparse.ArgumentParser(description='Export VERL logs to TensorBoard')
    parser.add_argument('--log', help='Path to log file')
    parser.add_argument('--batch-dir', help='Directory containing multiple log files')
    parser.add_argument('--logdir', default='./tb_logs', help='TensorBoard log directory')
    parser.add_argument('--run-name', help='Name for this run (default: derived from filename)')
    parser.add_argument('--dry-run', action='store_true', help='Parse only, do not export')

    args = parser.parse_args()

    if not args.log and not args.batch_dir:
        parser.error('Either --log or --batch-dir must be provided')

    log_files = []

    if args.batch_dir:
        # Batch mode: process all .log files in directory
        batch_path = Path(args.batch_dir)
        log_files = list(batch_path.glob('*.log'))
        print(f"Found {len(log_files)} log files in {args.batch_dir}")
    else:
        log_files = [Path(args.log)]

    for log_path in log_files:
        print(f"\n{'='*60}")
        print(f"Processing: {log_path}")

        # Parse log file
        metrics_list = parse_log_file(str(log_path))
        print(f"Found {len(metrics_list)} steps with metrics")

        if not metrics_list:
            print("No metrics found in log file!")
            continue

        # Show sample
        print("Sample metrics from first step:")
        for key, value in list(metrics_list[0].items())[:5]:
            print(f"  {key}: {value}")

        if args.dry_run:
            print("Dry run mode - not exporting")
            continue

        # Determine run name
        run_name = args.run_name if args.run_name else get_run_name_from_log(log_path)

        # Export to TensorBoard
        export_to_tensorboard(metrics_list, args.logdir, run_name)

    print(f"\n{'='*60}")
    print(f"All exports complete!")
    print(f"View with: tensorboard --logdir {args.logdir}")


if __name__ == '__main__':
    main()
