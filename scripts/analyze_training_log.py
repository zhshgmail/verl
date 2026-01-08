#!/usr/bin/env python3
"""
Training Log Analyzer

Parses veRL training logs to extract key metrics and detect anomalies
like model collapse (entropy spikes, score crashes).

Usage:
    python scripts/analyze_training_log.py /path/to/training.log
    python scripts/analyze_training_log.py /path/to/training.log --plot
"""

import re
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class StepMetrics:
    """Metrics for a single training step."""
    step: int
    score: Optional[float] = None
    entropy: Optional[float] = None
    reward: Optional[float] = None
    kl: Optional[float] = None
    loss: Optional[float] = None
    response_length: Optional[float] = None
    max_length_ratio: Optional[float] = None
    sigma: Optional[float] = None
    sigma_id: Optional[int] = None
    timestamp: Optional[str] = None


@dataclass
class TrainingAnalysis:
    """Analysis results for training log."""
    total_steps: int
    final_score: Optional[float]
    max_score: Optional[float]
    collapse_detected: bool
    collapse_step: Optional[int]
    collapse_reason: Optional[str]
    metrics_by_step: List[StepMetrics]
    anomalies: List[Dict[str, Any]]
    summary: Dict[str, Any]


class TrainingLogAnalyzer:
    """Analyzes veRL training logs."""

    # Regex patterns for parsing log lines
    PATTERNS = {
        'step': r'step[:\s]+(\d+)',
        'score': r'(?:score|val_accuracy)[:\s]+([\d.]+)%?',
        'entropy': r'entropy[:\s]+([\d.]+)',
        'reward': r'reward[:\s]+([\d.-]+)',
        'kl': r'kl[:\s]+([\d.]+)',
        'loss': r'(?:actor_)?loss[:\s]+([\d.]+)',
        'response_length': r'response_length[:\s]+([\d.]+)',
        'max_length': r'max_length[:\s]+([\d.]+)%?',
        'sigma': r'[Ss]igma[:\s]+([\d.]+)',
        'sigma_id': r'[Ss]igma\s*ID[:\s]+(\d+)',
        'aqn': r'\[AQN\]',
    }

    # Thresholds for anomaly detection
    ENTROPY_SPIKE_THRESHOLD = 2.0  # Entropy > 2.0 indicates collapse
    SCORE_CRASH_THRESHOLD = 0.3    # Score drop > 30% indicates crash
    MAX_LENGTH_THRESHOLD = 0.8     # >80% hitting max length is suspicious

    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.metrics: List[StepMetrics] = []
        self.raw_lines: List[str] = []

    def parse_log(self) -> List[StepMetrics]:
        """Parse training log file."""
        print(f"Parsing log: {self.log_path}")

        if not self.log_path.exists():
            raise FileNotFoundError(f"Log file not found: {self.log_path}")

        with open(self.log_path, 'r') as f:
            self.raw_lines = f.readlines()

        current_step = None
        current_metrics = {}

        for line in self.raw_lines:
            line = line.strip()

            # Extract step number
            step_match = re.search(self.PATTERNS['step'], line, re.IGNORECASE)
            if step_match:
                # Save previous step if exists
                if current_step is not None and current_metrics:
                    self.metrics.append(StepMetrics(
                        step=current_step,
                        **current_metrics
                    ))
                current_step = int(step_match.group(1))
                current_metrics = {}

            # Extract metrics
            for metric_name, pattern in self.PATTERNS.items():
                if metric_name in ['step', 'aqn']:
                    continue

                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    value = match.group(1)
                    try:
                        if metric_name == 'sigma_id':
                            current_metrics[metric_name] = int(value)
                        else:
                            current_metrics[metric_name] = float(value)
                    except ValueError:
                        pass

        # Save last step
        if current_step is not None and current_metrics:
            self.metrics.append(StepMetrics(
                step=current_step,
                **current_metrics
            ))

        print(f"Parsed {len(self.metrics)} steps")
        return self.metrics

    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in training metrics."""
        anomalies = []

        prev_score = None
        prev_entropy = None

        for metrics in self.metrics:
            step = metrics.step

            # Check for entropy spike
            if metrics.entropy is not None:
                if metrics.entropy > self.ENTROPY_SPIKE_THRESHOLD:
                    anomalies.append({
                        'step': step,
                        'type': 'entropy_spike',
                        'value': metrics.entropy,
                        'threshold': self.ENTROPY_SPIKE_THRESHOLD,
                        'severity': 'critical' if metrics.entropy > 5.0 else 'warning',
                    })

                # Check for sudden entropy increase
                if prev_entropy is not None:
                    entropy_increase = metrics.entropy - prev_entropy
                    if entropy_increase > 1.0:
                        anomalies.append({
                            'step': step,
                            'type': 'sudden_entropy_increase',
                            'value': entropy_increase,
                            'from': prev_entropy,
                            'to': metrics.entropy,
                            'severity': 'warning',
                        })

                prev_entropy = metrics.entropy

            # Check for score crash
            if metrics.score is not None:
                if prev_score is not None and prev_score > 10:
                    score_drop = (prev_score - metrics.score) / prev_score
                    if score_drop > self.SCORE_CRASH_THRESHOLD:
                        anomalies.append({
                            'step': step,
                            'type': 'score_crash',
                            'value': score_drop,
                            'from': prev_score,
                            'to': metrics.score,
                            'severity': 'critical',
                        })

                prev_score = metrics.score

            # Check for max length ratio
            if metrics.max_length_ratio is not None:
                if metrics.max_length_ratio > self.MAX_LENGTH_THRESHOLD:
                    anomalies.append({
                        'step': step,
                        'type': 'max_length_spike',
                        'value': metrics.max_length_ratio,
                        'threshold': self.MAX_LENGTH_THRESHOLD,
                        'severity': 'warning',
                    })

        return anomalies

    def detect_collapse(self) -> Tuple[bool, Optional[int], Optional[str]]:
        """Detect if model collapsed during training."""
        for i, metrics in enumerate(self.metrics):
            # Check for entropy-based collapse
            if metrics.entropy is not None and metrics.entropy > 5.0:
                # Verify it's not just noise by checking subsequent steps
                if i + 1 < len(self.metrics):
                    next_entropy = self.metrics[i + 1].entropy
                    if next_entropy is not None and next_entropy > 3.0:
                        return True, metrics.step, f"Entropy spike to {metrics.entropy:.2f}"

            # Check for score-based collapse
            if metrics.score is not None and metrics.score < 5:
                # Check if previous step had reasonable score
                if i > 0:
                    prev_score = self.metrics[i - 1].score
                    if prev_score is not None and prev_score > 50:
                        return True, metrics.step, f"Score crashed from {prev_score:.1f}% to {metrics.score:.1f}%"

        return False, None, None

    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.metrics:
            return {'error': 'No metrics parsed'}

        scores = [m.score for m in self.metrics if m.score is not None]
        entropies = [m.entropy for m in self.metrics if m.entropy is not None]

        summary = {
            'total_steps': len(self.metrics),
            'final_step': self.metrics[-1].step if self.metrics else None,
        }

        if scores:
            summary['score'] = {
                'initial': scores[0] if scores else None,
                'final': scores[-1] if scores else None,
                'max': max(scores),
                'min': min(scores),
                'mean': sum(scores) / len(scores),
            }

        if entropies:
            summary['entropy'] = {
                'initial': entropies[0] if entropies else None,
                'final': entropies[-1] if entropies else None,
                'max': max(entropies),
                'min': min(entropies),
                'mean': sum(entropies) / len(entropies),
            }

        # Check for AQN
        aqn_lines = [l for l in self.raw_lines if re.search(self.PATTERNS['aqn'], l)]
        summary['aqn_enabled'] = len(aqn_lines) > 0
        summary['aqn_log_count'] = len(aqn_lines)

        return summary

    def analyze(self) -> TrainingAnalysis:
        """Run full analysis on training log."""
        self.parse_log()
        anomalies = self.detect_anomalies()
        collapse_detected, collapse_step, collapse_reason = self.detect_collapse()
        summary = self.generate_summary()

        scores = [m.score for m in self.metrics if m.score is not None]

        return TrainingAnalysis(
            total_steps=len(self.metrics),
            final_score=scores[-1] if scores else None,
            max_score=max(scores) if scores else None,
            collapse_detected=collapse_detected,
            collapse_step=collapse_step,
            collapse_reason=collapse_reason,
            metrics_by_step=self.metrics,
            anomalies=anomalies,
            summary=summary,
        )


def print_analysis(analysis: TrainingAnalysis):
    """Print analysis results."""
    print(f"\n{'='*60}")
    print("TRAINING LOG ANALYSIS")
    print(f"{'='*60}")

    print(f"\nSummary:")
    print(f"  Total steps: {analysis.total_steps}")
    print(f"  Final score: {analysis.final_score:.2f}%" if analysis.final_score else "  Final score: N/A")
    print(f"  Max score: {analysis.max_score:.2f}%" if analysis.max_score else "  Max score: N/A")

    if analysis.collapse_detected:
        print(f"\n  ⚠️  MODEL COLLAPSE DETECTED at step {analysis.collapse_step}")
        print(f"      Reason: {analysis.collapse_reason}")
    else:
        print(f"\n  ✓ No collapse detected")

    if analysis.anomalies:
        print(f"\nAnomalies ({len(analysis.anomalies)}):")
        for anomaly in analysis.anomalies[:10]:  # Show first 10
            severity = "⚠️" if anomaly['severity'] == 'warning' else "🚨"
            print(f"  {severity} Step {anomaly['step']}: {anomaly['type']} = {anomaly['value']:.4f}")

        if len(analysis.anomalies) > 10:
            print(f"  ... and {len(analysis.anomalies) - 10} more")

    # Show score progression
    if analysis.metrics_by_step:
        scores_with_step = [(m.step, m.score) for m in analysis.metrics_by_step if m.score is not None]
        if scores_with_step:
            print(f"\nScore Progression:")
            # Show every 10th step or key steps
            key_steps = scores_with_step[::max(1, len(scores_with_step)//10)]
            for step, score in key_steps:
                bar = "█" * int(score / 5) if score else ""
                print(f"  Step {step:4d}: {score:6.2f}% {bar}")


def plot_analysis(analysis: TrainingAnalysis, output_path: Optional[str] = None):
    """Plot training metrics (optional, requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Extract data
    steps = [m.step for m in analysis.metrics_by_step]
    scores = [m.score for m in analysis.metrics_by_step]
    entropies = [m.entropy for m in analysis.metrics_by_step]
    sigmas = [m.sigma for m in analysis.metrics_by_step]

    # Plot score
    ax1 = axes[0, 0]
    valid_scores = [(s, sc) for s, sc in zip(steps, scores) if sc is not None]
    if valid_scores:
        ax1.plot([x[0] for x in valid_scores], [x[1] for x in valid_scores], 'b-', label='Score')
        if analysis.collapse_step:
            ax1.axvline(x=analysis.collapse_step, color='r', linestyle='--', label='Collapse')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Score (%)')
    ax1.set_title('Training Score')
    ax1.legend()
    ax1.grid(True)

    # Plot entropy
    ax2 = axes[0, 1]
    valid_entropies = [(s, e) for s, e in zip(steps, entropies) if e is not None]
    if valid_entropies:
        ax2.plot([x[0] for x in valid_entropies], [x[1] for x in valid_entropies], 'r-', label='Entropy')
        ax2.axhline(y=2.0, color='orange', linestyle='--', label='Warning threshold')
        if analysis.collapse_step:
            ax2.axvline(x=analysis.collapse_step, color='r', linestyle='--', label='Collapse')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Entropy')
    ax2.set_title('Response Entropy')
    ax2.legend()
    ax2.grid(True)

    # Plot sigma
    ax3 = axes[1, 0]
    valid_sigmas = [(s, sg) for s, sg in zip(steps, sigmas) if sg is not None]
    if valid_sigmas:
        ax3.plot([x[0] for x in valid_sigmas], [x[1] for x in valid_sigmas], 'g-', label='Sigma')
        if analysis.collapse_step:
            ax3.axvline(x=analysis.collapse_step, color='r', linestyle='--', label='Collapse')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Sigma')
    ax3.set_title('AQN Noise Level')
    ax3.legend()
    ax3.grid(True)

    # Anomaly timeline
    ax4 = axes[1, 1]
    if analysis.anomalies:
        anomaly_steps = [a['step'] for a in analysis.anomalies]
        anomaly_types = [a['type'] for a in analysis.anomalies]
        colors = ['red' if a['severity'] == 'critical' else 'orange' for a in analysis.anomalies]
        ax4.scatter(anomaly_steps, range(len(anomaly_steps)), c=colors, s=50)
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Anomaly Index')
        ax4.set_title(f'Anomalies ({len(analysis.anomalies)} total)')
    else:
        ax4.text(0.5, 0.5, 'No anomalies detected', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Anomalies')
    ax4.grid(True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze veRL training log")
    parser.add_argument("log_path", type=str, help="Path to training log file")
    parser.add_argument("--plot", action="store_true", help="Generate plot")
    parser.add_argument("--plot-output", type=str, help="Save plot to file")
    parser.add_argument("--output", type=str, help="Save analysis to JSON file")

    args = parser.parse_args()

    analyzer = TrainingLogAnalyzer(args.log_path)
    analysis = analyzer.analyze()

    print_analysis(analysis)

    if args.plot or args.plot_output:
        plot_analysis(analysis, args.plot_output)

    if args.output:
        # Convert to dict for JSON
        result = {
            'total_steps': analysis.total_steps,
            'final_score': analysis.final_score,
            'max_score': analysis.max_score,
            'collapse_detected': analysis.collapse_detected,
            'collapse_step': analysis.collapse_step,
            'collapse_reason': analysis.collapse_reason,
            'anomalies': analysis.anomalies,
            'summary': analysis.summary,
            'metrics': [
                {
                    'step': m.step,
                    'score': m.score,
                    'entropy': m.entropy,
                    'sigma': m.sigma,
                }
                for m in analysis.metrics_by_step
            ],
        }
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nAnalysis saved to: {args.output}")

    # Exit code based on collapse
    sys.exit(1 if analysis.collapse_detected else 0)


if __name__ == "__main__":
    main()
