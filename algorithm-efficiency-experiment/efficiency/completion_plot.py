"""Standalone offline plotter, run with the established plotting environment."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(directory):
    directory = Path(directory)
    summary = json.loads((directory / 'summary.json').read_text())
    metrics = summary['metrics']
    fig, axis = plt.subplots(figsize=(11, max(4, .47 * len(metrics) + 1)))
    positions = np.arange(len(metrics))
    if summary['track'] == 'gpu':
        estimates = np.array([m['stream64_cpu_speedup']['estimate'] for m in metrics])
        bounds = np.array([m['stream64_cpu_speedup']['interval'] for m in metrics])
        axis.errorbar(estimates, positions, xerr=np.maximum(0, np.stack([estimates - bounds[:, 0], bounds[:, 1] - estimates])),
                      fmt='o', color='#126c82', capsize=3)
        axis.axvline(1, color='#a94242', linestyle='--')
        axis.set_xscale('log')
        axis.set_yticks(positions, [m['cell_id'] for m in metrics])
        axis.set_xlabel('CPU time / stream64 time; paired 95% interval (above 1 favors GPU)')
    else:
        valid = [m['valid'] / m['planned'] for m in metrics]
        correct = [m['correct'] / m['planned'] for m in metrics]
        axis.barh(positions - .18, valid, height=.35, color='#accdd0', label='Valid completions')
        axis.barh(positions + .18, correct, height=.35, color='#126c82', label='Strictly correct')
        axis.set_yticks(positions, [m['label'].replace('_', ' ') for m in metrics])
        axis.set_xlim(0, 1.05)
        axis.set_xlabel('Fraction of all planned answers, including quarantined turns')
        axis.legend(loc='lower right')
    axis.invert_yaxis()
    axis.set_title(f"{summary['track'].upper()} {summary['stage']}: {summary['decision'].replace('_', ' ')}")
    axis.grid(axis='x', alpha=.2)
    fig.tight_layout()
    for extension in ('svg', 'png'):
        fig.savefig(directory / ('results.' + extension), dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main(sys.argv[1])
