"""Offline CPU comparison reports. Existing sealed artifacts are immutable."""
import csv
import html
import json
from pathlib import Path
import subprocess
import sys

from .common import ROOT, PLOT_PYTHON, atomic_json, read_jsonl
from .cpu_compare_spec import EVENTS, analyze
from .reliability_report import STYLE


def read(path):
    return json.loads(Path(path).read_text())


def build_report(directory, charts=True):
    directory = Path(directory)
    if (directory / 'checksums.json').exists():
        raise ValueError('Sealed CPU comparison report is immutable')
    config = read(directory / 'config.json'); events, damaged = read_jsonl(directory / EVENTS)
    summary = analyze(events, config)
    telemetry, telemetry_errors = read_jsonl(directory / 'telemetry.jsonl')
    initialization = [dict(instance=e['instance'], configuration=e['configuration'],
                           cpu_initialization_ms=e['environment']['cpu_initialization_ms'])
                      for e in events if e['event'] == 'worker_ready']
    summary.update(event_errors=damaged, telemetry_errors=telemetry_errors,
        failures=[e for e in events if e['event'] == 'failure'], initialization=initialization,
        governor_transitions=[e for e in events if e['event'] == 'governor_transition'],
        telemetry={key: max((r[key] for r in telemetry if r.get(key) is not None), default=None)
                   for key in ('cpu_temp_c', 'host_cpu_percent', 'worker_tree_rss_bytes', 'worker_cpu_seconds')},
        throttle_flags=sorted({r['throttle_flags'] for r in telemetry if r.get('throttle_flags') is not None}),
        scope='Six tested streaming attention shapes only; individual requests and 16-call batches.',
        energy_measured=False,
        charts=dict(available=False, reason='disabled' if not charts else 'pending'))
    if damaged or summary['failures']:
        summary.update(complete=False, accepted=False, decision='incomplete')
    if not summary['complete'] and any(e['event'] == 'pilot_gate' and not e['fits'] for e in events):
        summary['decision'] = 'deferred_by_pilot'
    atomic_json(directory / 'summary.json', summary)

    with (directory / 'measurements.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('section', 'block', 'fixture', 'shape', 'configuration', 'repeat', 'position',
                         'label', 'call', 'job_started', 'full_ms', 'native_ms', 'verification_ms',
                         'observer_ms', 'cpu_seconds', 'correct', 'matches_warmup', 'output_sha256'))
        for event in events:
            if event['event'] != 'measurement':
                continue
            result = event['response']['result']
            for index, row in enumerate(result.get('requests', [result])):
                writer.writerow((event['section'], event['block'], event['fixture_id'], event['cell_id'],
                    event['configuration'], event['repeat'], event['position'], event['label'], index,
                    event['started'], event['total_ms'] if index == 0 else '', row['request_ms'],
                    (row['ended'] - row['computed']) * 1000, result['observer_ms'] if index == 0 else '',
                    row['cpu_seconds'], row['correct'], row['matches_warmup'], row['output_sha256']))
    if charts:
        try:
            completed = subprocess.run([str(PLOT_PYTHON if PLOT_PYTHON.exists() else Path(sys.executable)),
                '-B', '-m', 'efficiency.cpu_compare_report', str(directory.resolve())],
                cwd=ROOT, capture_output=True, text=True, timeout=120)
            available = completed.returncode == 0 and all((directory / name).is_file()
                for name in ('cpu-compare.png', 'cpu-compare.svg', 'equivalence.png', 'equivalence.svg'))
            summary['charts'] = dict(available=available,
                reason=None if available else 'Plotting unavailable: ' + (completed.stderr.strip()[-1000:] or 'missing chart output'))
        except (OSError, subprocess.TimeoutExpired) as exc:
            summary['charts'] = dict(available=False, reason=f'Plotting unavailable: {type(exc).__name__}: {exc}')
    atomic_json(directory / 'summary.json', summary)
    chart_html = ('<img src="cpu-compare.png" alt="Paired speedup confidence intervals">'
                  '<img src="equivalence.png" alt="Duplicate control equivalence intervals">'
                  if summary['charts']['available'] else '<p>' + html.escape(summary['charts']['reason']) + '</p>')
    (directory / 'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>CPU policy comparison</title>' + STYLE + '<h1>CPU policy comparison</h1><p>'
        + html.escape(summary['decision']) + '</p>'
        '<p>Baseline: ondemand governor and default OpenMP environment. Candidate: performance '
        'governor, bound threads and passive waiting. The native calculation and thread routing are identical.</p>'
        '<p>Times include submission through validated result. Initialization is reported separately. '
        'Each confidence interval resamples complete paired blocks. Repetitions and duplicate controls '
        'are averaged within each shape, with equal weight for every shape. The pilot is excluded. '
        'Calls within a batch are not independent statistical samples.</p>'
        '<p>Qualification requires at least 1.10× speedup and a paired 95% lower bound above 1 '
        'for both workloads, no shape more than 5% slower, and stable duplicate controls in both '
        'configurations. Results apply only to these six attention shapes. Energy was not measured. '
        'Application defaults remain unchanged.</p>' + chart_html
        + '<p><a href="measurements.csv">Measurements</a> · <a href="cpu-compare.jsonl">Raw evidence</a> · '
        '<a href="validation.json">Independent offline audit</a> · <a href="summary.json">JSON summary</a> · '
        '<a href="provenance.json">Historical selection</a> · <a href="manifest.json">Source hashes</a> · '
        '<a href="config.json">Configuration</a> · <a href="protocol.json">Frozen protocol</a></p>'
        '<pre>' + html.escape(json.dumps(summary, indent=2)) + '</pre></html>')
    return summary


def plots(directory):
    directory = Path(directory)
    if (directory / 'checksums.json').exists():
        raise ValueError('Sealed CPU comparison report is immutable')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    summary = read(directory / 'summary.json'); metrics = summary['metrics']
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, kind in zip(axes, ('single', 'batch')):
        metric = next((m for m in metrics if m['kind'] == kind), None)
        if metric is not None:
            entries = {'aggregate': metric['speedup'], **metric['per_cell_speedup']}
            for index, (name, value) in enumerate(entries.items()):
                low, high = value['interval']
                axis.plot([low, high], [index, index], color='steelblue')
                axis.scatter(value['estimate'], index, color='black')
            axis.set_yticks(range(len(entries)), list(entries))
        else:
            axis.text(.5, .5, 'No complete measurement matrix', ha='center', transform=axis.transAxes)
        axis.axvline(1, color='gray', linestyle=':')
        axis.axvline(1.10, color='green', linestyle='--')
        axis.set_title('Individual requests' if kind == 'single' else '16-call batches')
        axis.set_xlabel('Baseline / candidate time, paired 95% interval')
    figure.suptitle('CPU policy comparison: ' + summary['decision']); figure.tight_layout()
    figure.savefig(directory / 'cpu-compare.png', dpi=150); figure.savefig(directory / 'cpu-compare.svg'); plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    for row, kind in enumerate(('single', 'batch')):
        metric = next((m for m in metrics if m['kind'] == kind), None)
        for col, configuration in enumerate(('baseline', 'candidate')):
            axis = axes[row, col]
            if metric is not None:
                entries = metric['equivalence'][configuration]
                for index, (name, value) in enumerate(entries.items()):
                    low, high = value['interval']; axis.plot([low, high], [index, index], color='steelblue')
                    axis.scatter(value['estimate'], index, color='black')
                axis.set_yticks(range(len(entries)), list(entries))
            else:
                axis.text(.5, .5, 'No complete measurement matrix', ha='center', transform=axis.transAxes)
            axis.axvspan(.95, 1.05, color='green', alpha=.15); axis.axvline(1, color='gray', linestyle=':')
            axis.set_title(f'{configuration}: {kind} controls'); axis.set_xlabel('Duplicate B / A time, paired 95% interval')
    figure.tight_layout(); figure.savefig(directory / 'equivalence.png', dpi=150)
    figure.savefig(directory / 'equivalence.svg'); plt.close(figure)


if __name__ == '__main__':
    plots(sys.argv[1])
