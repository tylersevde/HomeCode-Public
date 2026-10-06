"""Offline, portable presentation of frozen roadmap results."""
import csv
import html
import json
from pathlib import Path
import subprocess

from .common import ROOT, PLOT_PYTHON, atomic_json, utc


def report(directory, summary, validation, disposition):
    directory = Path(directory)
    metrics = summary.get('metrics', [])
    fields = list(dict.fromkeys(key for row in metrics for key in row))
    with (directory / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ['status'])
        writer.writeheader()
        for row in metrics:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    chart = ''
    if metrics and summary.get('track') in ('gpu', 'npu'):
        try:
            subprocess.run([str(PLOT_PYTHON), '-B', str(ROOT / 'efficiency/completion_plot.py'), str(directory)],
                           check=True, capture_output=True, text=True, timeout=120)
            chart = '<figure><img src="results.svg" alt="Measured per-workload results"><figcaption>' \
                    'Frozen planned coverage; diagnostic results do not qualify the original task.</figcaption></figure>'
        except (OSError, subprocess.SubprocessError) as exc:
            atomic_json(directory / 'chart-failure.json', dict(error=str(exc), scientific_data_changed=False))
    escape = lambda value: html.escape(str(value))
    if summary.get('track') == 'gpu':
        columns = ('Shape', 'CPU ms', 'Earlier GPU ms', 'Stream64 ms', 'CPU / stream64', '95% interval', 'Frozen route')
        values = [[m['cell_id'], f"{m['cpu_mean_ms']:.3f}", f"{m['mean_ms']['gpu']:.3f}",
                   f"{m['mean_ms']['stream64']:.3f}", f"{m['stream64_cpu_speedup']['estimate']:.3f}",
                   m['stream64_cpu_speedup']['interval'], m['route']] for m in metrics]
        scope = 'Synthetic streaming attention, including transfers, synchronization and result validation. Ratios above one favor stream64 against the declared CPU route. Aggregate routing combines these fixed-shape measurements; dynamic dispatcher overhead was not measured. No LLM acceleration claim.'
    else:
        columns = ('Condition', 'Planned answers', 'Observed', 'Valid', 'Strictly correct', 'Per-size correct', 'Qualified')
        values = [[m['label'], m['planned'], m['observed'], m['valid'], m['correct'], m['per_size_correct'], m['qualified']] for m in metrics]
        scope = 'Original full-table four-turn qualification is separate from simplified diagnostic arms. Missing and quarantined turns remain in planned denominators.'
    table = '<table><thead><tr>' + ''.join('<th>' + escape(c) + '</th>' for c in columns) + '</tr></thead><tbody>'
    table += ''.join('<tr>' + ''.join('<td>' + escape(c) + '</td>' for c in row) + '</tr>' for row in values) + '</tbody></table>'
    payload = dict(disposition=disposition, summary=summary, validation=validation)
    raw = escape(json.dumps(payload, indent=2))
    (directory / 'report.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>Completion experiment evidence</title><style>body{max-width:85em;margin:2em auto;padding:0 1em;'
        'font:16px system-ui;color:#182c39}pre{white-space:pre-wrap}table{border-collapse:collapse;width:100%}'
        'th,td{padding:.6em;text-align:left;border-bottom:1px solid #ccd5da}img{max-width:100%}</style>'
        '<h1>' + escape(summary.get('track', 'Completion')) + ' — ' + escape(summary.get('stage', 'experiment')) + '</h1>'
        '<p><strong>Disposition: ' + escape(disposition) + '</strong>. Independent audit passed: ' + escape(validation.get('passed', False)) + '.</p>'
        '<p>' + scope + ' Application defaults remain unchanged.</p>'
        '<p><a href="metrics.csv">Metrics CSV</a> · <a href="summary.json">Summary JSON</a> · '
        '<a href="validation.json">Independent audit</a></p>' + table + chart + '<details><summary>Complete result and audit</summary><pre>' + raw + '</pre></details>')
    (directory / 'CONTEXT.txt').write_text(f'{utc()}\nDisposition: {disposition}\n'
        f'Audit passed: {validation.get("passed", False)}\nReport: report.html\nDefaults unchanged.\n')
