import argparse
import json
import os
from pathlib import Path
import sys

from .common import ROOT, MODEL, PLOT_PYTHON, atomic_json, emit


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'completion':
        from .completion_service import main as completion_main
        return completion_main(argv[1:])
    if argv and argv[0] == 'cpu-compare':
        from .cpu_compare_service import main as compare_main
        return compare_main(argv[1:])
    if argv and argv[0] == 'cpu-policy':
        from .cpu_policy import main as policy_main
        return policy_main(argv[1:])
    parser = argparse.ArgumentParser(description='Pi 5 / AI HAT+ 2 caching experiment')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('completion', help='Supervise the separately budgeted completion roadmap')
    sub.add_parser('cpu-compare', help='Start, inspect or stop the bounded CPU comparison')
    sub.add_parser('cpu-policy', help='Temporarily run a command with a qualified CPU preset')
    preflight = sub.add_parser('preflight', help='Verify archive, runtime and model identity without inference')
    preflight.add_argument('--phase', choices=('cpu', 'hat', 'all'), default='all')
    preflight.add_argument('--model', default=str(MODEL))
    preflight.add_argument('--output', type=Path)
    run = sub.add_parser('run', help='Supervise a bounded measurement session')
    run.add_argument('--phase', choices=('cpu', 'hat', 'all'), default='all')
    run.add_argument('--profile', choices=('smoke', 'pilot', 'diagnostic', 'diagnostic-smoke', 'state-isolation', 'cache-validation', 'hybrid-validation', 'language-validation', 'language-strict-validation'), default='pilot')
    run.add_argument('--source-run', type=Path, help='Verified source artifacts for a diagnostic or validation profile')
    run.add_argument('--replay-fixture', help='Replay just one selected diagnostic fixture in both cache arms')
    run.add_argument('--replay-order', type=int, choices=(1, 2, 3), help='One-based cyclic order for --replay-fixture')
    run.add_argument('--state-case', choices=('small_failure','same_table_control','medium_failure','long_reverse'),
                     help='Replay one state-isolation case; keep all settings and state conditions')
    run.add_argument('--replay-run', type=Path, help='Replay a verified validation run; not fresh evidence')
    run.add_argument('--output', required=True, type=Path)
    run.add_argument('--max-seconds', type=float, default=3600)
    run.add_argument('--model', default=str(MODEL))
    improve = sub.add_parser('improve', help='Bounded CPU/Vulkan/HAT attention feedback experiment')
    improve.add_argument('--source-run', required=True, type=Path)
    improve.add_argument('--output', required=True, type=Path)
    improve.add_argument('--max-seconds', type=float, default=3600)
    improve.add_argument('--model', default=str(MODEL))
    improve.add_argument('--replay-run', type=Path, help='Replay frozen choices and routes; no adaptive search')
    improve.add_argument('--resume-run', type=Path, help='Continue a checksummed stage timeout within its unused total budget')
    coordinate = sub.add_parser('coordinate', help='Controlled CPU/GPU/NPU thread and process coordination experiment')
    coordinate.add_argument('--source-run', required=True, type=Path)
    coordinate.add_argument('--output', required=True, type=Path)
    coordinate.add_argument('--max-seconds', type=float, default=1800)
    coordinate.add_argument('--model', default=str(MODEL))
    adviser = sub.add_parser('validate-adviser', help='Bounded NPU adviser contract and CPU fallback validation')
    adviser.add_argument('--source-run', required=True, type=Path)
    adviser.add_argument('--output', required=True, type=Path)
    adviser.add_argument('--max-seconds', type=float, default=1800)
    adviser.add_argument('--model', default=str(MODEL))
    research = sub.add_parser('research', help='Five-stage bounded attention research campaign')
    research.add_argument('--stage', choices=('profile','gpu','context','combined','confirm'), required=True)
    research.add_argument('--campaign', type=Path, required=True)
    research.add_argument('--output', type=Path, required=True)
    research.add_argument('--max-seconds', type=float, default=7200)
    research.add_argument('--model', default=str(MODEL))
    research.add_argument('--secondary-model', type=Path)
    quality = sub.add_parser('quality', help='Four-hour bounded NPU quality development and confirmation')
    quality.add_argument('--stage', choices=('develop','confirm'), required=True)
    quality.add_argument('--campaign', type=Path, required=True)
    quality.add_argument('--output', type=Path, required=True)
    quality.add_argument('--max-seconds', type=float, default=7200)
    quality.add_argument('--model', default=str(MODEL))
    study = sub.add_parser('study', help='Eight-hour independent GPU streaming and NPU model studies')
    study.add_argument('--track', choices=('gpu-stream','npu-model'), required=True)
    study.add_argument('--stage', choices=('develop','confirm'), required=True)
    study.add_argument('--campaign', type=Path, required=True)
    study.add_argument('--output', type=Path, required=True)
    study.add_argument('--max-seconds', type=float, default=7200)
    study.add_argument('--model', default=str(MODEL))
    from .study_spec import LLAMA
    study.add_argument('--challenger-model', default=str(LLAMA))
    refine = sub.add_parser('refine', help='CPU scheduling calibration and Llama stop-policy study')
    from .refine_spec import CAPS
    refine.add_argument('--stage', choices=tuple(CAPS), required=True)
    refine.add_argument('--source-campaign', type=Path, required=True)
    refine.add_argument('--campaign', type=Path, required=True)
    refine.add_argument('--output', type=Path, required=True)
    refine.add_argument('--max-seconds', type=float)
    reliability = sub.add_parser('reliability', help='Sixteen-hour CPU/NPU reliability and gated GPU integration campaign')
    from .reliability_spec import CAPS as RELIABILITY_CAPS
    reliability.add_argument('--stage', choices=tuple(RELIABILITY_CAPS), required=True)
    reliability.add_argument('--source-campaign', type=Path, required=True)
    reliability.add_argument('--campaign', type=Path, required=True)
    reliability.add_argument('--output', type=Path, required=True)
    reliability.add_argument('--max-seconds', type=float)
    query = sub.add_parser('query', help='Retrieve one exact record and verify the HAT answer')
    query.add_argument('--facts', required=True, type=Path, help='JSON array of {item, color} records')
    query.add_argument('--item', required=True)
    query.add_argument('--output', required=True, type=Path)
    query.add_argument('--max-seconds', type=float, default=300)
    query.add_argument('--model', default=str(MODEL))
    ask = sub.add_parser('ask', help='Interpret a record question and execute it on the CPU')
    ask.add_argument('--facts', required=True, type=Path)
    ask.add_argument('--question', required=True)
    ask.add_argument('--engine', choices=('auto','cpu','hat','strict'), default='auto')
    ask.add_argument('--output', required=True, type=Path)
    ask.add_argument('--max-seconds', type=float, default=300)
    ask.add_argument('--model', default=str(MODEL))
    report = sub.add_parser('report', help='Analyze saved observations; does not access the HAT')
    report.add_argument('directory', type=Path)
    report.add_argument('--no-charts', action='store_true', help='Produce HTML/JSON/CSV without plotting')
    worker = sub.add_parser('_worker', help=argparse.SUPPRESS)
    worker.add_argument('--phase', choices=('cpu', 'hat'), required=True)
    worker.add_argument('--output', required=True, type=Path)
    worker.add_argument('--state-trial', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.command == 'reliability':
            from .reliability_campaign import run
            return run(args)
        if args.command == 'refine':
            from .refine_campaign import run
            return run(args)
        if args.command == 'study':
            from .study_campaign import run
            return run(args)
        if args.command == 'quality':
            from .quality_campaign import run
            return run(args)
        if args.command == 'research':
            from .research_campaign import run
            return run(args)
        if args.command == 'validate-adviser':
            if not 1 <= args.max_seconds <= 1800:
                parser.error('--max-seconds must be between 1 and 1800')
            args.profile, args.phase = 'adviser-validation', 'hat'
            from .runner import run
            code = run(args)
            from .adviser_report import build_report
            build_report(args.output.resolve(), charts=False)
            return code
        if args.command == 'coordinate':
            if not 1 <= args.max_seconds <= 1800:
                parser.error('--max-seconds must be between 1 and 1800')
            args.profile, args.phase = 'coordination', 'hat'
            from .runner import run
            code = run(args)
            from .coordination_report import build_report
            build_report(args.output.resolve(), charts=False)
            return code
        if args.command == 'improve':
            if not 1 <= args.max_seconds <= 3600:
                parser.error('--max-seconds must be between 1 and 3600')
            if args.resume_run and args.replay_run:
                parser.error('--resume-run and --replay-run are mutually exclusive')
            if args.resume_run:
                from .feedback_protocol import resume_budget
                args.resume_budget = resume_budget(args.resume_run)
                args.max_seconds = min(args.max_seconds, args.resume_budget['remaining_seconds'])
                prior_config = json.loads((args.resume_run/'config.json').read_text())
                if prior_config.get('replay_run'):
                    args.replay_run = Path(prior_config['replay_run'])
            from .attention_native import build
            build()
            args.profile, args.phase = 'feedback-attention', 'hat'
            from .runner import run
            return run(args)
        if args.command == 'preflight':
            from .runner import inventory
            os.environ.setdefault('HAILORT_LOGGER_PATH', '/tmp')
            value = inventory(args.phase, args.model)
            if args.output:
                atomic_json(args.output, value)
            print(json.dumps(value, indent=2))
            return 0
        if args.command == 'ask':
            if not 1 <= args.max_seconds <= 3600:
                parser.error('--max-seconds must be between 1 and 3600')
            from .hybrid import FactIndex
            from .language import needs_hat, validate_question
            args.ask_records = json.loads(args.facts.read_text())
            FactIndex(args.ask_records)
            validate_question(args.question)
            args.profile = 'language-query'
            args.phase = 'hat' if needs_hat(args.question,args.engine) else 'cpu'
            from .runner import run
            code = run(args)
            from .language_report import build_report
            build_report(args.output.resolve(),charts=False)
            if (args.output/'result.json').exists():
                print((args.output/'result.json').read_text())
            return code
        if args.command == 'query':
            if not 1 <= args.max_seconds <= 3600:
                parser.error('--max-seconds must be between 1 and 3600')
            from .hybrid import FactIndex
            records = json.loads(args.facts.read_text())
            index = FactIndex(records)
            found = index.lookup(args.item)
            args.query_records = records
            args.profile = 'hybrid-query'
            args.phase = 'hat' if found else 'cpu'
            from .runner import run
            code = run(args)
            from .hybrid_report import build_report
            build_report(args.output.resolve(), charts=False)
            result = args.output / 'result.json'
            if result.exists():
                print(result.read_text())
            return code
        if args.command == 'run':
            if not 1 <= args.max_seconds <= 3600:
                parser.error('--max-seconds must be between 1 and 3600')
            if args.profile.startswith('diagnostic') or args.profile in ('state-isolation', 'cache-validation', 'hybrid-validation', 'language-validation', 'language-strict-validation'):
                if args.phase != 'hat' or not args.source_run:
                    parser.error('This profile requires --phase hat and --source-run')
                if not (args.source_run / 'checksums.json').is_file():
                    parser.error('--source-run must contain a checksummed completed source run')
            elif args.source_run:
                parser.error('--source-run requires a diagnostic or validation profile')
            if args.replay_fixture or args.replay_order:
                if args.profile != 'diagnostic' or not (args.replay_fixture and args.replay_order):
                    parser.error('Replay requires --profile diagnostic, --replay-fixture and --replay-order')
            if args.state_case and args.profile != 'state-isolation':
                parser.error('--state-case requires --profile state-isolation')
            if args.replay_run and args.profile not in ('cache-validation', 'hybrid-validation', 'language-validation', 'language-strict-validation'):
                parser.error('--replay-run requires a validation profile')
            from .runner import run
            return run(args)
        if args.command == '_worker':
            import importlib
            config = json.loads((args.output / 'config.json').read_text())
            module = 'diagnostic' if config['profile'].startswith('diagnostic') else args.phase
            if config['profile'] == 'state-isolation':
                module = 'state_isolation'
            elif config['profile'] == 'cache-validation':
                module = 'cache_validation'
            elif config['profile'].startswith('hybrid-'):
                module = 'hybrid'
            elif config['profile'] == 'language-strict-validation':
                module = 'strict_protocol'
            elif config['profile'] == 'feedback-attention':
                module = 'feedback_protocol'
            elif config['profile'] == 'coordination':
                module = 'coordination'
            elif config['profile'] == 'adviser-validation':
                module = 'adviser_protocol'
            elif config['profile'] == 'research':
                module = 'research_protocol'
            elif config['profile'] == 'reliability':
                module = 'reliability_protocol'
            elif config['profile'] == 'cpu-compare':
                module = 'cpu_compare_protocol'
            elif config['profile'] == 'completion':
                from .completion_service import protocol_module
                module = protocol_module(config['track'])
            elif config['profile'] == 'refine':
                module = 'refine_protocol'
            elif config['profile'] == 'study':
                module = 'study_protocol'
            elif config['profile'] == 'quality':
                module = 'quality_protocol'
            elif config['profile'].startswith('language-'):
                module = 'language_protocol'
            try:
                implementation = importlib.import_module(f'efficiency.{module}')
                if args.state_trial:
                    if module != 'state_isolation':
                        raise ValueError('--state-trial requires state-isolation')
                    implementation.run_fresh(args.output, config, args.state_trial)
                else:
                    implementation.run(args.output, config)
            except Exception as exc:
                emit(args.output / f'{module}.jsonl', 'failure', error=f'{type(exc).__name__}: {exc}')
                raise
            return 0
        if args.command == 'report':
            if not args.no_charts:
                try:
                    import matplotlib
                except ImportError:
                    if PLOT_PYTHON.exists() and Path(sys.executable).absolute() != PLOT_PYTHON:
                        os.execv(str(PLOT_PYTHON), [str(PLOT_PYTHON), str(ROOT / 'experiment.py'),
                                                  'report', str(args.directory.resolve())])
                    raise RuntimeError('Matplotlib is unavailable; use --no-charts or install it in an analysis environment')
            config = json.loads((args.directory / 'config.json').read_text())
            if config['profile'] == 'state-isolation':
                from .state_report import build_report
            elif config['profile'] == 'feedback-attention':
                from .feedback_report import build_report
            elif config['profile'] == 'coordination':
                from .coordination_report import build_report
            elif config['profile'] == 'adviser-validation':
                from .adviser_report import build_report
            elif config['profile'] == 'research':
                if (args.directory/'validation.json').exists():
                    raise ValueError('Sealed research reports are immutable; open report.html or use the offline audit')
                from .research_report import build_report
            elif config['profile'] == 'study':
                from .study_report import build_report
            elif config['profile'] == 'reliability':
                from .reliability_report import build_report
            elif config['profile'] == 'cpu-compare':
                from .cpu_compare_report import build_report
            elif config['profile'] == 'refine':
                from .refine_report import build_report
            elif config['profile'] == 'quality':
                from .quality_report import build_report
            elif config['profile'] == 'cache-validation':
                from .cache_report import build_report
            elif config['profile'].startswith('hybrid-'):
                from .hybrid_report import build_report
            elif config['profile'] == 'language-strict-validation':
                from .strict_report import build_report
            elif config['profile'].startswith('language-'):
                from .language_report import build_report
            elif config['profile'].startswith('diagnostic'):
                from .diagnostic_report import build_report
            else:
                from .report import build_report
            build_report(args.directory.resolve(), charts=not args.no_charts)
            return 0
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
