"""Fail-closed adviser parsing, verified stop control, and explicit CPU fallback."""
import math
import re
import time

from .coordination_workers import HatWorker, owner
from .diagnostic import reset_context
from .feedback_spec import CATALOG, deterministic_choice
from .hybrid import generate_checked


def validate_case(case):
    p=case['payload'];eligible=p['eligible'];tested=p['tested'];history=p['development_ms']
    if (len(eligible)!=len(set(eligible)) or len(tested)!=len(set(tested)) or
        set(eligible)&set(tested) or set(eligible)|set(tested)!=set(CATALOG) or
        set(history)!=set(tested) or set(p['configurations'])!=set(eligible)):
        raise ValueError('Invalid eligible/tested partition')
    if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or v<=0 for v in history.values()):
        raise ValueError('Development times must be finite and positive')
    if any(p['configurations'][c]!=CATALOG[c] for c in eligible):raise ValueError('Candidate description differs')
    return p


def cpu_choice(case):
    p=validate_case(case)
    history=[dict(candidate=c,development_total_ms=ms) for c,ms in p['development_ms'].items()]
    return deterministic_choice(history,p['eligible'])


def parse_response(raw, eligible, original_stops, bounded=False):
    if raw.get('error'):return None,None,None,'generation_error'
    if (raw.get('completion_status')!='LOGICAL_END_OF_GENERATION' or
        not raw.get('token_ledger_valid') or raw.get('context_before')!=0 or
        raw.get('context_after')!=raw.get('expected_context_after')):
        return None,None,None,'incomplete_or_invalid_context'
    output=raw['output']
    terminal=next((s for s in sorted(original_stops,key=len,reverse=True) if output.endswith(s)),None)
    if terminal:
        body=output[:-len(terminal)].strip();boundary='model_terminal'
    elif bounded and re.fullmatch(r'\s*C[0-7]\s*',output):
        body=output.strip();boundary='candidate_stop'
    else:return None,None,None,'missing_or_suppressed_terminal'
    if re.fullmatch(r'C[0-7]',body):
        choice=body;normalization=None
    elif terminal and re.fullmatch(r'C[0-7]\.',body):
        choice=body[:-1];normalization='remove_one_trailing_period'
    else:return None,None,boundary,'malformed_proposal'
    if choice not in eligible:return None,normalization,boundary,'ineligible_or_repeated_proposal'
    return choice,normalization,boundary,None


def decide(case, raw=None, original_stops=(), bounded=False):
    p=validate_case(case)
    if not p['eligible']:
        return dict(choice=None,origin='abstain',normalization=None,completion_boundary=None,
                    fallback_reason=None,eligible_verified=True)
    if raw is None:
        return dict(choice=cpu_choice(case),origin='cpu',normalization=None,completion_boundary=None,
                    fallback_reason=None,eligible_verified=True)
    choice,normalization,boundary,reason=parse_response(raw,p['eligible'],original_stops,bounded)
    return dict(choice=choice if choice is not None else cpu_choice(case),
        origin=('hat_canonicalized' if normalization else 'hat_exact') if choice is not None else 'cpu_fallback',
        normalization=normalization if choice is not None else None,completion_boundary=boundary,
        fallback_reason=reason,eligible_verified=True)


def generate_controlled(llm, render, environment, messages, bounded):
    """Always restore original stop settings; never recover suppressed candidate text."""
    started=time.monotonic();original=list(environment['stop_tokens'])
    requested=list(dict.fromkeys(original+list(CATALOG))) if bounded else original
    raw=None;active=None;restored=False;reset_verified=False;restoration_error=None
    try:
        reset_context(llm)
        llm.set_stop_tokens(requested)
        active=list(llm.get_stop_tokens())
        if active!=requested:raise RuntimeError('Stop-setting readback mismatch')
        # Original terminators alone are used for visible-token timing: candidate IDs
        # are answer text, even when they also terminate bounded generation.
        raw=generate_checked(llm,render,messages,'',True,environment['parameters'],
            original,environment['experiment_limit_tokens'])
    except Exception as exc:
        raw=dict(error=f'{type(exc).__name__}: {exc}',output=None,completion_status='ERROR',
                 token_ledger_valid=False,messages=messages,effective_parameters=environment['parameters'])
    finally:
        restore_started=time.monotonic()
        try:
            llm.set_stop_tokens(original)
            restored=list(llm.get_stop_tokens())==original
            reset_context(llm);reset_verified=True
        except Exception as exc:
            restoration_error=f'{type(exc).__name__}: {exc}'
        restoration_ms=(time.monotonic()-restore_started)*1000
    return dict(raw=raw,requested_stops=requested,active_stops=active,restored=restored,
        reset_verified=reset_verified,restoration_error=restoration_error,
        restoration_ms=restoration_ms,started=started,ended=time.monotonic(),
        request_ms=(time.monotonic()-started)*1000)


class AdviserWorker(HatWorker):
    def perform(self, task):
        if owner()!=self.owner:raise RuntimeError('HAT context owner changed')
        if task['operation']!='generate':raise ValueError('Unknown adviser worker operation')
        return generate_controlled(self.adviser.llm,self.adviser.render,self.environment,
                                   task['messages'],task['bounded'])
