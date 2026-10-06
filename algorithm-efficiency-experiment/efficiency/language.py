"""Bounded interpretation; all factual execution uses the immutable CPU index."""
import re
import time

from .diagnostic import LOGICAL_END, reset_context
from .hat import COLORS, assistant_content
from .hybrid import generate_checked

ENGINES = ('cpu','hat','auto')
QUERY_ENGINES = (*ENGINES, 'strict')
SYSTEM = '''Translate one question into one command. Output only LOOKUP itemNNN, COUNT color, LIST color, or ABSTAIN.
LOOKUP retrieves an item's color. COUNT counts items of one color. LIST returns their item IDs.
Colors: blue, green, red, yellow, white, black, pink, brown. Copy the explicit identifier or color.
ABSTAIN for ambiguous, unsupported, negated, hypothetical, multi-operation or write requests. Do not answer the question.
Examples:
Please retrieve item777's recorded color. => LOOKUP item777
Compute the cardinality of items colored red. => COUNT red
Name the identifiers belonging to green. => LIST green
What is item888's mass? => ABSTAIN
Count blue or red items. => ABSTAIN
Remove item999. => ABSTAIN'''
WARMUPS = ["Please retrieve item777's recorded color.",
           'Compute the cardinality of items colored red.',
           'Name the identifiers belonging to green.',
           "What is item888's mass?", 'Count blue or red items.', 'Remove item999.']


def validate_question(question):
    if not isinstance(question,str) or not question.strip():
        raise ValueError('Question must be a nonempty string')


def normalize(question):
    validate_question(question)
    return ' '.join(question.casefold().split()).rstrip('.?!').rstrip()


def parse_cpu(question):
    text = normalize(question)
    colors = '(?:'+'|'.join(COLORS)+')'
    patterns = (
        ('LOOKUP',r'what color is (item[0-9]+)'),('LOOKUP',r'look up (item[0-9]+)'),
        ('COUNT',fr'how many items are ({colors})'),('COUNT',fr'count ({colors}) items'),
        ('LIST',fr'which items are ({colors})'),('LIST',fr'list ({colors}) items'))
    for operation,pattern in patterns:
        match = re.fullmatch(pattern,text)
        if match:
            return dict(operation=operation,argument=match[1])
    return None


def needs_hat(question,engine):
    validate_question(question)
    if engine not in QUERY_ENGINES:
        raise ValueError('Unknown engine')
    return engine=='hat' or (engine=='auto' and parse_cpu(question) is None)


def messages(question):
    return [dict(role='system',content=SYSTEM),dict(role='user',content=question)]


def parse_command(output,stops):
    try:
        body = assistant_content(output,stops).strip()
    except ValueError:
        return None,'missing_terminal_token'
    if '<|' in body or '|>' in body or any(s in body for s in stops):
        return None,'exposed_control_marker'
    if body=='ABSTAIN':
        return dict(operation='ABSTAIN',argument=None),None
    match = re.fullmatch(r'(LOOKUP) (item[0-9]+)',body)
    if not match:
        match = re.fullmatch(r'(COUNT|LIST) ('+'|'.join(COLORS)+')',body)
    if not match:
        return None,'malformed_command'
    return dict(operation=match[1],argument=match[2]),None


def argument_error(question,command):
    """Argument grounding is structural; it cannot prove the intended operation."""
    text = question.casefold()
    ids = set(re.findall(r'\bitem[0-9]+\b',text))
    colors = set(re.findall(r'\b(?:'+'|'.join(COLORS)+r')\b',text))
    if command['operation']=='LOOKUP':
        if len(ids)!=1 or colors:
            return 'ambiguous_or_missing_argument'
        return None if ids=={command['argument']} else 'argument_mismatch'
    if len(colors)!=1 or ids:
        return 'ambiguous_or_missing_argument'
    return None if colors=={command['argument']} else 'argument_mismatch'


def execute(index,command):
    operation,argument = command['operation'],command['argument']
    if operation=='LOOKUP':
        found = index.lookup(argument)
        return dict(status='ok' if found else 'not_found',answer=found[0] if found else None)
    if operation not in ('COUNT','LIST') or argument not in COLORS:
        raise ValueError('Unsupported CPU operation')
    matches = sorted(item for item,color in index.records if color==argument)
    return dict(status='ok',answer=len(matches) if operation=='COUNT' else matches)


def ask(index,question,engine='auto',llm=None,render=None,parameters=None,stops=None,limit=None):
    """No evaluation labels enter this function; malformed interpretations abstain."""
    start = time.perf_counter()
    validate_question(question)
    if engine not in QUERY_ENGINES:
        raise ValueError('Unknown engine')
    result = dict(engine=engine,question=question,operation=None,argument=None,answer=None,
        status='abstain',abstention_reason=None,interpretation_source='cpu_parser',hat_called=False,
        candidate_command=None,table_sha256=index.table_sha256)
    if engine=='strict':
        from .strict_language import VERSION, EXAMPLES, parse_strict
        candidate,rule,reason = parse_strict(question)
        result.update(interpretation_source='strict_parser',grammar_version=VERSION,
            matched_rule=rule,candidate_command=candidate,abstention_reason=reason)
        if reason: result['supported_examples'] = list(EXAMPLES)
    else:
        candidate = parse_cpu(question) if engine!='hat' else None
    if engine=='strict':
        pass
    elif candidate is None and engine=='cpu':
        result['abstention_reason'] = 'unrecognized_cpu_grammar'
    elif candidate is None:
        result['interpretation_source'] = 'hat'
        if llm is None:
            raise RuntimeError('HAT engine required for this question')
        prompt_messages = messages(question)
        tokens = llm.tokenize(render(prompt_messages))
        if len(tokens)+16+len(llm.tokenize(llm.get_generation_recovery_sequence()))>limit:
            result['abstention_reason'] = 'context_budget'
        else:
            raw = generate_checked(llm,render,prompt_messages,'',True,parameters,stops,limit)
            result.update(raw,hat_called=True)
            candidate,reason = parse_command(raw['output'],stops)
            result['candidate_command'] = candidate
            if raw['completion_status']!=LOGICAL_END or not raw['token_ledger_valid']:
                reason = raw['completion_status'] if raw['completion_status']!=LOGICAL_END else 'invalid_context_ledger'
                reset_context(llm)
                candidate = None
            if reason:
                result['abstention_reason'] = reason
                candidate = None
            elif candidate['operation']=='ABSTAIN':
                result['abstention_reason'] = 'model_abstained'
                candidate = None
    else:
        result['candidate_command'] = candidate
    interpreted = time.perf_counter()
    if candidate is not None:
        reason = argument_error(question,candidate)
        if reason:
            result['abstention_reason'] = reason
        else:
            result.update(execute(index,candidate),**candidate)
    result.update(interpretation_ms=(interpreted-start)*1000,
        execution_validation_ms=(time.perf_counter()-interpreted)*1000)
    result['request_ms'] = (time.perf_counter()-start)*1000
    return result
