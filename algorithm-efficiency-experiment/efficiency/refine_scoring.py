"""Native/control termination and whole-answer semantics kept separate."""
import re
from .hat import COLORS

def score(output,expected,queried_item,native_stops,effective_stops,status):
    result=dict(factual_correct=None,strict_correct=False,format_correct=False,
                parsed_color=None,answer_category='malformed',score_reason=None)
    suffix=next((s for s in sorted(effective_stops,key=len,reverse=True) if s and output.endswith(s)),None)
    body=output[:-len(suffix)] if suffix else None
    result.update(terminal_suffix=suffix,assistant_body=body)
    if status!='LOGICAL_END_OF_GENERATION':
        result.update(answer_category='truncated' if status=='MAX_TOKENS_REACHED' else 'incomplete',score_reason=status)
        return result
    if suffix is None:
        result['score_reason']='missing_terminal_token';return result
    if not body.strip():
        result['score_reason']='empty_output';return result
    if '<|' in body or '|>' in body or any(s in body for s in native_stops):
        result['score_reason']='exposed_control_marker';return result
    text = ' '.join(body.casefold().split()).strip().strip('.,!?;:').strip()
    colors = '|'.join(COLORS)
    one_word = re.fullmatch(f'({colors})', text)
    patterns = [f'the color of (item[0-9]+) is ({colors})',
                f'(item[0-9]+) is ({colors})', f'(item[0-9]+) = ({colors})']
    color = one_word[1] if one_word else None
    if color is None:
        for pattern in patterns:
            matched = re.fullmatch(pattern, text)
            if matched:
                if matched[1] != queried_item.casefold():
                    result['score_reason'] = 'wrong_identifier'
                    return result
                color = matched[2]
                break
    if color is None:
        matched = re.fullmatch(f'it is ({colors})', text)
        color = matched[1] if matched else None
    if color is None:
        result['score_reason'] = 'unsupported_or_ambiguous_answer'
        return result
    correct = color == expected
    result.update(parsed_color=color, factual_correct=correct, format_correct=bool(one_word),
                  strict_correct=bool(one_word) and correct,
                  answer_category='correct_fact' if correct else 'wrong_fact')
    return result
