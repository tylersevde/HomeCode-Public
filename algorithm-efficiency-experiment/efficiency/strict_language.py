"""Versioned, closed English grammar; contains no evaluation labels or model calls."""
import re

from .hat import COLORS

VERSION = 'strict-english-v1'
EXAMPLES = ('What color is item001?', 'How many items are blue?', 'List blue items.')
# These are the thirty previously observed supported forms, now a public contract.
# Evaluation imports neither these rules nor their compiled patterns for labels.
FORMS = {
    'LOOKUP': (
        'what color is {item}', 'look up {item}',
        'tell me the color assigned to {item}', "i would like to know {item}'s color",
        'give me the color recorded for {item}', 'what is the color value for {item}',
        "report {item}'s color", 'find the color belonging to {item}',
        'return the stored color of {item}', 'for {item}, which color is on record'),
    'COUNT': (
        'how many items are {color}', 'count {color} items',
        'give me the number of items colored {color}', 'what is the total number of {color} items',
        'i need a count of the items whose color is {color}', 'tell me how many entries have color {color}',
        'find the quantity of {color} items', 'what is the size of the set of {color} items',
        'return the number of records with color {color}', 'how large is the group of items that are {color}'),
    'LIST': (
        'which items are {color}', 'list {color} items',
        'give me the identifiers of items colored {color}', 'show the item ids whose color is {color}',
        'i need the names of all {color} items', 'find every item that has color {color}',
        'return the identifiers for the {color} group', 'tell me which records have color {color}',
        'enumerate the items marked {color}', 'what are the item ids associated with {color}')}


def specification():
    return dict(version=VERSION, forms=FORMS, colors=COLORS,
        plural_nouns=['items', 'entries', 'records'], optional_prefix='please ',
        normalization='casefold; collapse whitespace; strip trailing .?! and whitespace',
        identifier='item[0-9]+; preserve digits', match='entire normalized question; unique command',
        examples=EXAMPLES)


def compile_rules():
    rules = []
    for operation, forms in FORMS.items():
        for number, form in enumerate(forms, 1):
            pieces = re.split(r'(\{item\}|\{color\}|\b(?:items|entries|records)\b)', form)
            parts = []
            for piece in pieces:
                if piece == '{item}': parts.append(r'(?P<argument>item[0-9]+)')
                elif piece == '{color}': parts.append('(?P<argument>'+'|'.join(COLORS)+')')
                elif piece in ('items', 'entries', 'records'): parts.append('(?:items|entries|records)')
                else: parts.append(re.escape(piece))
            rules.append((f'{operation.lower()}-{number:02d}', operation,
                          re.compile(r'(?:please )?'+''.join(parts))))
    return tuple(rules)


RULES = compile_rules()


def parse_strict(question):
    if not isinstance(question, str) or not question.strip():
        raise ValueError('Question must be a nonempty string')
    text = ' '.join(question.casefold().split()).rstrip('.?!').rstrip()
    matches = [(rule, operation, match['argument']) for rule, operation, pattern in RULES
               if (match := pattern.fullmatch(text))]
    commands = {(operation, argument) for _, operation, argument in matches}
    if len(commands) != 1:
        return None, None, 'conflicting_strict_interpretations' if commands else 'unrecognized_strict_grammar'
    operation, argument = next(iter(commands))
    return dict(operation=operation, argument=argument), matches[0][0], None
