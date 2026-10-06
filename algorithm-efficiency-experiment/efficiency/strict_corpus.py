"""Pre-authored evaluation questions, independent of the strict runtime grammar."""
from .hat import COLORS
from .language import WARMUPS
from .language_protocol import expected_result

CONTRACT = {
    'LOOKUP': (
        'Please tell me the color assigned to {item}.',
        'Please give me the color recorded for {item}.',
        'Please return the stored color of {item}.',
        'Please find the color belonging to {item}.',
        'Please look up {item}.'),
    'COUNT': (
        'Please count {color} entries.', 'How many records are {color}?',
        'Give me the number of records colored {color}.',
        'What is the size of the set of {color} entries?',
        'How large is the group of records that are {color}?'),
    'LIST': (
        'Please list {color} records.', 'Which entries are {color}?',
        'Give me the identifiers of entries colored {color}.',
        'Enumerate the records marked {color}.', 'Tell me which entries have color {color}?')}
UNFAMILIAR = {
    'LOOKUP': (
        '{item} has a color on file; what is it?',
        'Under the color field for {item}, what value would I find?',
        'Can you identify the recorded hue of {item}?',
        'Looking only at {item}, report its color value.',
        'For the entry named {item}, supply the saved color.'),
    'COUNT': (
        'If I keep only the {color} items, how many remain?',
        'After filtering for {color}, how many records would be left?',
        'Report a single integer: the number of entries colored {color}.',
        'The {color} subset contains how many items?',
        'What would a tally of every {color} entry come to?'),
    'LIST': (
        'If I keep only the {color} items, which identifiers remain?',
        'After filtering for {color}, which record IDs would be left?',
        'Report the IDs, one for each entry colored {color}.',
        'The {color} subset contains which item identifiers?',
        'What identifiers belong to every entry whose color is {color}?')}
NEGATIVES = (
    ('missing_referent', 'In collection {n}, return the color of the one I mean.'),
    ('multiple_items', 'Look up {item}; also look up {other_item}.'),
    ('multiple_colors', 'List {color} items together with {other_color} items.'),
    ('negation', 'Count entries whose color is anything except {color}.'),
    ('unsupported_color', 'Count {unknown_color} records.'),
    ('unsupported_attribute', 'For {item}, provide the stored weight.'),
    ('unsupported_aggregation', 'What fraction of records are {color}?'),
    ('compound_request', 'How many items are {color}, and which IDs do they have?'),
    ('hypothetical_override', 'Pretend {item} has color {color}; tell me its color using that assumption.'),
    ('mutation', "Change {item}'s color to {color}."))


def build_corpus(fixtures, previous_questions=()):
    fixtures = sorted(fixtures, key=lambda f: (f['target_tokens'], f['fixture_id']))
    if len(fixtures) != 30: raise ValueError('Thirty verified tables are required')
    absent = [(f, c) for f in fixtures for c in COLORS if c not in dict(f['table']).values()]
    if not absent: raise ValueError('An empty color filter is required')
    corpus = []
    for stratum, forms in (('contract', CONTRACT), ('unfamiliar', UNFAMILIAR)):
        for operation, templates in forms.items():
            for number, template in enumerate(templates, 1):
                family = f'{stratum}-{operation.lower()}-{number:02d}'
                empty_fixture, empty_color = absent[(len(corpus)//4) % len(absent)]
                alternatives = [c for c in COLORS if c != empty_color]
                for variant in range(4):
                    f = fixtures[len(corpus) % 30]
                    if operation == 'LOOKUP':
                        positions = (0, len(f['table'])//2, len(f['table'])-1)
                        argument = f['table'][positions[variant]][0] if variant < 3 else 'item99999'
                    elif variant == 3:
                        f, argument = empty_fixture, empty_color
                    else:
                        argument = alternatives[((number-1)*3+variant) % len(alternatives)]
                    corpus.append(dict(case_id=f'{family}-{variant+1}', family_id=family,
                        stratum=stratum, category=operation, supported=True, fixture_id=f['fixture_id'],
                        question=template.format(item=argument, color=argument),
                        expected_command=dict(operation=operation, argument=argument),
                        expected_result=expected_result(f['table'], operation, argument)))
    for number, (category, template) in enumerate(NEGATIVES, 1):
        for variant in range(4):
            f = fixtures[len(corpus) % 30]
            family = f'negative-{number:02d}'
            corpus.append(dict(case_id=f'{family}-{variant+1}', family_id=family,
                stratum='negative', category=category, supported=False, fixture_id=f['fixture_id'],
                question=template.format(n=variant+1, item=f'item{variant+2:03d}', other_item=f'item{variant+6:03d}',
                    color=COLORS[(number-1+variant) % 8], other_color=COLORS[(number+variant) % 8],
                    unknown_color=('purple', 'orange', 'gray', 'violet')[variant]),
                expected_command=None, expected_result=dict(status='abstain', answer=None)))
    questions = {c['question'] for c in corpus}
    if len(questions) != 160 or questions & (set(previous_questions) | set(WARMUPS)):
        raise ValueError('Duplicate or previously used question')
    if len({c['fixture_id'] for c in corpus}) != 30: raise ValueError('All tables must be represented')
    return corpus
