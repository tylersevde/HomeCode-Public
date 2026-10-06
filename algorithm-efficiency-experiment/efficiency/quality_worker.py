"""A quality-only adapter; legacy context workers and defaults remain unchanged."""
from copy import deepcopy

from .coordination_workers import owner
from .quality_spec import CONDITIONS, build_fixture, messages
from .research_workers import Context


class QualityContext(Context):
    def __init__(self, directory, config, stack):
        super().__init__(directory, config, stack)
        self.environment['recovery_token_ids'] = list(self.llm.tokenize(self.llm.get_generation_recovery_sequence()))

    def perform(self, task):
        if owner() != self.owner:
            raise RuntimeError('NPU context ownership changed')
        if task['operation'] == 'quality_fixture':
            return build_fixture(self.render, self.llm.tokenize, task['spec'])
        if task['operation'] != 'quality_dialogue':
            raise ValueError('Unknown quality task')
        arm = task['condition']
        if {k: arm[k] for k in ('id','prompt_id','prompt','max_generated_tokens')} not in CONDITIONS:
            raise ValueError('Unknown quality condition')
        original = self.parameters
        self.parameters = dict(original, max_generated_tokens=arm['max_generated_tokens'])
        fixture = deepcopy(task['fixture'])
        fixture['initial_messages'] = messages(fixture['table'], arm['prompt_id'])
        try:
            result = self.dialogue(fixture, 'rebuild', task['deadline'])
            for row in result['rows']:
                row.update(condition=deepcopy(arm), parameters=dict(row['parameters']),
                           recovery_tokens=len(self.environment['recovery_token_ids']))
            return result
        finally:
            self.parameters = original
