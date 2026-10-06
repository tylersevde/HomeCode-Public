"""Matched NPU factor arms preserving raw model history and all denominators."""
from copy import deepcopy
import time

from .coordination_workers import owner
from .reliability_worker import ReliableContext, select_fact
from .completion_npu_spec import planned_messages


class CompletionContext(ReliableContext):
    def perform(self, task):
        if owner() != self.owner:
            raise RuntimeError('NPU ownership changed')
        if task['operation'] != 'completion_dialogue':
            return super().perform(task)
        arm, fixture = task['arm'], task['fixture']
        if arm['table_mode'] not in ('full_table', 'selected_fact') or arm['history_mode'] not in ('actual_history', 'independent'):
            raise ValueError('Unknown NPU factor arm')
        if not arm['diagnostic_only'] and (arm['table_mode'], arm['history_mode']) != ('full_table', 'actual_history'):
            raise ValueError('Simplified task cannot qualify')
        self.set_condition(arm['condition'])
        rows, failures, quarantined, history = [], [], [], []
        started = time.monotonic()
        try:
            for turn in range(4):
                if time.monotonic() >= task['deadline']:
                    raise TimeoutError('Completion NPU dialogue deadline')
                tick = time.monotonic()
                messages = planned_messages(fixture, arm, turn, history)
                fact = select_fact(fixture['table'], fixture['questions'][turn]) if arm['table_mode'] == 'selected_fact' else None
                selector_ms = (time.monotonic() - tick) * 1000
                try:
                    row = self.request(messages, '', 'rebuild', fixture['answers'][turn], fixture['questions'][turn])
                except ValueError as exc:
                    failures.append(dict(turn=turn, category='context_contract', error=str(exc)))
                    if arm['history_mode'] == 'actual_history':
                        quarantined = list(range(turn, 4))
                        break
                    quarantined.append(turn)
                    self.reset()
                    continue
                rows.append(dict(turn=turn, selected_fact=fact, selector_ms=selector_ms, **row))
                if arm['history_mode'] == 'actual_history':
                    if not row['valid']:
                        quarantined = list(range(turn + 1, 4))
                        break
                    # Even a factually wrong answer is retained exactly as generated.
                    history = deepcopy(messages) + [dict(role='assistant', content=row['assistant_body'])]
        finally:
            self.reset()
        return dict(rows=rows, contract_failures=failures, quarantined_turns=quarantined,
                    reset_verified=True, session_ms=(time.monotonic() - started) * 1000,
                    arm=deepcopy(arm), diagnostic_only=arm['diagnostic_only'],
                    native_stops=list(self.native_stops), effective_stops=list(self.stops))
