"""Device owners for isolated CPU repeats and the full-table/one-fact comparison."""
from copy import deepcopy
import re
import time
from .coordination_workers import owner
from .research_workers import Numeric
from .refine_worker import CalibratedNumeric, TerminationContext, sample
from .reliability_spec import sizing
from .hat import SYSTEM

def select_fact(table,question):
    match=re.fullmatch(r'What color is (item[0-9]+)\?',question)
    if not match:raise ValueError('Unsupported fact question')
    matches=[list(row) for row in table if len(row)==2 and row[0]==match[1]]
    if len(matches)!=1:raise ValueError('Reference fact must occur exactly once')
    return matches[0]

def fact_messages(fact,question):
    return [dict(role='system',content=SYSTEM),dict(role='user',content=f'Reference facts: {fact[0]} = {fact[1]}. {question}')]

class ReliableNumeric(CalibratedNumeric):
    def measure(self,*args,**kwargs):
        t0=time.monotonic();before=sample();t1=time.monotonic()
        result=Numeric.measure(self,*args,**kwargs)
        t2=time.monotonic();after=sample();t3=time.monotonic()
        if any(s['governor']!=self.policy['governor'] for s in (before,after)):raise RuntimeError('Governor drift')
        result.update(policy=self.policy,system_before=before,system_after=after,observer_ms=((t1-t0)+(t3-t2))*1000)
        return result

    def perform(self,task):
        if owner()!=self.owner:raise RuntimeError('Device ownership changed')
        if task['operation']!='batch':return super().perform(task)
        ids=task['request_ids']
        if not ids or len(set(ids))!=len(ids):raise ValueError('Invalid batch request IDs')
        observation_start=time.monotonic();before=sample();started=time.monotonic();rows=[]
        for request_id in ids:
            if time.monotonic()>=task['deadline']:raise TimeoutError('Numerical batch deadline')
            # One transport and governor observation per batch; preserve every native call and verification.
            rows.append(dict(request_id=request_id,**Numeric.measure(self,task['fixture_id'],task['arm'])))
        observation_end=time.monotonic();after=sample();ended=time.monotonic()
        if any(s['governor']!=self.policy['governor'] for s in (before,after)):raise RuntimeError('Governor drift')
        return dict(requests=rows,policy=self.policy,system_before=before,system_after=after,
            worker_batch_ms=(ended-started)*1000,observer_ms=((started-observation_start)+(ended-observation_end))*1000,
            correct=all(r['correct'] and r['matches_warmup'] and not r['validation_errors'] for r in rows))

class ReliableContext(TerminationContext):
    def perform(self,task):
        if owner()!=self.owner:raise RuntimeError('Device ownership changed')
        if task['operation']=='size':return sizing(self.render,self.llm.tokenize,task['spec'])
        if task['operation']=='single_fact':
            # All four questions are independent; invalid answers do not contaminate a later context.
            self.set_condition('punctuation');fixture=task['fixture'];rows=[];contracts=[]
            started=time.monotonic()
            try:
                for turn,question in enumerate(fixture['questions']):
                    if time.monotonic()>=task['deadline']:raise TimeoutError('One-fact deadline')
                    tick=time.monotonic();fact=select_fact(fixture['table'],question);selected=time.monotonic()
                    messages=fact_messages(fact,question)
                    row=self.request(messages,'','rebuild',fixture['answers'][turn],question)
                    rows.append(dict(turn=turn,selected_fact=fact,selector_ms=(selected-tick)*1000,**row))
            finally:self.reset()
            return dict(rows=rows,session_ms=(time.monotonic()-started)*1000,reset_verified=True,
                contract_failures=contracts,quarantined_turns=[],condition='single_fact',diagnostic_only=True,
                native_stops=list(self.native_stops),effective_stops=list(self.stops))
        return super().perform(task)
