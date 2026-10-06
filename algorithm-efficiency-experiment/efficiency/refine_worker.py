"""Measured OpenMP policies and an owned Llama stop-policy session."""
from copy import deepcopy
import ctypes as ct
import os
from pathlib import Path
import re
from .research_workers import Numeric
from .study_worker import ModelContext
from .refine_spec import POLICIES
from .refine_scoring import score
from .refine_governor import GOVERNOR

def environment(policy):
    return dict(OMP_DYNAMIC='FALSE',OMP_PROC_BIND='CLOSE' if policy['binding']=='bound' else 'FALSE',
        OMP_PLACES='{0},{1},{2},{3}' if policy['binding']=='bound' else None,GOMP_CPU_AFFINITY=None,
        OMP_WAIT_POLICY='PASSIVE' if policy['waiting']=='passive' else None,
        GOMP_SPINCOUNT='0' if policy['waiting']=='passive' else None)

def sample():
    return dict(governor=GOVERNOR.read_text().strip(),
        frequency_khz=int(GOVERNOR.with_name('scaling_cur_freq').read_text()),
        loadavg=list(os.getloadavg()),proc_stat=Path('/proc/stat').read_text().splitlines()[0])

class CalibratedNumeric(Numeric):
    def __init__(self,directory,config,stack):
        self.policy=POLICIES[config['policy']]
        desired=environment(self.policy)
        if any(os.environ.get(k)!=v for k,v in desired.items()):raise RuntimeError('OpenMP launch environment differs')
        super().__init__(directory,config,stack)
        lib=ct.CDLL(str(self.directory/'native-build/libaffinity.so'))
        lib.refine_probe.argtypes=[ct.c_int,ct.POINTER(ct.c_int)];lib.refine_probe.restype=ct.c_int
        probes=[]
        for size in (1,4):
            values=(ct.c_int*16)();team=lib.refine_probe(size,values)
            rows=[dict(tid=values[i*4],cpu=values[i*4+1],mask=values[i*4+2],place=values[i*4+3]) for i in range(team)]
            if team!=size:raise RuntimeError('Actual OpenMP team differs')
            if self.policy['binding']=='bound':
                if [r['mask'] for r in rows]!=[1<<i for i in range(size)]:raise RuntimeError('OpenMP binding differs')
            elif any(r['mask']!=15 for r in rows):raise RuntimeError('Unbound OpenMP affinity differs')
            probes.append(dict(requested=size,team=team,threads=rows))
        self.environment.update(policy=self.policy,openmp_environment=desired,probes=probes,observed=sample())
        if self.environment['observed']['governor']!=self.policy['governor']:raise RuntimeError('Requested governor is not active')
    def measure(self,*args,**kwargs):
        before=sample();result=super().measure(*args,**kwargs);after=sample()
        if before['governor']!=self.policy['governor'] or after['governor']!=self.policy['governor']:raise RuntimeError('Governor changed during measurement')
        result.update(policy=self.policy,system_before=before,system_after=after)
        return result

class TerminationContext(ModelContext):
    def __init__(self,directory,config,stack):
        super().__init__(directory,config,stack)
        self.native_stops=list(self.stops);self.condition='native'
        stack.callback(self.restore)
        self.environment.update(native_stops=self.native_stops,conditions=['native','punctuation'])
    def restore(self):
        self.llm.clear_context();self.llm.set_stop_tokens(self.native_stops)
        self.stops=list(self.native_stops);self.reset()
    def set_condition(self,condition):
        if condition not in ('native','punctuation'):raise ValueError('Unknown termination condition')
        self.llm.clear_context();self.stops=self.native_stops+(['.','\n'] if condition=='punctuation' else [])
        self.llm.set_stop_tokens(self.stops);self.reset();self.condition=condition
    def request(self,messages,accumulated,arm,expected,question):
        row=super().request(messages,accumulated,arm,expected,question)
        scoring=score(row['output'],expected,re.search(r'item[0-9]+',question)[0],self.native_stops,self.stops,row['status'])
        terminal=scoring['terminal_suffix'];body=scoring['assistant_body']
        row.update(score=scoring,assistant_body=body,answer_correct=scoring['strict_correct'],
            native_stops=list(self.native_stops),effective_stops=list(self.stops),condition=self.condition,
            terminal_suffix=terminal,terminal_token_ids=list(self.llm.tokenize(terminal)) if terminal else [],
            output_token_ids=list(self.llm.tokenize(row['output'])),
            valid=bool(row['ledger_valid'] and terminal and body.strip() and row['status']=='LOGICAL_END_OF_GENERATION'))
        # A custom stop must be visible and fully accounted for; never infer hidden suffixes.
        if self.condition=='punctuation' and (not row['ledger_valid'] or (row['status']=='LOGICAL_END_OF_GENERATION' and terminal is None)):
            from .common import emit
            emit(self.directory/'refine.jsonl','compatibility_failure',row=row)
            raise RuntimeError('Stop-policy compatibility failure: unaccountable termination/token ledger')
        return row
    def perform(self,task):
        if task['operation']=='restore':
            self.restore();return dict(stops=self.llm.get_stop_tokens(),context_tokens=self.llm.get_context_usage_size())
        if task['operation']=='dialogue':
            self.set_condition(task['condition'])
            result=self.dialogue(task['fixture'],'rebuild',task['deadline'])
            result.update(native_stops=list(self.native_stops),effective_stops=list(self.stops),condition=self.condition)
            return result
        return super().perform(task)
