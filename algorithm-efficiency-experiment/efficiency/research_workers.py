"""Persistent device owners for research; no shared live Hailo/Vulkan contexts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

from .attention_native import Native,errors
from .common import digest_file,digest_value
from .coordination_workers import owner
from .feedback_spec import input_hash
from .research_spec import validate_snapshot


class Numeric:
    def __init__(self,directory,config,stack):
        self.directory=Path(directory);self.owner=owner()
        gpu=config.get('numeric_device','both')
        self.cpu=stack.enter_context(Native(self.directory/'native-build')) if gpu!='gpu' else None
        self.gpu=stack.enter_context(Native(self.directory/'native-build',gpu=True)) if gpu!='cpu' else None
        self.environment=dict(device=self.gpu.identity if self.gpu else 'CPU',numeric_device=gpu,
            cpu_initialization_ms=self.cpu.initialization_ms if self.cpu else None,
            gpu_initialization_ms=self.gpu.initialization_ms if self.gpu else None)
        self.metadata={r['fixture_id']:r for r in json.loads((self.directory/'fixtures.json').read_text())}
        self.data={};self.outputs={}

    def load(self,fid):
        self.data.clear();meta=self.metadata[fid];path=self.directory/'fixtures'/(fid+'.npz')
        if digest_file(path)!=meta['file_sha256']:raise ValueError('Fixture changed')
        with np.load(path,allow_pickle=False) as f:self.data[fid]=(f['x'],f['w'],f['expected'])
        x,_,_=self.data[fid];self.output=np.empty_like(x)
        for device in (self.cpu,self.gpu):
            if device:device.configure(x.shape)
        return dict(fixture_id=fid)

    def measure(self,fid,arm,persist=False):
        x,w,expected=self.data[fid];native=self.cpu if arm['backend'].startswith('native') else self.gpu
        if native is None:raise ValueError('Request sent to wrong device owner')
        started=time.monotonic()
        result=native.run(x,w,self.output,self.metadata[fid]['mode'],arm['backend'],
                          variant=arm.get('variant',0),profiling=arm.get('profiling',False))
        computed=time.monotonic();check=errors(self.output,expected);checksum=input_hash(self.output)
        ended=time.monotonic();relative=f'outputs/{checksum}.npy'
        if persist:
            path=self.directory/relative
            if not path.exists():np.save(path,self.output,allow_pickle=False)
            self.outputs[fid,arm['arm']]=(checksum,relative)
        known=self.outputs.get((fid,arm['arm']))
        matches=known is not None and known[0]==checksum
        if not matches and not persist:
            # Preserve unexpected outputs too; mark the IO-contaminated sample explicitly.
            np.save(self.directory/relative,self.output,allow_pickle=False)
        return dict(fixture_id=fid,**arm,started=started,computed=computed,ended=ended,
            worker_request_ms=(ended-started)*1000,**result,**check,output_sha256=checksum,
            output_file=relative,matches_warmup=matches,unexpected_output_io=not matches and not persist)

    def perform(self,task):
        if owner()!=self.owner:raise RuntimeError('Device ownership changed')
        op=task['operation'];fid=task.get('fixture_id')
        if op=='load':return self.load(fid)
        if op in ('warm','measure'):
            return self.measure(fid,task['arm'],persist=op=='warm')
        if op=='batch':
            rows=[]
            for i in task['request_ids']:
                if time.monotonic()>=task['deadline']:raise TimeoutError('Numerical batch deadline')
                rows.append(dict(request_id=i,**self.measure(fid,task['arm'])))
            return dict(requests=rows,correct=all(r['correct'] and r['matches_warmup'] for r in rows))
        raise ValueError('Unknown numerical task')


class Context:
    def __init__(self,directory,config,stack):
        from hailo_platform import VDevice,__version__
        from hailo_platform.pyhailort.pyhailort import LLM
        from .state_isolation import resolve_defaults,parameter_settings
        from .diagnostic import renderer
        self.directory=Path(directory);self.owner=owner();start=time.monotonic()
        device=stack.enter_context(VDevice());self.llm=stack.enter_context(LLM(device,config['model']))
        llm=self.llm;defaults=resolve_defaults(llm,__version__)
        self.parameters=parameter_settings(defaults,'penalty_1_0')[0]
        self.stops=llm.get_stop_tokens();self.template=llm.prompt_template();self.render=renderer(self.template)
        self.limit=min(1792,llm.max_context_capacity()-256);self.model_sha=digest_file(config['model'])
        self.environment=dict(hailort_version=__version__,model_path=config['model'],model_sha256=self.model_sha,
            parameters=self.parameters,model_defaults=defaults,stop_tokens=self.stops,prompt_template=self.template,
            capacity_tokens=llm.max_context_capacity(),experiment_limit_tokens=self.limit,loading_seconds=time.monotonic()-start)

    def reset(self):
        self.llm.clear_context()
        if self.llm.get_context_usage_size()!=0:raise RuntimeError('NPU reset failed')
        if self.llm.get_stop_tokens()!=self.stops:raise RuntimeError('NPU stop settings changed')

    def request(self,messages,accumulated,arm,expected,question):
        from .hat import exact_suffix,assistant_content,visible_text
        from .state_isolation import snapshot_memory_guard
        from .diagnostic import score_answer
        llm=self.llm;started=time.monotonic();cpu_started=time.process_time()
        full=self.render(messages);suffix,old,new,ids=exact_suffix(full,accumulated,llm.tokenize)
        before=llm.get_context_usage_size()
        recovery=len(llm.tokenize(llm.get_generation_recovery_sequence()))
        if len(ids)+self.parameters['max_generated_tokens']+recovery>self.limit:raise ValueError('Context budget overflow')
        if before!=len(old):raise ValueError('Before-request context ledger differs')
        checkpoint=None;clear_ms=0
        if arm=='rebuild':
            tick=time.monotonic();self.reset();clear_ms=(time.monotonic()-tick)*1000
        elif arm=='checkpoint' and before:
            tick=time.monotonic();blob=llm.save_context();saved=time.monotonic()
            metadata=dict(model_sha256=self.model_sha,sha256=hashlib.sha256(blob).hexdigest(),context_tokens=before)
            memory=snapshot_memory_guard(len(blob));validate_snapshot(metadata,blob,self.model_sha,before)
            self.reset();cleared=time.monotonic();llm.load_context(blob);loaded=time.monotonic()
            after_load=llm.get_context_usage_size()
            if after_load!=before:raise ValueError('Restored context ledger differs')
            checkpoint=dict(**metadata,bytes=len(blob),**memory,restored_tokens=after_load,
                save_ms=(saved-tick)*1000,validation_and_clear_ms=(cleared-saved)*1000,load_ms=(loaded-cleared)*1000)
            del blob
        submitted=full if arm=='rebuild' else suffix
        chunks=[];first=None;generation_started=time.monotonic()
        with llm.generate(submitted,**self.parameters) as generation:
            while str(generation.generation_status).endswith('.GENERATING'):
                chunks.append(generation.read(timeout_ms=90000))
                if first is None and visible_text(''.join(chunks),self.stops).strip():first=time.monotonic()
            status=str(generation.generation_status).split('.')[-1]
        generated=time.monotonic();output=''.join(chunks);after=llm.get_context_usage_size()
        after_ids=llm.tokenize(full+output);expected_after=len(after_ids)
        ledger=after==expected_after
        try:body=assistant_content(output,self.stops);terminal=True
        except ValueError:body=None;terminal=False
        valid=ledger and terminal and status=='LOGICAL_END_OF_GENERATION'
        queried=re.search(r'item[0-9]+',question)[0]
        score=score_answer(output,expected,queried,self.stops,status)
        submitted_ids=llm.tokenize(submitted)
        ended=time.monotonic()
        return dict(started=started,ended=ended,request_ms=(ended-started)*1000,cpu_seconds=time.process_time()-cpu_started,
            generation_ms=(generated-generation_started)*1000,first_visible_ms=(first-started)*1000 if first else None,
            clear_ms=clear_ms,checkpoint=checkpoint,effective_prompt=full,submitted_prompt=submitted,
            prompt_sha256=digest_value(full),prompt_tokens=len(ids),submitted_tokens=len(submitted_ids),
            context_before=before,context_after=after,expected_context_after=expected_after,ledger_valid=ledger,
            history_token_ids=old,prompt_token_ids=ids,submitted_token_ids=submitted_ids,after_token_ids=after_ids,
            output=output,status=status,valid=valid,assistant_body=body,answer_correct=score['strict_correct'],
            expected_answer=expected,score=score,stream_chunks=chunks,parameters=self.parameters)

    def dialogue(self,fixture,arm,deadline):
        start=time.monotonic();self.reset();messages=deepcopy(fixture['initial_messages']);accumulated='';rows=[]
        failures=[];quarantined=[]
        try:
            for turn in range(4):
                if time.monotonic()>=deadline:raise TimeoutError('Context dialogue deadline')
                if turn:messages.append(dict(role='user',content=fixture['questions'][turn]))
                try:row=self.request(messages,accumulated,arm,fixture['answers'][turn],fixture['questions'][turn])
                except ValueError as exc:
                    failures.append(dict(turn=turn,category='context_contract',error=str(exc)))
                    quarantined=list(range(turn,4));break
                rows.append(dict(turn=turn,**row))
                if not row['valid']:quarantined=list(range(turn+1,4));break
                accumulated=row['effective_prompt']+row['output']
                messages.append(dict(role='assistant',content=row['assistant_body']))
        finally:self.reset()
        return dict(rows=rows,session_ms=(time.monotonic()-start)*1000,reset_verified=True,
                    contract_failures=failures,quarantined_turns=quarantined)

    def perform(self,task):
        if owner()!=self.owner:raise RuntimeError('NPU ownership changed')
        op=task['operation']
        if op=='fixture':
            from .hat import build_fixture
            return build_fixture(self.render,self.llm.tokenize,task['target'],task['seed'],4)
        if op=='dialogue':return self.dialogue(task['fixture'],task['arm'],task['deadline'])
        if op=='prepare':
            self.followup=None
            fixture=task['fixture'];self.reset();messages=deepcopy(fixture['initial_messages'])
            row=self.request(messages,'','rebuild',fixture['answers'][0],fixture['questions'][0])
            if not row['valid']:
                self.reset()
                return dict(valid=False,initial=row,reset_verified=True)
            messages.extend([dict(role='assistant',content=row['assistant_body']),dict(role='user',content=fixture['questions'][1])])
            self.followup=(messages,row['effective_prompt']+row['output'],fixture['answers'][1],fixture['questions'][1])
            return dict(valid=True,initial=row)
        if op=='followup':
            if self.followup is None:
                started=time.monotonic();self.reset();ended=time.monotonic()
                return dict(skipped=True,valid=False,answer_correct=False,skip_reason='invalid_prelude',
                    generated=False,reset_verified=True,started=started,ended=ended,request_ms=(ended-started)*1000)
            messages,accumulated,expected,question=self.followup
            try:return self.request(messages,accumulated,task['arm'],expected,question)
            finally:self.followup=None;self.reset()
        if op=='reset':self.reset();return dict(reset_verified=True)
        raise ValueError('Unknown NPU context task')
