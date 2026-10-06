"""Owned, model-aware Hailo sessions; complete prompt rebuilding on every turn."""
from copy import deepcopy
import re
import time
from jinja2 import Environment, StrictUndefined
from .common import digest_value
from .coordination_workers import owner
from .research_workers import Context
from .study_spec import BINDINGS, MODEL_HASHES, PARAMETERS, sizing


def renderer(template,model):
    # Qwen's native template deliberately treats missing optional tool_calls as false.
    environment=Environment(undefined=StrictUndefined) if model=='llama' else Environment()
    compiled=environment.from_string(template)
    bindings=BINDINGS if model=='llama' else dict(tools=None)
    return lambda messages:compiled.render(messages=messages,add_generation_prompt=True,**bindings)


class ModelContext(Context):
    def __init__(self,directory,config,stack):
        super().__init__(directory,config,stack)
        self.model=config['model_id']
        if self.model_sha!=MODEL_HASHES[self.model]:raise ValueError('Model identity differs from frozen package')
        self.parameters=deepcopy(PARAMETERS);self.render=renderer(self.template,self.model)
        self.recovery=list(self.llm.tokenize(self.llm.get_generation_recovery_sequence()))
        probe=self.render([dict(role='system',content='Test'),dict(role='user',content='Test')])
        if self.model=='llama':
            if not probe.startswith(BINDINGS['bos_token']) or probe.count(BINDINGS['bos_token'])!=1:
                raise ValueError('Llama BOS template contract failed')
            if set(self.stops)!={'<|end_of_text|>','<|eom_id|>','<|eot_id|>'}:raise ValueError('Llama native stop contract failed')
        self.reset()
        self.environment.update(parameters=deepcopy(self.parameters),model_id=self.model,
            template_bindings=deepcopy(BINDINGS if self.model=='llama' else dict(tools=None)),
            recovery_token_ids=self.recovery,template_probe=probe,template_probe_ids=list(self.llm.tokenize(probe)),
            reset_verified=True,context_policy='full_rebuild_each_turn')

    def request(self,messages,accumulated,arm,expected,question):
        from .hat import assistant_content,visible_text
        from .diagnostic import score_answer
        llm=self.llm;started=time.monotonic();cpu=time.process_time();prior=llm.get_context_usage_size()
        self.reset();cleared=time.monotonic()
        full=self.render(messages);ids=list(llm.tokenize(full))
        if len(ids)+32+len(self.recovery)>self.limit:raise ValueError('Context budget overflow')
        before=llm.get_context_usage_size()
        if before!=0:raise ValueError('Context not empty before full rebuild')
        chunks=[];first=None;generating=time.monotonic()
        with llm.generate(full,**self.parameters) as generation:
            while str(generation.generation_status).endswith('.GENERATING'):
                chunks.append(generation.read(timeout_ms=90000))
                if first is None and visible_text(''.join(chunks),self.stops).strip():first=time.monotonic()
            status=str(generation.generation_status).split('.')[-1]
        generated=time.monotonic();output=''.join(chunks);after=llm.get_context_usage_size()
        after_ids=list(llm.tokenize(full+output));ledger=after==len(after_ids)
        try:body=assistant_content(output,self.stops);terminal=True
        except ValueError:body=None;terminal=False
        score=score_answer(output,expected,re.search(r'item[0-9]+',question)[0],self.stops,status)
        ended=time.monotonic()
        return dict(started=started,ended=ended,request_ms=(ended-started)*1000,cpu_seconds=time.process_time()-cpu,
            clear_ms=(cleared-started)*1000,generation_ms=(generated-generating)*1000,
            first_visible_ms=(first-started)*1000 if first else None,prior_context_tokens=prior,context_before=before,
            effective_prompt=full,submitted_prompt=full,prompt_sha256=digest_value(full),messages=deepcopy(messages),
            prompt_token_ids=ids,submitted_token_ids=ids,prompt_tokens=len(ids),recovery_token_ids=self.recovery,
            after_token_ids=after_ids,context_after=after,expected_context_after=len(after_ids),ledger_valid=ledger,
            output=output,status=status,assistant_body=body,valid=ledger and terminal and status=='LOGICAL_END_OF_GENERATION',
            answer_correct=score['strict_correct'],score=score,expected_answer=expected,parameters=deepcopy(self.parameters),stream_chunks=chunks)

    def perform(self,task):
        if owner()!=self.owner:raise RuntimeError('NPU ownership changed')
        if task['operation']=='size':return sizing(self.render,self.llm.tokenize,task['spec'],task.get('count'))
        if task['operation']=='dialogue':return self.dialogue(task['fixture'],'rebuild',task['deadline'])
        raise ValueError('Unknown study model operation')
