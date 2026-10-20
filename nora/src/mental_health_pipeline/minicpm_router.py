"""Pinned, search-only MiniCPM adapter with the exact supplied one-field prompt."""
import hashlib
import json
import re
import time

from .qwen_router import QwenJointRouter,QwenRoute,QwenRouterError

PROMPT_SHA256='4584aa7b5a03d0783e25b8b97cf0de6b9d3dd8bc7e6f65dbd44f0aa36edccf30'
SCHEMA={'type':'object','properties':{'q':{'type':'string'}},'required':['q'],'additionalProperties':False}


def validate_query(value):
    if not isinstance(value,dict) or set(value)!={'q'} or not isinstance(value['q'],str):
        raise QwenRouterError('minicpm_invalid_output')
    query=' '.join(value['q'].split())
    if len(query)>4096 or (query and len(query)<5):raise QwenRouterError('minicpm_invalid_output')
    return query


class MiniCPMRouter(QwenJointRouter):
    """Reuses loopback-only/no-proxy/no-redirect transport, not Qwen identity."""
    def __init__(self,prompt_path,*,model,expected_digest,package_version='v2',**kwargs):
        if not re.fullmatch('[0-9a-f]{64}',expected_digest):raise ValueError('MiniCPM digest must be pinned')
        if package_version not in {'v1','v2'}:raise ValueError('Unknown supplied package')
        super().__init__(prompt_path,model=model,**kwargs)
        self.expected_digest=expected_digest;self.backend_name='minicpm_'+package_version
        self.factual_skip_rescue=False

    def _prompt(self):
        try:raw=self.prompt_path.read_bytes()
        except OSError:raise QwenRouterError('minicpm_prompt_missing') from None
        if hashlib.sha256(raw).hexdigest()!=PROMPT_SHA256:raise QwenRouterError('minicpm_prompt_hash_mismatch')
        return raw.decode('utf-8')

    def _identity(self,timeout):
        data=self._request_json(self.tags_url,timeout=timeout);models=data.get('models')
        if not isinstance(models,list):raise QwenRouterError('minicpm_invalid_response')
        found=next((v for v in models if isinstance(v,dict) and self.model in (v.get('name'),v.get('model'))),None)
        if found is None:raise QwenRouterError('minicpm_model_missing')
        if found.get('digest','').removeprefix('sha256:')!=self.expected_digest:
            raise QwenRouterError('minicpm_model_digest_mismatch')

    def route(self,original_query):
        if not isinstance(original_query,str) or not original_query.strip():raise ValueError('Nonempty original message required')
        if len(original_query.encode('utf-8'))>16384:raise QwenRouterError('minicpm_input_limit')
        started=time.monotonic();deadline=started+self.timeout_seconds
        if not self._slot.acquire(timeout=self.timeout_seconds):raise QwenRouterError('minicpm_busy')
        try:
            prompt=self._prompt();remaining=deadline-time.monotonic()
            if remaining<=0:raise QwenRouterError('minicpm_timeout')
            self._identity(min(remaining,3.0));remaining=deadline-time.monotonic()
            if remaining<=0:raise QwenRouterError('minicpm_timeout')
            result=self._request_json(self.api_url,timeout=remaining,payload={
                'model':self.model,'stream':False,'think':False,'format':SCHEMA,'keep_alive':self.keep_alive,
                'options':{'temperature':0,'seed':42,'num_ctx':self.context_tokens,'num_predict':160,'num_gpu':self.num_gpu},
                'messages':[{'role':'system','content':prompt},{'role':'user','content':original_query}]})
            if time.monotonic()>deadline:raise QwenRouterError('minicpm_timeout')
            if result.get('done') is not True or result.get('done_reason')=='length':raise QwenRouterError('minicpm_incomplete_response')
            try:
                msg=result['message']
                if msg.get('role')!='assistant' or str(msg.get('thinking') or '').strip():raise QwenRouterError('minicpm_invalid_output')
                query=validate_query(json.loads(msg['content']))
            except (KeyError,TypeError,AttributeError,ValueError):raise QwenRouterError('minicpm_invalid_output') from None
            if not query and self.factual_skip_rescue:
                from .retrieval_recovery import clear_factual_mental_health_question
                if clear_factual_mental_health_question(original_query):
                    return QwenRoute('RETRIEVE',original_query,'application_factual_skip_rescue',
                        (time.monotonic()-started)*1000,
                        application_override='clear_factual_question_use_original',raw_decision='NO_RETRIEVE')
            return QwenRoute('RETRIEVE' if query else 'NO_RETRIEVE',query,
                'mental_health_knowledge' if query else 'model_no_retrieval',(time.monotonic()-started)*1000)
        finally:self._slot.release()


def build_minicpm_router(config):
    if config.rag_router_backend=='qwen_one_field':
        from .qwen_router import MODEL_DIGEST
        prompt=config.rag_project/'upstream/minicpm_2026-10-20/v2/scripts/minicpm_joint_router_prompt_v1.txt'
        router=MiniCPMRouter(prompt,model='qwen3:4b',expected_digest=MODEL_DIGEST,
            api_url=config.qwen_api_url,timeout_seconds=config.qwen_timeout_seconds,
            num_gpu=config.qwen_num_gpu,context_tokens=4096)
        if config.output_guard_backend in {'local_qwen','local_hybrid'}:
            from .local_output import JudgeSettings
            router.context_tokens=JudgeSettings.load(config.local_output_settings).context_tokens
        router.backend_name='qwen_one_field'
        router.factual_skip_rescue=True
        return router
    settings=json.loads((config.project_root/'config/minicpm_router.json').read_text(encoding='utf-8'))
    version=config.rag_router_backend.removeprefix('minicpm_')
    suffix='replacement_files' if version=='v1' else 'scripts'
    prompt=config.rag_project/'upstream/minicpm_2026-10-20'/version/suffix/'minicpm_joint_router_prompt_v1.txt'
    return MiniCPMRouter(prompt,model=settings['model'],expected_digest=settings['digest'],package_version=version,
        api_url=config.qwen_api_url,timeout_seconds=config.qwen_timeout_seconds,num_gpu=config.qwen_num_gpu,context_tokens=4096,keep_alive='5m')
