from dataclasses import replace
from types import SimpleNamespace
import pytest
from mental_health_pipeline.query_semantics import contextual_search_query,preserve_query_meaning,evidence_query_terms
from mental_health_pipeline.conversation_quality import reasked_answered_reason
from test_orchestrator import pipeline,Generator
from test_conversation_repair import ReplySequence,message


def history(text):return ({'role':'user','content':text},{'role':'assistant','content':'Here is a general explanation.'})


@pytest.mark.parametrize('latest,previous,expected',[
    ('does it help with anxiety?','What is CBT?','does cognitive behavioral therapy help with anxiety?'),
    ('and for depression?','What is CBT?','and for depression? (about cognitive behavioral therapy)'),
    ('can it work without medication?','What is EMDR?','can EMDR work without medication?'),
    ('does it help?','What is CBT versus DBT?',None),
    ('does it help?','I feel nervous',None),
    ('what about my exam?','What is CBT?',None),
    ('is it sunny?','What is CBT?',None),
    ('what about DBT?','What is CBT?',None),
    ('I want someone to listen.','What is CBT?',None),
])
def test_search_followup_is_explicit_user_anchored_and_conservative(latest,previous,expected):
    query,used=contextual_search_query(latest,history(previous))
    assert query==(expected or latest) and used==(expected is not None)


def test_search_never_uses_assistant_invented_subject():
    assert contextual_search_query('does it help?',[{'role':'assistant','content':'Try CBT.'}])==('does it help?',False)


@pytest.mark.parametrize('original,rewrite,restored',[
    ('cofee late shift make sleeping worse how?','What are the consequences of not getting enough sleep?',True),
    ('What is CBT for worry?','What is cognitive behavioral therapy for anxiety?',False),
    ('How does CBT work without medication?','How does CBT work with medication?',True),
    ('CBT for panic attacks?','CBT for anxiety?',True),
    ('PTSD and flashbacks?','How can trauma cause flashbacks in PTSD?',False),
])
def test_rewrite_subject_loss_and_clinical_negation(original,rewrite,restored):
    query,reasons=preserve_query_meaning(original,rewrite)
    assert query==(original if restored else rewrite) and bool(reasons)==restored


def test_alias_expansion_ignores_question_glue():
    terms=evidence_query_terms('Can you explain CBT for someone who worries?')
    assert {'cognitive','behavioral','therapy','worries'}<=terms
    assert not {'can','you','for','someone','explain'}&terms


def test_paraphrased_answered_question_gets_one_rewrite_not_a_stale_display(tmp_path):
    first='What is making you nervous?'
    stale='That sounds difficult. Why do you feel nervous?'
    fresh='An exam tomorrow can feel like a lot. Which topic would you like to review first?'
    generator=ReplySequence([first,stale,fresh]);app=pipeline(tmp_path,generator=generator)
    try:
        app.process(message('I feel nervous'))
        result=app.process(message('I have an exam tomorrow',2))
        assert result.response==fresh and not result.used_fallback
        assert 'already answers' in generator.requests[-1].response_policy
    finally:app.close()


def test_repeated_safety_questions_are_not_treated_as_answered_reason():
    packet=SimpleNamespace(system_route=SimpleNamespace(value='crisis_response'),
        crisis_tier=SimpleNamespace(value='crisis'),crisis_labels=('suicidal_ideation',),boundary_route=SimpleNamespace(value='allow'))
    assert not reasked_answered_reason('because I have an exam','Why do you feel nervous?',
        history('What makes you nervous?'),packet)


def test_contextual_search_does_not_replace_classifier_or_generator_input(tmp_path):
    from mental_health_pipeline.rag import RetrievalResult,RetrievalMode
    generator=ReplySequence(['CBT is a therapy approach.','It can be discussed with a qualified clinician.'])
    app=pipeline(tmp_path,generator=generator);calls=[]
    class Rag:
        def retrieve(self,text,route):
            calls.append(text);return RetrievalResult(text,str(route),RetrievalMode.STANDARD)
    app.config=replace(app.config,enable_rag=True)
    app.components=replace(app.components,rag=Rag())
    try:
        app.process(message('What is CBT?'))
        app.process(message('does it help with anxiety?',2))
        assert calls[-1]=='does cognitive behavioral therapy help with anxiety?'
        assert generator.requests[-1].user_text=='does it help with anxiety?'
    finally:app.close()


def test_phoenix_restores_original_query_before_real_retrieval_when_rewrite_loses_subject():
    from test_qwen_router import backend,StubRouter,StubRetriever
    from mental_health_pipeline.qwen_router import QwenRoute
    original='cofee late shift make sleeping worse how?'
    b=backend(StubRouter(QwenRoute('RETRIEVE','What are the consequences of not getting enough sleep?','mental_health_knowledge')))
    b._retriever=StubRetriever();result=b.retrieve(original)
    assert b._retriever.calls==[(original,3,True)]
    assert result['routing_diagnostics']['query_integrity']=='original_query_restored'


def test_specific_next_question_is_not_a_reasked_generic_reason():
    from test_conditional_prompt import packet
    h=({'role':'assistant','content':'What makes you nervous?'},)
    assert not reasked_answered_reason('I have an exam tomorrow','Which exam topic feels hardest?',h,packet())


def test_whole_history_is_not_used_to_guess_a_subject_after_topic_change():
    h=history('What is CBT?')+history('I have an interview tomorrow.')
    assert contextual_search_query('does it help?',h)==('does it help?',False)


def test_evidence_compression_prefers_subject_over_question_glue():
    from mental_health_pipeline.generation import MistralGenerator,GenerationConfig,GenerationRequest
    from mental_health_pipeline.rag import RetrievalResult,RetrievalMode,RetrievedDocument,Citation
    text='Someone can explain what this is for you. Cognitive behavioral therapy links thoughts and behavior.'
    retrieval=RetrievalResult('What is CBT?','normal_chat',RetrievalMode.STANDARD,
        documents=(RetrievedDocument(text,Citation('source-1','test','Title')),))
    tokenizer=lambda text,**kwargs:{'input_ids':text.split()}
    gen=MistralGenerator(GenerationConfig(evidence_token_budget=9))
    result=gen._bounded_evidence(tokenizer,GenerationRequest('Can you explain CBT for someone?', '',retrieval=retrieval))
    assert result.retrieval.documents[0].text=='Cognitive behavioral therapy links thoughts and behavior.'


@pytest.mark.parametrize('user,reply,expected',[
    ('Actually I meant an interview, not an exam.','Which exam subject is hardest?',True),
    ('It is a test, not an interview.','What job are you interviewing for?',True),
    ('Actually I meant an interview, not an exam.','What part of the interview worries you?',False),
    ('I have both an interview and an exam.','Which exam subject is hardest?',False),
    ('Actually I meant an interview, not an exam.','We can focus on what matters to you.',False),
])
def test_explicit_topic_correction_check_is_narrow(user,reply,expected):
    from mental_health_pipeline.conversation_quality import ignored_topic_correction
    from test_conditional_prompt import packet
    assert ignored_topic_correction(user,reply,packet())==expected


def test_subject_filter_does_not_substitute_another_treatment_or_disorder_scope():
    from mental_health_pipeline.query_semantics import subject_aligned_documents
    from mental_health_pipeline.rag import RetrievedDocument,Citation
    def doc(text,title):return RetrievedDocument(text,Citation('source-1','test',title))
    relevant=doc('CBT teaches different ways of thinking and behaving.','Anxiety overview')
    irrelevant=doc('ECT is a biological treatment.','Treatment overview')
    narrow=doc('CBT is discussed here.','Borderline Personality Disorder')
    assert subject_aligned_documents('How does CBT help anxiety?',[relevant,irrelevant,narrow])==[relevant]
    assert subject_aligned_documents('What is EMDR?',[irrelevant])==[]
    assert subject_aligned_documents('Compare CBT and EMDR',[relevant,irrelevant])==[relevant,irrelevant]


@pytest.mark.parametrize('text,expected',[
    ('Could coffee near the end of a shift affect sleep?',True),
    ('Can caffeine disrupt sleep?',True),
    ('Can you help me? I cannot sleep.',False),
    ('Can we talk? I want someone to listen. I cannot sleep.',False),
    ('Coffee went cold. What is the weather?',False),
])
def test_factor_effect_recovery_is_not_a_generic_help_request(text,expected):
    from mental_health_pipeline.retrieval_recovery import clear_factual_mental_health_question
    assert clear_factual_mental_health_question(text)==expected


def test_long_copied_reply_with_small_new_tail_gets_one_rewrite(tmp_path):
    old=('An exam can feel demanding. Choose a topic, work through a practice problem, '
         'then take a short pause. You can focus on one manageable step rather than trying to cover everything.')
    stale=old+' You have an hour to review.'
    fresh='With one hour for maths, which topic would you like to practice first? We can work through one example together.'
    generator=ReplySequence([old,stale,fresh]);app=pipeline(tmp_path,generator=generator)
    try:
        app.process(message('I have an exam tomorrow'))
        result=app.process(message('It is maths and I have only an hour.',2))
        assert result.response==fresh and not result.used_fallback
        assert len(generator.requests)==3 and 'Write a fresh answer' in generator.requests[-1].response_policy
    finally:app.close()


def test_small_shared_acknowledgment_is_not_a_near_copy():
    from mental_health_pipeline.conversation_quality import near_copied_reply
    from test_conditional_prompt import packet
    assert not near_copied_reply('That sounds difficult. Which subject is it?',
        'That sounds difficult. What is happening?',packet())


def test_followup_chain_retains_user_subject_but_stops_at_topic_change():
    h=history('What is CBT?')+history('does it help with anxiety?')
    assert contextual_search_query('and for depression?',h)==('and for depression? (about cognitive behavioral therapy)',True)
    h+=history('I have an interview tomorrow.')
    assert contextual_search_query('does it help with anxiety?',h)==('does it help with anxiety?',False)


def test_therapy_medication_comparison_is_not_filtered_as_therapy_only():
    from mental_health_pipeline.query_semantics import subject_aligned_documents
    from mental_health_pipeline.rag import RetrievedDocument,Citation
    doc=RetrievedDocument('Antidepressants are discussed here.',Citation('source-1','test','Medication overview'))
    assert subject_aligned_documents('Compare CBT and antidepressants',[doc])==[doc]


def test_small_topic_word_swaps_do_not_make_a_recycled_paragraph_new():
    from mental_health_pipeline.conversation_quality import near_copied_reply
    from test_conditional_prompt import packet
    old=('Nervousness is a common feeling. A few things that might help: deep breathing '
        '(4 seconds in, 4 seconds hold, 4 seconds out), grounding (name 5 things you can see, '
        '4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste), or a short walk outside. '
        "If it's persistent, consider reaching out to a professional.")
    copied=old.replace('Nervousness is a common feeling','Exam nerves are normal').replace('a professional','a study group or a tutor')+' Good luck!'
    assert near_copied_reply(copied,old,packet())


def test_filtered_sources_have_contiguous_display_ids_without_losing_provenance():
    from mental_health_pipeline.rag import RAGService,RetrievedDocument,Citation
    from test_rag_generation import RecordingBackend
    def doc(text,index):return RetrievedDocument(text,Citation(f'source-{index}','test','Overview',child_id=f'child-{index}'))
    service=RAGService(RecordingBackend([doc('CBT is discussed here.',1),doc('ECT is discussed here.',2),doc('More about CBT.',3)]))
    result=service.retrieve('What is CBT?','normal_chat')
    assert [d.citation.source_id for d in result.documents]==['source-1','source-2']
    assert [d.citation.child_id for d in result.documents]==['child-1','child-3']


def test_compression_reindexes_only_display_ids_when_an_excerpt_drops():
    from mental_health_pipeline.generation import MistralGenerator,GenerationConfig,GenerationRequest
    from mental_health_pipeline.rag import RetrievalResult,RetrievalMode,RetrievedDocument,Citation
    docs=(RetrievedDocument(' '.join(['lengthy']*30),Citation('source-1','test','Long',child_id='long')),
          RetrievedDocument('CBT links thoughts and behavior.',Citation('source-3','test','Short',child_id='short')))
    result=RetrievalResult('What is CBT?','normal_chat',RetrievalMode.STANDARD,documents=docs)
    gen=MistralGenerator(GenerationConfig(evidence_token_budget=12))
    token=lambda text,**kwargs:{'input_ids':text.split()}
    bounded=gen._bounded_evidence(token,GenerationRequest('What is CBT?','',retrieval=result)).retrieval
    assert len(bounded.documents)==1
    assert bounded.documents[0].citation.source_id=='source-1'
    assert bounded.documents[0].citation.child_id=='short'


def test_console_exposes_effective_source_mapping_not_unapproved_draft():
    from mental_health_pipeline.testing_console import _summary
    from mental_health_pipeline.generation import GenerationResult,GenerationRequest
    from mental_health_pipeline.rag import RetrievalResult,RetrievalMode,RetrievedDocument,Citation
    effective=RetrievalResult('CBT','normal_chat',RetrievalMode.STANDARD,documents=(
        RetrievedDocument('Selected reference.',Citation('source-1','test','Title',child_id='original-child-3')),))
    result=GenerationResult('SECRET UNAPPROVED DRAFT',(),1,1,True,effective_retrieval=effective,evidence_snapshot_available=True)
    summary=_summary('generator',result,(GenerationRequest('CBT?',''),))
    assert summary['effective_evidence_sources'][0]['child_id']=='original-child-3'
    assert 'SECRET UNAPPROVED DRAFT' not in str(summary)


@pytest.mark.parametrize('query,title',[
    ('What is CBT for BPD?','Borderline Personality Disorder'),
    ('How is CBT used for OCD?','Obsessive-Compulsive Disorder'),
])
def test_source_scope_filter_respects_explicit_disorder_acronyms(query,title):
    from mental_health_pipeline.query_semantics import subject_aligned_documents
    from mental_health_pipeline.rag import RetrievedDocument,Citation
    doc=RetrievedDocument('CBT is discussed here.',Citation('source-1','test',title))
    assert subject_aligned_documents(query,[doc])==[doc]
