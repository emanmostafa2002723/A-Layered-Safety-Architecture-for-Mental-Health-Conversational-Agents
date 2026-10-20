import pytest
from mental_health_pipeline.output_contacts import has_generated_contact_details
from test_conversation_repair import ReplySequence,message
from test_orchestrator import pipeline,Crisis

@pytest.mark.parametrize('text',[
    'Call 9-8-8 now.', 'Call 911 now.', 'Text 9 8 8.',
    'Call +1 (555) 123-4567.', 'The crisis helpline is https://example.invalid/crisis',
])
def test_generated_contact_formats_are_not_trusted(text):
    assert has_generated_contact_details(text)

@pytest.mark.parametrize('text',[
    'Notice three things around you.', 'Ask your pharmacist about your dose.',
    'The year was 2026.', 'There were 25 sessions in the study.',
    'If you are in immediate danger, contact local emergency services now.',
])
def test_ordinary_numbers_and_generic_safety_guidance_are_not_contact_details(text):
    assert not has_generated_contact_details(text)

def test_greeting_with_generated_contacts_is_rewritten_before_display(tmp_path):
    generator=ReplySequence(['Hello. Call 9-8-8.', 'Hello! What would you like to talk about?'])
    app=pipeline(tmp_path,generator=generator)
    result=app.process(message('hi'))
    assert result.response=='Hello! What would you like to talk about?'
    assert 'Remove every telephone number' in generator.requests[-1].response_policy
    assert not result.used_fallback

def test_explicit_repeat_request_cannot_bypass_contact_check(tmp_path):
    generator=ReplySequence(['Call 9-8-8.', 'Call 9-8-8.'])
    result=pipeline(tmp_path,generator=generator).process(message('repeat that'))
    assert '9-8-8' not in result.response and result.used_fallback

def test_exact_application_resource_block_is_not_rejected(tmp_path):
    generator=ReplySequence(['I hear how difficult this is. Are you safe right now?'])
    app=pipeline(tmp_path,generator=generator,crisis=Crisis(tier='crisis',category='suicidal_ideation'))
    result=app.process(message('I have a plan to end my life now.'))
    assert result.resource_payload is not None
    assert len(generator.requests)==1

def test_extra_generated_contacts_are_rejected_even_with_application_block(tmp_path):
    generator=ReplySequence(['Call +1 (555) 123-4567. Are you safe right now?',
        'I hear how difficult this is. Are you safe right now?'])
    app=pipeline(tmp_path,generator=generator,crisis=Crisis(tier='crisis',category='suicidal_ideation'))
    result=app.process(message('I have a plan to end my life now.'))
    assert '555' not in result.response and len(generator.requests)==2
