import pytest

from mental_health_pipeline.schemas import (
    ContractError,
    PipelineRequest,
    RequestScope,
    RequestValidationError,
)


def test_valid_request_keeps_location_outside_message():
    request = PipelineRequest(
        latest_user_message="I feel anxious.",
        session_id="session-1",
        scope=RequestScope(
            age_eligibility_18_plus=True,
            preferred_language="en",
            current_location_country="ca",
        ),
    )
    request.validate()
    assert request.scope.normalized_country() == "CA"
    assert "CA" not in request.latest_user_message


@pytest.mark.parametrize("message", ["", "   "])
def test_empty_message_rejected(message):
    request = PipelineRequest(
        latest_user_message=message,
        session_id="s1",
        scope=RequestScope(age_eligibility_18_plus=True),
    )
    with pytest.raises(RequestValidationError):
        request.validate()


def test_invalid_crisis_contract_rejected():
    from mental_health_pipeline.schemas import CrisisDecision

    with pytest.raises(ContractError):
        CrisisDecision.from_mapping({"tier": "safe", "classifier": {}})
