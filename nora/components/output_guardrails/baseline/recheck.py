"""Prepare the next candidate; orchestration must still invoke the complete gate."""
from copy import deepcopy
from baseline.contracts import valid_final_result
from validators.contract_validator import ContractValidator

def prepare_rewrite_input(previous_input, previous_result, rewritten_response, config=None):
    if ContractValidator().validate(previous_input)['status'] != 'OK':
        raise ValueError('Invalid original guardrail input')
    if not valid_final_result(previous_result, previous_input, config):
        raise ValueError('Invalid original guardrail result')
    if (previous_result['decision'], previous_result['action']) != ('BLOCK', 'REWRITE'):
        raise ValueError('Rewrite was not authorized')
    if not isinstance(rewritten_response, str) or not rewritten_response.strip():
        raise ValueError('Invalid rewritten candidate')
    new_input = deepcopy(previous_input)
    new_input['generated_response'] = rewritten_response
    new_input['execution_context']['rewrite_attempt'] += 1
    if ContractValidator().validate(new_input)['status'] != 'OK':
        raise ValueError('Rewritten candidate violates input contract')
    return new_input
