from baseline.contracts import checker_result, input_contract_error

class ContractValidator:
    checker = 'contract_validator'

    def validate(self, guardrail_input):
        try:
            error = input_contract_error(guardrail_input)
            if error is not None:
                return checker_result(self.checker, 'INVALID_INPUT', details={'error': error})
            return checker_result(self.checker)
        except Exception:
            # Do not include raw input or exception messages in returned/logged metadata.
            return checker_result(self.checker, 'ERROR', details={'error': 'CONTRACT_VALIDATION_UNAVAILABLE'})
