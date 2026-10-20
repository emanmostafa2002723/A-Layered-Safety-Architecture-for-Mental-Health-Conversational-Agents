"""Explicit real-model acceptance runner. Missing resources produce BLOCKED.

Run preflight with python -m integration.acceptance. Add --execute only with
approved provider configuration, reviewed data and project acceptance gates.
No provider is contacted by preflight. Reports never contain credentials/text.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from integration.evaluation import validate_dataset, validate_acceptance_gates, run_evaluation, apply_acceptance_gates
from integration.service import GuardrailService, ProviderSpec
from providers.http_transport import HTTPConfig


def _provider(data):
    required = {'endpoint', 'model_id', 'model_revision', 'deployment_approved'}
    optional = {'api_key_env', 'structured_output'}
    if type(data) is not dict or not required <= set(data) or set(data) - required - optional:
        raise ValueError('Invalid provider configuration')
    if data['deployment_approved'] is not True:
        raise ValueError('Provider is unapproved')
    spec = ProviderSpec(HTTPConfig(data['endpoint'], data['model_id'],
        api_key_env=data.get('api_key_env'), structured_output=data.get('structured_output', True)),
        data['model_revision'], True)
    if spec.http.api_key_env is not None and not os.environ.get(spec.http.api_key_env):
        raise ValueError('Provider credential is unavailable')
    return spec


def preflight(configuration=None, dataset=None):
    """Validate everything before constructing a service or making any request."""
    blockers = []
    if configuration is None:
        blockers.append('APPROVED_PROVIDER_CONFIGURATION_AND_GATES_MISSING')
    else:
        try:
            if type(configuration) is not dict or set(configuration) != {'semantic', 'factuality', 'acceptance_gates'}:
                raise ValueError('Invalid acceptance configuration')
            validate_acceptance_gates(configuration['acceptance_gates'])
            _provider(configuration['semantic']); _provider(configuration['factuality'])
        except Exception:
            blockers.append('PROVIDER_CONFIGURATION_OR_GATES_INVALID_OR_UNAPPROVED')
    if dataset is None:
        blockers.append('INDEPENDENTLY_REVIEWED_DATASET_MISSING')
    else:
        try:
            validate_dataset(dataset)
        except Exception:
            blockers.append('DATASET_INVALID_OR_NOT_ADJUDICATED')
    # Coverage gates must pass before sending even the first case to a provider.
    if not blockers:
        from config.taxonomy import REASON_CODES
        from config.routes import SUPPORTED_ROUTES
        gates = configuration['acceptance_gates']; cases = dataset['cases']
        if (len(cases) < gates['min_cases']
                or sum(c['expected_decision'] == 'ALLOW' for c in cases) < gates['min_safe_cases']
                or {c['input']['system_route'] for c in cases} != SUPPORTED_ROUTES
                or not set(gates['required_languages']) <= {c['language'] for c in cases}
                or any(sum(code in c['expected_codes'] for c in cases) < gates['min_positive_per_code']
                    for code in REASON_CODES)):
            blockers.append('INSUFFICIENT_REVIEWED_DATASET_COVERAGE')
        factual = [c for c in cases if c['input']['response_policy']['factual_verification']['required']]
        if (not any('UNSUPPORTED_MEDICAL_CLAIM' in c['expected_codes'] for c in factual)
                or not any(c['expected_decision'] == 'ALLOW' and any(
                    e['review_status'] == 'approved' and e['source_id'] and e['source_version']
                    for e in c['input']['verification_evidence']) for c in factual)):
            blockers.append('FACTUALITY_EVALUATION_CASES_MISSING')
    return {'schema_version':'guardrail-acceptance-readiness-1.0',
        'status':'BLOCKED' if blockers else 'READY_NOT_RUN', 'blockers':blockers,
        'real_model_evaluation_executed':False}


def evaluate(configuration=None, dataset=None, *, execute=False):
    ready = preflight(configuration, dataset)
    if ready['status'] == 'BLOCKED' or not execute:
        return ready
    # Own the approved inputs; later caller mutation cannot change a run.
    configuration, dataset = deepcopy(configuration), deepcopy(dataset)
    try:
        service = GuardrailService(semantic=_provider(configuration['semantic']),
            factuality=_provider(configuration['factuality']))
        report = apply_acceptance_gates(run_evaluation(service, dataset), configuration['acceptance_gates'])
        report['models'] = {key:{field:configuration[key][field] for field in ('model_id','model_revision')}
            for key in ('semantic','factuality')}
        report['dataset_sha256'] = hashlib.sha256(json.dumps(dataset,sort_keys=True,
            ensure_ascii=False,allow_nan=False).encode('utf-8')).hexdigest()
        return {**report, 'real_model_evaluation_executed':True}
    except Exception:
        return {'status':'FAIL', 'blockers':['ACCEPTANCE_EXECUTION_FAILED'],
            'real_model_evaluation_executed':True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config'); parser.add_argument('--dataset')
    parser.add_argument('--output'); parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    try:
        configuration = json.loads(Path(args.config).read_text(encoding='utf-8')) if args.config else None
        dataset = json.loads(Path(args.dataset).read_text(encoding='utf-8')) if args.dataset else None
        report = evaluate(configuration, dataset, execute=args.execute)
    except Exception:
        report = {'status':'BLOCKED','blockers':['ACCEPTANCE_INPUT_FILES_UNAVAILABLE_OR_INVALID'],
            'real_model_evaluation_executed':False}
    raw = json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)
    if args.output:
        Path(args.output).write_text(raw+'\n',encoding='utf-8')
    else:
        print(raw)
    return 0 if report.get('acceptance_status') == 'PASS' or report.get('status') == 'READY_NOT_RUN' else 2


if __name__ == '__main__':
    raise SystemExit(main())
