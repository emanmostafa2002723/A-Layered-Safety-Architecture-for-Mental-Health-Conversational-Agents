"""Adjudicated-corpus evaluation. Reports behavior without inventing acceptance gates.

This module never calls a model unless run_evaluation is explicitly invoked with
a configured service and a reviewed dataset. Reports contain no candidate/evidence text.
"""
from baseline.contracts import input_contract_error
from integration.application import _validated_delivery_response
from copy import deepcopy
from time import monotonic
from config.taxonomy import REASON_CODES
from config.routes import SUPPORTED_ROUTES
import math

def validate_dataset(dataset):
    if (type(dataset) is not dict or dataset.get('schema_version') != 'guardrail-evaluation-1.0'
            or dataset.get('acceptance_dataset') is not True
            or type(dataset.get('dataset_version')) is not str or not dataset['dataset_version'].strip()
            or type(dataset.get('cases')) is not list or not dataset['cases']):
        raise ValueError('An independently adjudicated dataset is required')
    identifiers = set()
    for case in dataset['cases']:
        if (type(case) is not dict or type(case.get('id')) is not str or not case['id'].strip()
                or len(case['id']) > 128 or case['id'] in identifiers
                or case.get('review_status') != 'adjudicated'
                or type(case.get('reviewer_id')) is not str or not case['reviewer_id'].strip()
                or type(case.get('language')) is not str or not case['language'].strip()
                or input_contract_error(case.get('input')) is not None
                or type(case.get('expected_codes')) is not list
                or any(type(c) is not str or c not in REASON_CODES for c in case['expected_codes'])
                or len(set(case['expected_codes'])) != len(case['expected_codes'])
                or type(case.get('expected_decision')) is not str
                or case['expected_decision'] not in {'ALLOW', 'BLOCK'}
                or case['expected_decision'] == 'ALLOW' and case['expected_codes']):
            raise ValueError('Invalid adjudicated evaluation case')
        identifiers.add(case['id'])

def run_evaluation(service, dataset):
    validate_dataset(dataset)
    dataset = deepcopy(dataset)
    counts = {code: {'true_positive': 0, 'false_positive': 0, 'false_negative': 0} for code in REASON_CODES}
    results = []
    unsafe_displays = failures = safe_blocks = 0
    routes, languages = set(), set()
    for case in dataset['cases']:
        started = monotonic()
        error = None
        exposed, candidate_released, raw_action = False, False, None
        try:
            delivered = service.process(deepcopy(case['input']))
            raw_action = delivered.delivery_action
            exposed = type(delivered.delivered_text) is str and bool(delivered.delivered_text)
            candidate_released = exposed and delivered.delivered_text == case['input']['generated_response']
            # Share the display boundary's identity/context/fallback checks. Keep
            # the raw exposure above so rejection cannot hide a leaked candidate.
            _validated_delivery_response(delivered, case['input'],
                runtime=getattr(service, 'runtime', None),
                registry=getattr(service, 'registry', None),
                rewriter=getattr(service, 'rewriter', None))
            observed = set(delivered.guardrail_result['reason_codes'])
            status, decision, action = (delivered.guardrail_result['guardrail_status'],
                delivered.guardrail_result['decision'], delivered.delivery_action)
        except Exception:
            observed, status, decision, action = set(), None, None, None
            error = 'EVALUATION_OBSERVATION_FAILED'
        elapsed_ms = (monotonic() - started) * 1000
        expected = set(case['expected_codes'])
        routes.add(case['input']['system_route']); languages.add(case['language'])
        unsafe_display = case['expected_decision'] == 'BLOCK' and (raw_action == 'DISPLAY'
            or candidate_released or exposed and error is not None)
        unsafe_displays += unsafe_display
        failures += error is not None or status != 'OK'
        safe_blocks += case['expected_decision'] == 'ALLOW' and action != 'DISPLAY'
        for code in REASON_CODES:
            if code in expected:
                counts[code]['true_positive' if code in observed else 'false_negative'] += 1
            elif code in observed:
                counts[code]['false_positive'] += 1
        results.append({'id': case['id'], 'route': case['input']['system_route'], 'language': case['language'],
            'guardrail_status': status, 'decision': decision, 'delivery_action': action,
            'expected_codes': sorted(expected), 'observed_codes': sorted(observed),
            'unsafe_display': unsafe_display, 'elapsed_ms':elapsed_ms, 'observation_error':error})
    for metrics in counts.values():
        detected = metrics['true_positive'] + metrics['false_negative']
        predicted = metrics['true_positive'] + metrics['false_positive']
        metrics['recall'] = metrics['true_positive'] / detected if detected else None
        metrics['precision'] = metrics['true_positive'] / predicted if predicted else None
    latency = sorted(row['elapsed_ms'] for row in results)
    return {'dataset_version': dataset['dataset_version'], 'case_count': len(results),
        'latency_ms':{'min':latency[0], 'p50':latency[math.ceil(len(latency)*0.5)-1],
            'p95':latency[math.ceil(len(latency)*0.95)-1], 'max':latency[-1]},
        'safe_case_count': sum(case['expected_decision'] == 'ALLOW' for case in dataset['cases']),
        'unsafe_display_count': unsafe_displays, 'execution_failure_count': failures, 'safe_block_count': safe_blocks,
        'per_code': counts, 'routes_covered': sorted(routes), 'languages_covered': sorted(languages),
        'missing_code_coverage': [code for code, metrics in counts.items()
            if not metrics['true_positive'] and not metrics['false_negative']],
        'missing_route_coverage': sorted(SUPPORTED_ROUTES - routes),
        'acceptance_status': 'REQUIRES_PROJECT_APPROVED_GATES', 'cases': results}

def validate_acceptance_gates(gates):
    """Thresholds must be supplied by the project; there are no invented defaults."""
    required = {'approval_id', 'min_cases', 'min_safe_cases', 'min_positive_per_code', 'required_languages',
        'minimum_recall', 'minimum_precision', 'max_execution_failures', 'max_unsafe_displays', 'max_safe_blocks'}
    if (type(gates) is not dict or set(gates) != required
            or type(gates['approval_id']) is not str or not gates['approval_id'].strip()):
        raise ValueError('Explicit project-approved acceptance gates are required')
    for key in ('min_cases', 'min_safe_cases', 'min_positive_per_code', 'max_execution_failures', 'max_unsafe_displays', 'max_safe_blocks'):
        if type(gates[key]) is not int or gates[key] < (1 if key.startswith('min_') else 0):
            raise ValueError('Invalid count threshold')
    if (type(gates['required_languages']) is not list or not gates['required_languages']
            or any(type(v) is not str or not v.strip() for v in gates['required_languages'])):
        raise ValueError('Explicit evaluation languages are required')
    for key in ('minimum_recall', 'minimum_precision'):
        values = gates[key]
        if type(values) is not dict or set(values) != set(REASON_CODES):
            raise ValueError('Every frozen reason code requires an explicit threshold')
        if any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
            raise ValueError('Invalid model threshold')

def apply_acceptance_gates(report, gates):
    validate_acceptance_gates(gates)
    failures = []
    if report['case_count'] < gates['min_cases']: failures.append('INSUFFICIENT_CASES')
    if report['safe_case_count'] < gates['min_safe_cases']: failures.append('INSUFFICIENT_SAFE_CONTROLS')
    if report['missing_route_coverage']: failures.append('MISSING_ROUTE_COVERAGE')
    if not set(gates['required_languages']) <= set(report['languages_covered']): failures.append('MISSING_LANGUAGE_COVERAGE')
    for metric, threshold in (('execution_failure_count', 'max_execution_failures'),
            ('unsafe_display_count', 'max_unsafe_displays'), ('safe_block_count', 'max_safe_blocks')):
        if report[metric] > gates[threshold]: failures.append(metric.upper())
    for code, metrics in report['per_code'].items():
        if metrics['true_positive'] + metrics['false_negative'] < gates['min_positive_per_code']:
            failures.append(code + ':INSUFFICIENT_POSITIVE_CASES')
        for metric, threshold in (('recall', 'minimum_recall'), ('precision', 'minimum_precision')):
            if metrics[metric] is None or metrics[metric] < gates[threshold][code]:
                failures.append(code + ':' + metric.upper())
    return {**report, 'acceptance_status': 'FAIL' if failures else 'PASS',
        'acceptance_approval_id': gates['approval_id'], 'gate_failures': failures}
