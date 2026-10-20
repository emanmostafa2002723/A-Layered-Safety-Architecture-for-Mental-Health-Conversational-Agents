"""Owned finding snapshots; execution failure is distinct from a material violation."""
from copy import deepcopy
from dataclasses import dataclass
import json
from baseline.contracts import validate_checker_result, valid_individual_findings
from config.fusion_policy import canonical_reasons, effective_severity, execution_health, SEVERITY_RANK
from config.taxonomy import REASON_CODES

@dataclass(frozen=True)
class FindingCollection:
    by_name: dict
    violations: tuple
    reason_codes: tuple
    invalid: bool
    health_status: str = 'OK'
    reason_sources: object = None

def collect_findings(checker_results, route=None):
    if type(checker_results) is not list or len(checker_results) > 64:
        return FindingCollection({}, (), (), True, 'ERROR', {})
    by_name, findings, statuses, sources = {}, [], [], {}
    invalid = False
    for result in checker_results:
        if type(result) is not dict:
            invalid = True
            continue
        name = result.get('checker')
        recognizable = type(name) is str and 0 < len(name) <= 256
        valid = recognizable and validate_checker_result(result, name)
        if not valid:
            invalid = True
        else:
            snapshot = deepcopy(result)
            statuses.append(snapshot['status'])
            if name in by_name:
                invalid = True
            else:
                by_name[name] = snapshot
        # Malformed envelopes can still carry independently schema-valid findings.
        for finding in valid_individual_findings(result):
            finding['severity'] = effective_severity(finding['reason_code'], finding['severity'], route)
            findings.append(finding)
            sources.setdefault(finding['reason_code'], set()).add(name)
    if invalid:
        statuses.append('ERROR')
    findings.sort(key=lambda v: (-SEVERITY_RANK[v['severity']], REASON_CODES.index(v['reason_code']),
        json.dumps(v, sort_keys=True, allow_nan=False)))
    reasons = tuple(canonical_reasons(v['reason_code'] for v in findings))
    return FindingCollection(by_name, tuple(findings), reasons, invalid, execution_health(statuses),
        {code: tuple(sorted(sources[code])) for code in reasons})
