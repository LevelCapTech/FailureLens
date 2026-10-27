"""Offline Gate A pilot verification with stdlib and existing quality gates."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / 'datasets/schemas/record.schema.json').read_text())
RECORD_PATH = ROOT / 'datasets/generated/sample_project_empty_average_gatea_0001.json'
MANIFEST_PATH = ROOT / 'datasets/generated/pilot_manifest.json'


def structural_errors(value: object, definition: dict, path: str = '$') -> list[str]:
    if '$ref' in definition:
        definition = SCHEMA['$defs'][definition['$ref'].split('/')[-1]]
    if 'oneOf' in definition:
        if any(not structural_errors(value, choice, path) for choice in definition['oneOf']):
            return []
        return [f'{path}: no oneOf branch matched']
    kind = definition.get('type')
    if isinstance(kind, list):
        if any(not structural_errors(value, {**definition, 'type': option}, path) for option in kind):
            return []
        return [f'{path}: type mismatch']
    types = {'object': dict, 'array': list, 'string': str, 'integer': int, 'number': (int, float), 'null': type(None), 'boolean': bool}
    if kind and (not isinstance(value, types[kind]) or (kind in ('integer', 'number') and isinstance(value, bool))):
        return [f'{path}: expected {kind}']
    errors: list[str] = []
    if kind == 'object':
        assert isinstance(value, dict)
        properties = definition.get('properties', {})
        errors += [f'{path}.{key}: missing' for key in definition.get('required', []) if key not in value]
        if definition.get('additionalProperties') is False:
            errors += [f'{path}.{key}: unknown' for key in value if key not in properties]
        for key in value.keys() & properties.keys():
            errors += structural_errors(value[key], properties[key], f'{path}.{key}')
    if kind == 'array':
        assert isinstance(value, list)
        errors += [error for index, item in enumerate(value) for error in structural_errors(item, definition['items'], f'{path}[{index}]')]
    if 'enum' in definition and value not in definition['enum']:
        errors.append(f'{path}: not in enum')
    if 'minimum' in definition and isinstance(value, (int, float)) and value < definition['minimum']:
        errors.append(f'{path}: below minimum')
    return errors


record = json.loads(RECORD_PATH.read_text())
manifest = json.loads(MANIFEST_PATH.read_text())
errors = structural_errors(record, SCHEMA)
quality_path = ROOT / 'packages/python/src/failurelens/quality.py'
spec = importlib.util.spec_from_file_location('failurelens_quality_pilot', quality_path)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
result = module.validate_record_set([record])[0]
# A one-repository pilot can only occupy one side; no held-out eval is formed.
train_repos = {record['source_repository']} if manifest['split'] == 'train' else set()
eval_repos = {record['source_repository']} if manifest['split'] == 'eval' else set()
split_overlap = sorted(train_repos & eval_repos)
print(json.dumps({'structural': {'passed': int(not errors), 'rejected': int(bool(errors)), 'errors': errors}, 'semantic': {'passed': int(result.passed), 'rejected': int(not result.passed), 'reasons': [reason.value for reason in result.rejection_reasons]}, 'split': {'train_repositories': sorted(train_repos), 'eval_repositories': sorted(eval_repos), 'overlap': split_overlap}}, indent=2))
if errors or not result.passed or split_overlap:
    raise SystemExit(1)
