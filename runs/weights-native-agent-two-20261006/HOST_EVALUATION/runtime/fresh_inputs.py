"""Explicit fresh random baseline and hash-bound local public-training inputs.

No network, maker calls, historical core, oracle weights or implicit defaults.
Report IDs stay reserved; their task files are never opened by this loader.
"""
import copy
import hashlib
import json
from pathlib import Path
import random
import re


def configuration(value):
    if type(value) is not dict or set(value) != {'manifest', 'sha256'}:
        raise ValueError('fresh inputs require exactly manifest and sha256')
    path, digest = value['manifest'], value['sha256']
    if (type(path) is not str or not Path(path).is_absolute()
            or type(digest) is not str or not re.fullmatch('[0-9a-f]{64}', digest)):
        raise ValueError('fresh manifest requires an absolute path and SHA256')
    return copy.deepcopy(value)


def _read(path, digest):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError('fresh input hash mismatch: ' + str(path))
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate fresh input field')
            result[key] = value
        return result
    return json.loads(payload, object_pairs_hook=unique), payload


def make_baseline(templates, seed):
    """Create declared random literals using the existing cell constructors."""
    import numpy as np
    import skillcore as S
    if (type(seed) is not int or seed < 0 or type(templates) is not list
            or not 1 <= len(templates) <= 8 or len(set(templates)) != len(templates)):
        raise ValueError('one to eight distinct registered templates and a seed required')
    rng = random.Random(seed)
    cells = []
    for template in templates:
        if template not in S.TEMPLATES or S.TEMPLATES[template]['niche'] == 'legacy':
            raise ValueError('fresh baseline cannot require a legacy template')
        params = {key: np.array([rng.uniform(-9, 9) for _ in range(int(np.prod(shape)))])
                  .reshape(shape).tolist() for key, shape in S.TEMPLATES[template]['shapes'].items()}
        cells.append(S.make_cell(template, params, niche='fresh_random',
                                 origin='fresh random baseline; not LLM-authored', seed=seed))
    return {'schema': 'cellforge-fresh-baseline/1', 'method': 'uniform[-9,9]',
            'seed': seed, 'templates': list(templates), 'llm_provenance': False,
            'cells': cells, 'bank': S.export_bank(cells, S.default_router(cells))}


def _task(value):
    if type(value) is not dict or set(value) != {'train', 'test'}:
        raise ValueError('training task requires train/test pairs')
    for panel in ('train', 'test'):
        if type(value[panel]) is not list or not 1 <= len(value[panel]) <= 20:
            raise ValueError('bounded nonempty task pairs required')
        for pair in value[panel]:
            if type(pair) is not dict or set(pair) != {'input', 'output'}:
                raise ValueError('training pair must have input/output grids')
            for grid in pair.values():
                if (type(grid) is not list or not 1 <= len(grid) <= 30
                        or type(grid[0]) is not list or not 1 <= len(grid[0]) <= 30
                        or any(type(row) is not list or len(row) != len(grid[0])
                               or any(type(v) is not int or not 0 <= v <= 9 for v in row) for row in grid)):
                    raise ValueError('invalid ARC grid')


def load(value):
    """Verify bytes and deterministic baseline before any campaign reservation."""
    config = configuration(value)
    manifest_path = Path(config['manifest'])
    manifest, manifest_bytes = _read(manifest_path, config['sha256'])
    if (type(manifest) is not dict or set(manifest) != {'schema', 'source', 'source_sha256', 'baseline'}
            or manifest['schema'] != 'cellforge-fresh-inputs/1'):
        raise ValueError('unsupported fresh inputs manifest')
    source_path = Path(manifest['source'])
    if not source_path.is_absolute():
        raise ValueError('absolute public-training source manifest required')
    source, source_bytes = _read(source_path, manifest['source_sha256'])
    if (source.get('schema') != 'cellforge-public-training-inputs/1'
            or source.get('repository') != 'https://github.com/arcprize/ARC-AGI-2'
            or not re.fullmatch('[0-9a-f]{40}', source.get('commit', ''))
            or source.get('license') != 'Apache-2.0'
            or source.get('evaluation_downloaded') is not False):
        raise ValueError('pinned official public-training source required')
    license_bytes = (source_path.parent / 'LICENSE').read_bytes()
    if hashlib.sha256(license_bytes).hexdigest() != source['license_sha256']:
        raise ValueError('dataset license bytes changed')
    baseline = manifest['baseline']
    if baseline != make_baseline(baseline['templates'], baseline['seed']):
        raise ValueError('fresh baseline differs from its declared random construction')
    panels = source['panels']
    if set(panels) != {'search_fitness', 'development_selection', 'report'}:
        raise ValueError('three explicit fresh task partitions required')
    seen, rows, report = set(), {}, []
    for panel, entries in panels.items():
        if type(entries) is not list or not 1 <= len(entries) <= 100:
            raise ValueError('bounded nonempty fresh partitions required')
        rows[panel] = []
        for item in entries:
            task_id = item['id']
            if not re.fullmatch('[0-9a-f]{8}', task_id) or task_id in seen:
                raise ValueError('invalid or overlapping fresh task IDs')
            seen.add(task_id)
            if panel == 'report':
                if item.get('downloaded') is not False:
                    raise ValueError('fresh report must remain reserved and unavailable')
                report.append(task_id)
                continue
            if item.get('downloaded') is not True:
                raise ValueError('development task is unavailable')
            path = source_path.parent / 'training' / (task_id + '.json')
            task, payload = _read(path, item['sha256'])
            if hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest() != item['git_blob']:
                raise ValueError('task differs from pinned upstream git blob')
            _task(task)
            rows[panel].append({'id': task_id, 'task': task})
    return {'config': config, 'manifest': manifest, 'manifest_bytes': manifest_bytes,
            'source_bytes': source_bytes, 'baseline': baseline, 'rows': rows, 'report_ids': report}
