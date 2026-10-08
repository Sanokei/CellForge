"""Offline lossless transport audit of already-saved tensors, never authoring.

Illustrative encoding only: no current importer accepts it, no model calls,
no runtime execution, no default-filled weights, no changed historical files.
"""
import hashlib
import json
import math
from pathlib import Path

VOCAB = [-1, -.5, -.25, 0, .25, .5, 1]
FORMAT = 'nca-author-flat-tensors/1'


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def checked_layout(layout):
    if not isinstance(layout, list) or not layout:
        raise ValueError('nonempty fixed layout required')
    names = []
    for entry in layout:
        if set(entry) != {'name', 'shape', 'positions'}:
            raise ValueError('exact layout fields required')
        if not isinstance(entry['name'], str) or not entry['name']:
            raise ValueError('tensor name required')
        shape = entry['shape']
        if not isinstance(shape, list) or not shape or any(type(n) is not int or n < 1 for n in shape):
            raise ValueError('positive dimensions required')
        if type(entry['positions']) is not int or entry['positions'] != math.prod(shape):
            raise ValueError('position count does not match shape')
        names.append(entry['name'])
    if names != sorted(set(names)):
        raise ValueError('unique lexicographically ordered names required')
    return sum(e['positions'] for e in layout)


def flatten(value, shape):
    if not shape:
        if type(value) not in (int, float) or not math.isfinite(value) or value not in VOCAB:
            raise ValueError('finite literal in frozen vocabulary required')
        return [value]
    if not isinstance(value, list) or len(value) != shape[0]:
        raise ValueError('exact nested shape required')
    result = []
    for child in value:
        result.extend(flatten(child, shape[1:]))
    return result


def encode(tensors, layout):
    checked_layout(layout)
    if type(tensors) is not dict or set(tensors) != {e['name'] for e in layout}:
        raise ValueError('tensor roster mismatch')
    values = []
    for entry in layout:
        values.extend(flatten(tensors[entry['name']], entry['shape']))
    return {'schema': FORMAT, 'layout_sha256': digest(layout), 'values': values}


def decode(payload, layout):
    count = checked_layout(layout)
    if type(payload) is not dict or set(payload) != {'schema', 'layout_sha256', 'values'}:
        raise ValueError('exact illustrative envelope fields required')
    if payload['schema'] != FORMAT or payload['layout_sha256'] != digest(layout):
        raise ValueError('fixed layout/schema mismatch')
    values = payload['values']
    if not isinstance(values, list) or len(values) != count:
        raise ValueError('all positions required; no defaults or sparse edits')
    flatten(values, [count])
    offset = 0
    def reshape(shape):
        nonlocal offset
        if not shape:
            value = values[offset]
            offset += 1
            return value
        return [reshape(shape[1:]) for _ in range(shape[0])]
    tensors = {e['name']: reshape(e['shape']) for e in layout}
    assert offset == count
    return tensors


def illustrative_schema(layout):
    count = checked_layout(layout)
    return {'type': 'object', 'additionalProperties': False,
            'properties': {'schema': {'const': FORMAT},
                           'layout_sha256': {'const': digest(layout)},
                           'values': {'type': 'array', 'minItems': count, 'maxItems': count,
                                      'items': {'type': 'number', 'enum': VOCAB}}},
            'required': ['schema', 'layout_sha256', 'values']}


def audit(root):
    run = root / 'runs/weights-nca-author-tranche-20261008'
    response_raw = (run/'attempts/01/RAW_RESPONSE.bin').read_bytes()
    if hashlib.sha256(response_raw).hexdigest() != 'ea8591a8811eed9a65416cc98edee40f67c9dec7b0c8ade2a4165c42f74e9335':
        raise ValueError('saved response identity changed')
    packet_raw = (run/'FOUNDER_PACKET.json').read_bytes()
    if hashlib.sha256(packet_raw).hexdigest() != 'f26d3567f9d02903bbb525072803ed4713714108665a30458df99f2cd1532e37':
        raise ValueError('frozen packet identity changed')
    layout = json.loads(packet_raw)['tensor_schema']
    original = json.loads(response_raw)
    proposal = encode(original['tensors'], layout)
    restored = {'schema': original['schema'], 'tensors': decode(proposal, layout)}
    encoded = canonical(proposal)
    schema = canonical(illustrative_schema(layout))
    old_schema = (run/'RESPONSE_SCHEMA.json').read_bytes()
    if hashlib.sha256(old_schema).hexdigest() != 'd7ebc66e50abf32d43db94d4984201725d8dc8c122f2795a2af57aef1d272f72':
        raise ValueError('frozen response schema identity changed')
    return {'schema': 'nca-offline-encoding-audit/1',
            'status': 'illustrative_only_not_accepted_by_any_current_run',
            'source_response_sha256': hashlib.sha256(response_raw).hexdigest(),
            'source_packet_sha256': hashlib.sha256(packet_raw).hexdigest(),
            'source_schema_sha256': hashlib.sha256(old_schema).hexdigest(),
            'layout_sha256': digest(layout), 'arrays': len(layout),
            'literal_positions_before_and_after': checked_layout(layout),
            'tensor_values_and_shapes_roundtrip_exact': restored == original,
            'canonical_original_bytes_roundtrip_exact': canonical(restored) == response_raw,
            'original_response_bytes': len(response_raw), 'illustrative_response_bytes': len(encoded),
            'response_bytes_saved': len(response_raw)-len(encoded),
            'original_schema_bytes': len(old_schema), 'illustrative_schema_bytes': len(schema),
            'original_packet_layout_bytes': len(canonical(layout)),
            'model_calls': 0, 'runtime_predictions': 0, 'new_numeric_values': 0,
            'token_measurement': 'Unavailable: tiktoken not installed and backend tokenizer not verified; byte counts only.',
            'backend_schema_compatibility': 'Not tested. No endpoint or model invoked.',
            'causal_interpretation': 'No evidence that formatting caused timeout or that this representation would complete faster.',
            'decoder_work': 'Validate one complete 1488-literal vector, then reshape it in fixed lexicographic tensor order into 96 frozen shapes. No fitting or generated code.'}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(Path(__file__).resolve().parents[1])
    result['audit_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.out.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
