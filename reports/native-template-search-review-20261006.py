"""Reproduce a necessary-condition check using shown search demonstrations only."""
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
first = repo / 'runs/weights-native-agent-one-20261006'
request = json.loads((first / 'REQUEST_MANIFEST.json').read_bytes())
packet_bytes = (first / 'AUTHORING_PACKET.json').read_bytes()
assert hashlib.sha256(packet_bytes).hexdigest() == request['packet_sha256']
packet = json.loads(packet_bytes)
source = (first / 'HOST_EVALUATION/runtime/skillcore.py').read_bytes()
assert hashlib.sha256(source).hexdigest() == request['skillcore_sha256']
function = source.decode().split('def assoc_induct(', 1)[1].split('\ndef ', 1)[0]
assert 'for a in np.unique(x):' in function
assert 'out[x == a] = int(np.argmax(logits))' in function

observations = []
for index, demo in enumerate(packet['demonstrations']):
    assert len(demo['input']) == len(demo['output'])
    counts, positions = defaultdict(Counter), defaultdict(dict)
    for row, (inputs, expected) in enumerate(zip(demo['input'], demo['output'])):
        assert len(inputs) == len(expected)
        for column, (color, target) in enumerate(zip(inputs, expected)):
            counts[color][target] += 1
            positions[color].setdefault(target, [row, column])
    conflicts = []
    for color in sorted(counts):
        if len(counts[color]) > 1:
            targets = sorted(counts[color])
            conflicts.append({'input_color': color, 'expected_colors': targets,
                'witness': [{'position': positions[color][target], 'expected': target}
                            for target in targets[:2]]})
    observations.append({'training_example_index': index,
        'shape': [len(demo['input']), len(demo['input'][0])],
        'minimum_pixel_errors_for_any_uniform_per_color_map':
            sum(sum(hist.values()) - max(hist.values()) for hist in counts.values()),
        'same_input_color_requires_different_output_colors': conflicts})

print(json.dumps({'schema': 'cellforge-search-template-review/1', 'task_id': packet['task_id'],
    'partition': 'search_fitness', 'template': 'assoc_induct',
    'shown_packet_sha256': request['packet_sha256'], 'skillcore_sha256': request['skillcore_sha256'],
    'review_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'invariant_from_source': 'For fixed input and context, every occurrence of one input color receives the same output color; dimensions are preserved.',
    'observations': observations, 'query_or_selection_or_report_data_read': False,
    'scope_of_conclusion': 'Exact reproduction of these shown demonstration transformations is impossible for this template, regardless of its weights. This does not prove an outcome on an unseen query or identify the intended spatial rule.',
    'recommendation': 'Stop same-template retries on this task; review a spatially sensitive cell or composition before any further author call.'}, sort_keys=True, indent=2))
