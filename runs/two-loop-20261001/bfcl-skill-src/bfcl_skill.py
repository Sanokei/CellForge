"""Additive structured-text tool cell; schemas from the owner's local BFCL source.

No calls are executed. Predictions are typed call objects. No BFCL answer is an
input feature. This is a new template, never a replacement for tool_call.
"""
import json
import random
import re

import numpy as np
import skillcore as S

TYPES = ('integer', 'float', 'string', 'boolean')
TEMPLATE = 'bfcl_call2'
NICHENAME = 'bfcl-call2'


def words(text):
    return set(re.findall(r'[a-z]+|\d+', text.lower().replace('_', ' ')))


def candidates(query):
    # Syntax only: labeled JSON values, then unlabelled scalar mentions.
    found = []
    decoder = json.JSONDecoder()
    for m in re.finditer(r'([A-Za-z][\w]*)\s*=\s*', query):
        try:
            value, _ = decoder.raw_decode(query[m.end():])
            if isinstance(value, (str, int, float, bool)):
                found.append((m.group(1), value, m.start()))
        except ValueError:
            pass
    if found:
        return found
    pattern = r'"([^"\n]*)"|\b(true|false)\b|(?<![\w.])-?\d+(?:\.\d+)?|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    for m in re.finditer(pattern, query):
        raw = m.group(0)
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        found.append(('', value, m.start()))
    return found


def value_type(v):
    if isinstance(v, bool):
        return 3
    if isinstance(v, int):
        return 0
    if isinstance(v, float):
        return 1
    return 2


def bfcl_call2(params, request, demos=None):
    query, tools = request['query'], request['tools']
    qwords = words(query)
    r = np.asarray(params['route'], float)
    scores = []
    for tool in tools:
        props = tool['parameters']['properties']
        desc, name = words(tool['description']), words(tool['name'])
        keys = set().union(*(words(k) for k in props)) if props else set()
        scores.append(float(r @ [len(qwords & desc), len(qwords & name), len(qwords & keys), 1]))
    tool = tools[int(np.argmax(scores))]
    args, vals = {}, candidates(query)
    if not vals:
        raise ValueError('no scalar candidates')
    bind = np.asarray(params['bind'], float)
    types = np.asarray(params['type'], float)
    used = set()
    for ordinal, key in enumerate(tool['parameters']['required']):
        prop = tool['parameters']['properties'][key]
        target = TYPES.index(prop['type'])
        ranked = []
        for i, (label, value, offset) in enumerate(vals):
            feature = [float(label == key), len(words(label) & words(key)),
                       float(i == ordinal), len(words(query[max(0, offset - 35):offset]) & words(prop.get('description', '')))]
            ranked.append(float(bind @ feature + types[target, value_type(value)]))
        for i in used:
            ranked[i] = -float('inf')
        pick = int(np.argmax(ranked))
        if not np.isfinite(ranked[pick]):
            raise ValueError('not enough values')
        used.add(pick)
        value = vals[pick][1]
        # Output conversion is itself selected by the tensor row.
        conversion = int(np.argmax(types[target]))
        if conversion == 0:
            value = int(value)
        elif conversion == 1:
            value = float(value)
        elif conversion == 2:
            value = str(value)
        elif isinstance(value, bool):
            value = value
        elif str(value).lower() in ('true', 'false'):
            value = str(value).lower() == 'true'
        else:
            raise ValueError('invalid boolean')
        args[key] = value
    return {'name': tool['name'], 'arguments': args}


S.TEMPLATES['bfcl_call2'] = {
    'shapes': {'route': (4,), 'bind': (4,), 'type': (4, 4)}, 'fn': bfcl_call2, 'aware': False,
    'doc': 'Input is {query: text, tools: BFCL schemas}. Output {name, arguments}, required scalar arguments only. '
           'route dot features [query/description token overlap, query/name overlap, query/argument-key overlap, constant]. '
           'bind dot features [exact label match, label/key token overlap, candidate index equals argument ordinal, '
           'argument-description overlap in preceding 35 query chars], plus type[row required type,column candidate type]. '
           'Types ordered integer,float,string,boolean. Select tool and unused candidates by argmax. '
           'Convert output using argmax of the required type row. Labeled candidates use key=JSON scalar syntax; '
           'otherwise candidates are quoted strings, booleans, numbers and capitalized phrases. '
           'Write all 24 literal numbers. Do not return code.'}


def schema_rows(source):
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    return [r for r in rows if len(r['function']) >= 2 and all(
        t['parameters'].get('required') and all(t['parameters']['properties'][k].get('type') in TYPES
        for k in t['parameters']['required']) for t in r['function'])]


def generated(row, seed, tier=0):
    rng = random.Random(seed)
    tools = json.loads(json.dumps(row['function']))
    rng.shuffle(tools)
    tool = rng.choice(tools)
    args = {}
    for key in tool['parameters']['required']:
        typ = tool['parameters']['properties'][key]['type']
        args[key] = {'integer': rng.randint(-50, 90), 'float': round(rng.uniform(-20, 30), 3),
                     'string': rng.choice(['Boston', 'Maple', 'Cobalt', 'River', 'Berlin']), 'boolean': bool(rng.randrange(2))}[typ]
    fields = list(args.items()) + [('unrelated_value', rng.randint(101, 999))]
    rng.shuffle(fields)
    if tier:
        fields = list(args.items())
        rng.shuffle(fields)
        query = tool['description'].lower() + ' ' + '; '.join(
            tool['parameters']['properties'][k].get('description', k).lower() + ': ' + json.dumps(v) for k, v in fields)
    else:
        query = tool['description'] + ' ' + '; '.join(k + '=' + json.dumps(v) for k, v in fields)
    return {'input': {'query': query, 'tools': tools}, 'output': {'name': tool['name'], 'arguments': args}}


def strict_equal(pred, expected):
    # Python otherwise equates true with 1 and 1 with 1.0.
    return json.dumps(pred, sort_keys=True) == json.dumps(expected, sort_keys=True)


def score(cell, pairs):
    outcomes = []
    for pair in pairs:
        try:
            outcomes.append(strict_equal(S.run(cell, pair['input'], []), pair['output']))
        except Exception:
            outcomes.append(False)
    return sum(outcomes), outcomes
