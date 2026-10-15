"""New lookahead structure for the formerly greedy resource-planning niche."""
from collections import deque
import json
import random

import numpy as np
import skillcore as S

TEMPLATE = 'resource_lookahead2'
NICHENAME = 'resource-lookahead2'
SHAPES = {'reward': (6,), 'action': (4,), 'horizon': (16,), 'beam': (6,)}


def transition(state, action, q):
    m, g, d, e = state
    if action == 0:
        m, e = m + 1, e - q['costs'][0]
    elif action == 1:
        m, g, e = m - 1, g + 1, e - q['costs'][1]
    elif action == 2:
        g, d, e = g - 1, d + 1, e - q['costs'][2]
    else:
        if e == q['max_energy']:
            return None
        e = q['max_energy']
    if min(m, g, e) < 0 or m + g > q['capacity'] or d > q['goal']:
        return None
    return (m, g, d, e)


def reference(q):
    queue = deque([(tuple(q['initial']), [])])
    seen = {tuple(q['initial'])}
    while queue:
        state, plan = queue.popleft()
        if state[2] == q['goal']:
            return plan
        if len(plan) >= q['max_steps']:
            continue
        for action in range(4):
            nxt = transition(state, action, q)
            if nxt is not None and nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, plan + [action]))
    return None


def execute(params, q):
    horizon = min(int(np.argmax(params['horizon'])) + 1, q['max_steps'])
    beam = (int(np.argmax(params['beam'])) + 1) * 8
    reward, action_weights = np.asarray(params['reward']), np.asarray(params['action'])

    def utility(state, plan):
        m, g, d, e = state
        features = [d / q['goal'], float(d == q['goal']), -len(plan), e / q['max_energy'],
                    m / q['capacity'], g / q['capacity']]
        return float(reward @ features + sum(action_weights[a] for a in plan))

    initial = tuple(q['initial'])
    frontier = [(initial, [])]
    completed = []
    visited = {initial: 0}
    for depth in range(horizon + 1):
        expanded = {}
        for state, plan in frontier:
            if state[2] == q['goal']:
                completed.append((state, plan))
                continue
            if depth == horizon:
                continue
            for action in range(4):
                nxt = transition(state, action, q)
                if nxt is None or visited.get(nxt, depth + 1) < depth + 1:
                    continue
                candidate = (nxt, plan + [action])
                if nxt not in expanded or utility(*candidate) > utility(*expanded[nxt]):
                    expanded[nxt] = candidate
        frontier = sorted(expanded.values(), key=lambda item: utility(*item), reverse=True)[:beam]
        for state, plan in frontier:
            visited[state] = len(plan)
        if not frontier:
            break
    options = completed + frontier
    if not options:
        return {'complete': False, 'steps': 0}, []
    state, plan = max(options, key=lambda item: utility(*item))
    return {'complete': state[2] == q['goal'], 'steps': len(plan)}, plan


def predict(params, q, demos=None):
    return execute(params, q)[0]


S.TEMPLATES[TEMPLATE] = {'shapes': SHAPES, 'fn': predict, 'aware': False,
    'doc': 'Resource planning with weight-controlled finite-horizon beam search. Input initial=[materials,goods,delivered,energy], '
           'goal,capacity,max_energy,costs=[acquire,assemble,deliver],max_steps. Actions 0 acquire material, '
           '1 assemble material into goods, 2 deliver goods, 3 recharge battery. Costs consume energy; legal '
           'transitions enforce inventory, capacity and energy. Output {complete:bool,steps:int}; need shortest legal completion. '
           'reward dot features [delivered/goal,completion_indicator,-steps,energy/max_energy,materials/capacity,goods/capacity] '
           'plus sum of action[a] over path. horizon argmax+1 (1..16); beam (argmax+1)*8 (8..48). '
           'At each breadth depth retain best utility state paths within beam; revisiting a state at greater depth is disallowed. '
           'Final argmax chooses among completed and frontier states. Every behavioral search priority and bound is a literal tensor. '
           'Write reward[6],action[4],horizon[16],beam[6], finite numbers [-9,9].'}


def schema_rows(source):
    # Separate task populations declared before authoring; no frozen old tasks read.
    return [{'id': 'planning-%d' % i, 'index': i} for i in range(120)]


def generated(row, seed, tier=1):
    rng = random.Random(seed * 101 + row['index'])
    for _ in range(1000):
        goal, capacity = rng.randint(1, 3), rng.randint(1, 3)
        max_energy = rng.randint(3, 9)
        m = rng.randint(0, min(capacity, 1))
        g = rng.randint(0, capacity - m)
        q = {'initial': [m, g, 0, rng.randint(0, max_energy)], 'goal': goal, 'capacity': capacity,
             'max_energy': max_energy, 'costs': [rng.randint(1, min(4, max_energy)) for _ in range(3)], 'max_steps': 16}
        plan = reference(q)
        if plan is not None and len(plan) >= 3:
            return {'input': q, 'output': {'complete': True, 'steps': len(plan)}, 'reference_plan_for_generated_example': plan}
    raise RuntimeError('no reachable generated planning task')


def score(cell, pairs):
    outcomes = []
    for pair in pairs:
        try:
            outcomes.append(S.run(cell, pair['input'], []) == pair['output'])
        except Exception:
            outcomes.append(False)
    return sum(outcomes), outcomes
