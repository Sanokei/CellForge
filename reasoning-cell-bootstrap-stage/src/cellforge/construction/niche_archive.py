"""Niche-champion archive and Red Queen acceptance for CellForge weight genomes.

Pure CPU, single thread, stdlib only: no gradients, no API calls, no downloads.

What lives here
  * WeightGenome: a content-hashed numeric weight artifact (named tensors) for the
    ``adjacency_recolor/1`` grid_step template (the 37-parameter toy specialist from
    notes/WEIGHT_SEED_INVENTORY_2026-09-30.json) with a pure-Python interpreter that
    mirrors the graph built by scripts/build_grid_adjacency_recolor_gene.py and
    scripts/build_grid_numeric_neighbor_gate.py, and records which cells fired.
  * GridStepGenerator: parametrised synthetic grid-step task generator; one niche per rule.
  * NicheArchive: one champion per niche, lineage, counterexamples, every edit charged
    to a Budget, within-niche competition, cross-niche breeding hooks, and the Red Queen
    acceptance rule (see ``red_queen_decide``). Persisted as hash-verified JSON files.
  * ``scripted_repair_proposer`` / ``random_edit_proposer``: offline stand-ins for the
    maker (the proposer interface is a plain callable, so a real maker plugs in).

Honest scope: the interpreter covers one template. Other skills need their own
``predict`` adapter registered in ``TEMPLATES``; nothing here trains or calls an LLM.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
import random
import time
from dataclasses import dataclass, field

TEMPLATE = 'adjacency_recolor/1'
SHAPES = {
    'adjacency_recolor.source.W': (1, 10), 'adjacency_recolor.source.b': (1,),
    'adjacency_recolor.marker.W': (1, 10), 'adjacency_recolor.marker.b': (1,),
    'adjacency_recolor.target': (10,),
    'neighbor_gate.W': (1, 4), 'neighbor_gate.b': (1,),
}
# Crossover / symbiosis groups: whole cells travel together.
GROUPS = {
    'source': ['adjacency_recolor.source.W', 'adjacency_recolor.source.b'],
    'marker': ['adjacency_recolor.marker.W', 'adjacency_recolor.marker.b'],
    'target': ['adjacency_recolor.target'],
    'gate': ['neighbor_gate.W', 'neighbor_gate.b'],
}
CELL_OF = {p: g for g, ps in GROUPS.items() for p in ps}
# Neighbour order (matches the op:shift list in the builder): up, down, left, right.
NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))
BGS = (0, 1, 5, 6, 7, 8, 9)
MAX_ABS = 100.0


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(obj):
    return hashlib.sha256(canon(obj).encode()).hexdigest()


def _shape(v):
    if not isinstance(v, list):
        raise ValueError('tensor must be a nested list')
    if v and isinstance(v[0], list):
        if not all(isinstance(r, list) and len(r) == len(v[0]) for r in v):
            raise ValueError('ragged tensor')
        return (len(v), len(v[0]))
    return (len(v),)


def _nparams(params):
    n = 0
    for v in params.values():
        s = _shape(v)
        n += s[0] * (s[1] if len(s) > 1 else 1)
    return n


@dataclass
class WeightGenome:
    params: dict
    template: str = TEMPLATE
    parents: list = field(default_factory=list)   # parent genome hashes
    origin: str = 'unknown'                       # seed / edit / crossover / ...
    meta: dict = field(default_factory=dict)      # free-form provenance (not hashed)

    def hash(self):
        return sha({'template': self.template, 'params': self.params})

    @property
    def n_params(self):
        return _nparams(self.params)

    def copy(self):
        return WeightGenome(copy.deepcopy(self.params), self.template, list(self.parents),
                            self.origin, copy.deepcopy(self.meta))

    def validate(self, max_params=64):
        if self.template not in TEMPLATES:
            raise ValueError('unknown template')
        if set(self.params) != set(SHAPES):
            raise ValueError('parameter names differ from template')
        for k, v in self.params.items():
            if _shape(v) != SHAPES[k]:
                raise ValueError(f'bad shape for {k}')
            for x in _flat(v):
                if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or abs(x) > MAX_ABS:
                    raise ValueError(f'bad scalar in {k}')
        if self.n_params > max_params:
            raise ValueError('parameter cap exceeded')

    def to_json(self):
        return {'template': self.template, 'params': self.params, 'parents': self.parents,
                'origin': self.origin, 'meta': self.meta, 'hash': self.hash(), 'n_params': self.n_params}

    @staticmethod
    def from_json(d):
        g = WeightGenome(d['params'], d['template'], d.get('parents', []), d.get('origin', 'unknown'), d.get('meta', {}))
        if g.hash() != d['hash']:
            raise ValueError('genome hash mismatch (artifact corrupted or edited)')
        return g


def _flat(v):
    for x in v:
        if isinstance(x, list):
            yield from x
        else:
            yield x


def _onehot_dot(W, b, color):
    return W[0][color] + b[0]


def toy_predict(genome, grid, trace=False):
    """Run the adjacency_recolor graph on one grid. Returns (prediction, fired).

    fired: {'cells': [...cells with non-zero activation...], 'pixels': {(r,c): {...}}} when trace.
    Outside the grid contributes 0 to neighbour sums (all-zero padding in the real graph).
    """
    p = genome.params
    h, w = len(grid), len(grid[0])
    sW, sb = p['adjacency_recolor.source.W'], p['adjacency_recolor.source.b']
    mW, mb = p['adjacency_recolor.marker.W'], p['adjacency_recolor.marker.b']
    tgt = p['adjacency_recolor.target']
    gW, gb = p['neighbor_gate.W'][0], p['neighbor_gate.b'][0]
    smask = [[_onehot_dot(sW, sb, grid[r][c]) for c in range(w)] for r in range(h)]
    mmask = [[_onehot_dot(mW, mb, grid[r][c]) for c in range(w)] for r in range(h)]
    out = [row[:] for row in grid]
    fired_cells = set()
    pixels = {}
    for r in range(h):
        for c in range(w):
            nb = []
            for dr, dc in NEIGHBORS:
                rr, cc = r + dr, c + dc
                nb.append(mmask[rr][cc] if 0 <= rr < h and 0 <= cc < w else 0.0)
            pre = sum(a * x for a, x in zip(gW, nb)) + gb
            adj = 1.0 if pre > 0 else 0.0
            gate = smask[r][c] * adj
            if smask[r][c] != 0:
                fired_cells.add('source')
            if any(x != 0 for x in nb):
                fired_cells.add('marker')
            if adj:
                fired_cells.add('neighbor_gate')
            if gate != 0:
                fired_cells.add('target')
            vec = [(1.0 if k == grid[r][c] else 0.0) * (1 - gate) + tgt[k] * gate for k in range(10)]
            out[r][c] = max(range(10), key=lambda k: (vec[k], -k))
            if trace:
                pixels[(r, c)] = {'source': smask[r][c], 'neighbors': nb, 'gate_pre': pre, 'gate': gate}
    return out, {'cells': sorted(fired_cells), 'pixels': pixels if trace else None}


TEMPLATES = {TEMPLATE: toy_predict}


def predict(genome, grid, trace=False):
    return TEMPLATES[genome.template](genome, grid, trace)


def toy_genome(source=2, target=3, marker=4, gate_w=(1, 1, 1, 1), gate_b=-1.5, origin='authored', meta=None):
    def oh(c):
        return [1.0 if i == c else 0.0 for i in range(10)]
    return WeightGenome({
        'adjacency_recolor.source.W': [oh(source)], 'adjacency_recolor.source.b': [0.0],
        'adjacency_recolor.marker.W': [oh(marker)], 'adjacency_recolor.marker.b': [0.0],
        'adjacency_recolor.target': oh(target),
        'neighbor_gate.W': [[float(x) for x in gate_w]], 'neighbor_gate.b': [float(gate_b)],
    }, TEMPLATE, [], origin, meta or {})


# ----------------------------------------------------------------------------------------
# Generators (the opponents): synthetic grid-step task families, one niche per gate rule.

@dataclass
class GridStepGenerator:
    niche: str
    gate_w: tuple = (1, 1, 1, 1)
    gate_b: float = -1.5
    source: int = 2
    target: int = 3
    marker: int = 4
    near_miss: float = 0.5     # probability (per grid) of planting a near-miss / positive site
    density: float = 0.25
    generation: int = 0

    def spec(self):
        d = dict(self.__dict__)
        d['gate_w'] = list(self.gate_w)
        return d

    @staticmethod
    def from_spec(d):
        d = dict(d)
        d['gate_w'] = tuple(d['gate_w'])
        return GridStepGenerator(**d)

    def id(self):
        return sha(self.spec())[:12]

    def rule(self, grid):
        h, w = len(grid), len(grid[0])
        out = [row[:] for row in grid]
        for r in range(h):
            for c in range(w):
                if grid[r][c] != self.source:
                    continue
                pre = self.gate_b
                for wt, (dr, dc) in zip(self.gate_w, NEIGHBORS):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and grid[rr][cc] == self.marker:
                        pre += wt
                if pre > 0:
                    out[r][c] = self.target
        return out

    def sample(self, seed, n):
        rng = random.Random(seed)
        pos, neg = [], []
        for mask in range(16):
            pre = self.gate_b + sum(wt for bit, wt in enumerate(self.gate_w) if mask >> bit & 1)
            (pos if pre > 0 else neg).append(mask)
        tasks = []
        for _ in range(n):
            h, w = rng.randint(4, 9), rng.randint(4, 9)
            bg = rng.choice(BGS)
            g = [[bg] * w for _ in range(h)]
            for r in range(h):
                for c in range(w):
                    u = rng.random()
                    if u < self.density:
                        g[r][c] = self.source
                    elif u < self.density + 0.2:
                        g[r][c] = self.marker
            for pool in (pos, neg):  # plant one positive and one near-miss site
                if pool and rng.random() < 0.5 + self.near_miss / 2:
                    r, c = rng.randrange(1, h - 1), rng.randrange(1, w - 1)
                    g[r][c] = self.source
                    m = rng.choice(pool)
                    for bit, (dr, dc) in enumerate(NEIGHBORS):
                        g[r + dr][c + dc] = self.marker if m >> bit & 1 else bg
            tasks.append({'input': g, 'expected': self.rule(g)})
        return tasks

    def mutate(self, rng):
        """Harder variant of the same niche rule (more near-miss sites, denser clutter)."""
        d = self.spec()
        d['near_miss'] = min(1.0, round(self.near_miss + 0.1, 3))
        d['density'] = min(0.45, round(self.density + rng.choice((0.0, 0.05)), 3))
        d['generation'] = self.generation + 1
        return GridStepGenerator.from_spec(d)


def default_generators():
    return [
        GridStepGenerator('any_adjacent', (1, 1, 1, 1), -0.5),
        GridStepGenerator('two_horizontal', (0, 0, 1, 1), -1.5),
        GridStepGenerator('two_vertical', (1, 1, 0, 0), -1.5),
        GridStepGenerator('two_of_four', (1, 1, 1, 1), -1.5),
    ]


# ----------------------------------------------------------------------------------------
# Budget, seeds, evaluation

class BudgetExhausted(Exception):
    pass


class Budget:
    """Every proposed edit is charged, whether valid, neutral, rejected or accepted."""

    def __init__(self, total):
        self.total, self.used = int(total), 0

    @property
    def left(self):
        return self.total - self.used

    def charge(self, n=1):
        if self.used + n > self.total:
            raise BudgetExhausted(f'budget {self.total} exhausted')
        self.used += n


class SeedAllocator:
    """Disjoint seed streams: 'shown' (given to the proposer) never equals 'fresh' (acceptance)."""

    def __init__(self, run_seed):
        self.run_seed, self.counters, self.issued = run_seed, {}, {}

    def next(self, purpose):
        i = self.counters.get(purpose, 0)
        self.counters[purpose] = i + 1
        s = int(sha([self.run_seed, purpose, i])[:12], 16)
        assert all(s not in v for k, v in self.issued.items() if k != purpose), 'seed collision across purposes'
        self.issued.setdefault(purpose, set()).add(s)
        return s


@dataclass
class EvalResult:
    n: int
    solved: int
    fp_grids: int          # grids where an expected-unchanged pixel was changed
    outputs_sha: str
    failures: list         # indices of failed tasks

    @property
    def fp_rate(self):
        return self.fp_grids / self.n if self.n else 0.0


def evaluate(genome, tasks):
    solved = fp = 0
    failures, preds = [], []
    for i, t in enumerate(tasks):
        out, _ = predict(genome, t['input'])
        preds.append(out)
        if out == t['expected']:
            solved += 1
        else:
            failures.append(i)
        if any(out[r][c] != t['input'][r][c] and t['expected'][r][c] == t['input'][r][c]
               for r in range(len(out)) for c in range(len(out[0]))):
            fp += 1
    return EvalResult(len(tasks), solved, fp, sha(preds), failures)


def counterexamples(genome, tasks, limit=5):
    """Failing instances with the predicted grid and which cells fired on the wrong pixels."""
    rows = []
    for t in tasks:
        out, fired = predict(genome, t['input'], trace=True)
        if out == t['expected']:
            continue
        wrong = [(r, c) for r in range(len(out)) for c in range(len(out[0])) if out[r][c] != t['expected'][r][c]]
        rows.append({'input': t['input'], 'expected': t['expected'], 'predicted': out,
                     'cells_fired': fired['cells'],
                     'wrong_pixels': [{'rc': list(rc), **fired['pixels'][rc],
                                       'kind': 'false_positive' if t['expected'][rc[0]][rc[1]] == t['input'][rc[0]][rc[1]] else 'miss'}
                                      for rc in wrong]})
        if len(rows) >= limit:
            break
    return rows


# ----------------------------------------------------------------------------------------
# Red Queen acceptance rule (pure function; unit-testable)

@dataclass
class Criteria:
    min_gain: int = 1            # extra solved fresh samples vs parent (paired, same samples)
    archive_tol: int = 0         # allowed solved-count drop on any earlier generator
    anchor_tol: float = 0.0      # allowed drop in real-anchor score (fraction)
    fp_tol: float = 0.0          # allowed rise in grid-level false-positive rate
    max_fp_rate: float = 1.0
    max_params: int = 64
    max_eval_grids: int = 100000


def red_queen_decide(parent, child, crit, *, novel, child_params, eval_grids):
    """parent/child: dict(fresh=EvalResult, archive=[EvalResult...], anchor=float)."""
    why = {}
    gain = child['fresh'].solved - parent['fresh'].solved
    why['fresh_gain'] = (gain >= crit.min_gain, f'gain {gain} (need >= {crit.min_gain})')
    drops = [p.solved - c.solved for p, c in zip(parent['archive'], child['archive'])]
    worst = max(drops, default=0)
    why['archive_no_regression'] = (worst <= crit.archive_tol, f'worst drop {worst} over {len(drops)} earlier generators')
    da = parent['anchor'] - child['anchor']
    why['anchor_no_drop'] = (da <= crit.anchor_tol + 1e-12, f'anchor {parent["anchor"]:.3f}->{child["anchor"]:.3f}')
    why['novel_or_strictly_better'] = (bool(novel) or gain > 0, 'novel signature' if novel else 'duplicate behaviour')
    pfp = _pooled_fp(parent)
    cfp = _pooled_fp(child)
    why['false_positive'] = (cfp <= pfp + crit.fp_tol + 1e-12 and cfp <= crit.max_fp_rate, f'fp {pfp:.3f}->{cfp:.3f}')
    why['caps'] = (child_params <= crit.max_params and eval_grids <= crit.max_eval_grids,
                   f'params {child_params}, eval grids {eval_grids}')
    return all(v[0] for v in why.values()), why


def _pooled_fp(side):
    rs = [side['fresh']] + list(side['archive'])
    n = sum(r.n for r in rs)
    return sum(r.fp_grids for r in rs) / n if n else 0.0


# ----------------------------------------------------------------------------------------
# Proposals

@dataclass
class Proposal:
    actor: str = 'unknown'
    model: str = 'offline'
    rationale: str = ''
    edits: list = field(default_factory=list)   # [{'param','index':[i(,j)],'value'}]
    genome: WeightGenome | None = None          # whole-genome challenger / breeding child
    kind: str = 'edits'                         # edits | challenger | crossover
    parents: list = field(default_factory=list)

    def digest(self):
        body = {'actor': self.actor, 'model': self.model, 'edits': self.edits,
                'genome': self.genome.hash() if self.genome else None, 'kind': self.kind}
        try:
            return sha(body)
        except ValueError:   # NaN/inf in a proposal: still hashable, then rejected as invalid
            return hashlib.sha256(repr(body).encode()).hexdigest()


def apply_edits(parent, edits):
    child = parent.copy()
    for e in edits:
        name, idx, val = e['param'], e['index'], e['value']
        if name not in child.params:
            raise ValueError(f'unknown parameter {name}')
        t = child.params[name]
        if len(idx) != len(_shape(t)):
            raise ValueError('index rank mismatch')
        for i, n in zip(idx, _shape(t)):
            if not isinstance(i, int) or not 0 <= i < n:
                raise ValueError('index out of range')
        if len(idx) == 2:
            t[idx[0]][idx[1]] = float(val)
        else:
            t[idx[0]] = float(val)
    child.parents, child.origin = [parent.hash()], 'edit'
    return child


def crossover_cells(a, b, rng):
    """Cell-group crossover: each cell (source/marker/target/gate) comes from a or b; both contribute."""
    groups = sorted(GROUPS)
    while True:
        pick = {g: rng.choice('ab') for g in groups}
        if len(set(pick.values())) == 2:
            break
    params = {}
    for g, ps in GROUPS.items():
        src = a if pick[g] == 'a' else b
        for p in ps:
            params[p] = copy.deepcopy(src.params[p])
    return WeightGenome(params, a.template, sorted([a.hash(), b.hash()]), 'crossover',
                        {'cells_from_a': [g for g in groups if pick[g] == 'a'],
                         'cells_from_b': [g for g in groups if pick[g] == 'b']})


def symbiotic_merge(a, b, rng):
    """Hook placeholder: a real symbiogenic merge needs two champions failing on disjoint sets.
    For this template there is no extra cell to add, so it degrades to crossover."""
    return crossover_cells(a, b, rng)


# ----------------------------------------------------------------------------------------
# The archive

@dataclass
class Outcome:
    niche: str
    status: str            # accepted | rejected | invalid | neutral | budget
    detail: dict
    child_hash: str | None = None


class NicheArchive:
    def __init__(self, root=None, *, budget=64, run_seed=0, criteria=None, n_fresh=24, n_archive=12,
                 n_anchor=24, anchors=None, breeders=None):
        self.root = root
        self.budget = Budget(budget)
        self.seeds = SeedAllocator(run_seed)
        self.crit = criteria or Criteria()
        self.n_fresh, self.n_archive, self.n_anchor = n_fresh, n_archive, n_anchor
        self.champions = {}     # niche -> WeightGenome
        self.history = {}       # niche -> [hashes of every champion ever, oldest first]
        self.retired_sigs = {}  # niche -> set(signature)
        self.generators = {}    # niche -> [GridStepGenerator] (every generation; never deleted)
        self.anchors = dict(anchors or {})  # niche -> list of frozen real-like tasks
        self.ledger = []
        self.counterexample_log = []
        self.breeders = {'crossover': crossover_cells, 'symbiotic_merge': symbiotic_merge}
        self.breeders.update(breeders or {})
        self.genomes = {}       # hash -> WeightGenome (content addressed)
        self._probe = None

    # -- generators / anchors ---------------------------------------------------------
    def add_generator(self, gen, *, consider=True, lo=0.0, hi=0.95):
        """Difficulty control: keep a new generator only if some archived solver solves a fraction
        >= lo (learnable) and the niche champion does not already solve >= hi (not trivial).
        The first generator of a niche is always kept. Kept generators are never deleted."""
        lst = self.generators.setdefault(gen.niche, [])
        if lst and consider and self.champions:
            tasks = gen.sample(self.seeds.next('difficulty'), self.n_fresh)
            fr = {n: evaluate(g, tasks).solved / len(tasks) for n, g in self.champions.items()}
            champ = fr.get(gen.niche, 0.0)
            if not (max(fr.values()) >= lo and champ < hi):
                self._log('generator_rejected', niche=gen.niche, gen=gen.id(), champion=champ, best=max(fr.values()))
                return False
        lst.append(gen)
        self._log('generator_added', niche=gen.niche, gen=gen.id(), generation=gen.generation)
        return True

    def freeze_anchor(self, niche, tasks):
        self.anchors[niche] = tasks
        self._log('anchor_frozen', niche=niche, n=len(tasks), sha=sha(tasks)[:16])

    def _fresh_tasks(self, niche, purpose):
        gen = self.generators[niche][-1]
        seed = self.seeds.next(purpose)
        n = self.n_fresh if purpose != 'shown' else max(self.n_fresh, 12)
        return gen.sample(seed, n), seed

    def _archive_tasks(self, niche):
        """Earlier generators of this niche (all but the newest) -> regression set."""
        out = []
        for g in self.generators[niche][:-1]:
            out.append(g.sample(int(sha(['archive', g.id()])[:12], 16), self.n_archive))
        return out

    def probe(self):
        if self._probe is None:
            self._probe = [t for gens in self.generators.values() for g in gens[:1]
                           for t in g.sample(0xC0FFEE, 6)]
        return self._probe

    def signature(self, genome):
        return evaluate(genome, self.probe()).outputs_sha

    # -- registration -------------------------------------------------------------------
    def _log(self, event, **kw):
        row = {'t': len(self.ledger), 'event': event, 'budget_used': self.budget.used, **kw}
        self.ledger.append(row)
        return row

    def _set_champion(self, niche, genome, reason):
        old = self.champions.get(niche)
        if old is not None:
            self.retired_sigs.setdefault(niche, set()).add(self.signature(old))
        self.champions[niche] = genome
        self.genomes[genome.hash()] = genome
        self.history.setdefault(niche, []).append(genome.hash())
        self._log('champion_set', niche=niche, genome=genome.hash(), reason=reason, parents=genome.parents)

    def seed(self, genomes):
        """Place each seed genome into the niche(s) where it beats the incumbent (paired fresh
        samples). Every placement attempt is an evaluation charged as one proposal."""
        for g in genomes:
            g.validate(self.crit.max_params)
            self.genomes[g.hash()] = g
            for niche in sorted(self.generators):
                if self.budget.left < 1:
                    return
                self.budget.charge(1)
                tasks, seed = self._fresh_tasks(niche, 'seed')
                cand = evaluate(g, tasks)
                inc = self.champions.get(niche)
                if inc is None or cand.solved > evaluate(inc, tasks).solved:
                    self._set_champion(niche, g, f'seed fresh {cand.solved}/{cand.n}')
                else:
                    self._log('seed_rejected', niche=niche, genome=g.hash(), solved=cand.solved, seed=seed)

    # -- counterexamples ---------------------------------------------------------------
    def collect_counterexamples(self, niche, limit=5):
        """Shown-seed samples (never used for acceptance). Logged with niche and champion hash."""
        tasks, seed = self._fresh_tasks(niche, 'shown')
        champ = self.champions[niche]
        rows = counterexamples(champ, tasks, limit)
        for r in rows:
            rec = {'niche': niche, 'champion': champ.hash(), 'seed': seed, **r}
            self.counterexample_log.append(rec)
        self._log('counterexamples', niche=niche, champion=champ.hash(), seed=seed, n=len(rows))
        return rows

    # -- submit: the charged, Red Queen-gated path -------------------------------------
    def submit(self, niche, proposal):
        try:
            self.budget.charge(1)
        except BudgetExhausted:
            return Outcome(niche, 'budget', {})
        parent = self.champions[niche]
        rec = {'niche': niche, 'actor': proposal.actor, 'model': proposal.model, 'kind': proposal.kind,
               'proposal': proposal.digest(), 'rationale': proposal.rationale, 'parent': parent.hash()}
        try:
            child = proposal.genome.copy() if proposal.genome is not None else apply_edits(parent, proposal.edits)
            if proposal.genome is not None and not child.parents:
                child.parents = list(proposal.parents) or [parent.hash()]
            child.validate(self.crit.max_params)
        except (ValueError, KeyError, TypeError, IndexError) as e:
            self._log('proposal', status='invalid', error=str(e), **rec)
            return Outcome(niche, 'invalid', {'error': str(e)})
        if child.hash() == parent.hash():
            self._log('proposal', status='neutral', **rec)
            return Outcome(niche, 'neutral', {}, child.hash())
        fresh_tasks, fseed = self._fresh_tasks(niche, 'fresh')
        arch_tasks = self._archive_tasks(niche)
        anchor_tasks = self.anchors.get(niche) or []
        P = {'fresh': evaluate(parent, fresh_tasks), 'archive': [evaluate(parent, t) for t in arch_tasks],
             'anchor': _anchor(parent, anchor_tasks)}
        C = {'fresh': evaluate(child, fresh_tasks), 'archive': [evaluate(child, t) for t in arch_tasks],
             'anchor': _anchor(child, anchor_tasks)}
        grids = 2 * (len(fresh_tasks) + sum(len(t) for t in arch_tasks) + len(anchor_tasks))
        sig = self.signature(child)
        known = set(self.retired_sigs.get(niche, ())) | {self.signature(parent)}
        ok, why = red_queen_decide(P, C, self.crit, novel=sig not in known, child_params=child.n_params, eval_grids=grids)
        rec.update(child=child.hash(), fresh_seed=fseed, fresh=[P['fresh'].solved, C['fresh'].solved, C['fresh'].n],
                   anchor=[P['anchor'], C['anchor']], criteria={k: [v[0], v[1]] for k, v in why.items()})
        if ok:
            self._set_champion(niche, child, f'accepted proposal {proposal.digest()[:12]}')
            self._log('proposal', status='accepted', **rec)
            return Outcome(niche, 'accepted', rec, child.hash())
        self._log('proposal', status='rejected', **rec)
        return Outcome(niche, 'rejected', rec, child.hash())

    # -- within-niche competition --------------------------------------------------------
    def challenge(self, niche, challenger, actor='challenger'):
        """Champion vs challenger on fresh generator samples, through the full acceptance rule."""
        return self.submit(niche, Proposal(actor=actor, kind='challenger', genome=challenger,
                                           parents=list(challenger.parents)))

    def compete(self, niche, challenger):
        """Uncharged diagnostic duel (no state change): paired fresh solved counts."""
        tasks, seed = self._fresh_tasks(niche, 'duel')
        a, b = evaluate(self.champions[niche], tasks), evaluate(challenger, tasks)
        return {'champion': a.solved, 'challenger': b.solved, 'n': a.n, 'seed': seed,
                'winner': 'challenger' if b.solved > a.solved else 'champion'}

    # -- cross-niche breeding ------------------------------------------------------------
    def breed_cross_niche(self, niche_a, niche_b, rng, breeder='crossover'):
        """Breed the two champions; offer the child to both niches (two charged proposals)."""
        a, b = self.champions[niche_a], self.champions[niche_b]
        child = self.breeders[breeder](a, b, rng)
        if child is None:
            return []
        out = []
        for n in (niche_a, niche_b):
            out.append(self.submit(n, Proposal(actor=f'breeder:{breeder}', kind='crossover', genome=child,
                                               rationale=f'{niche_a} x {niche_b}', parents=[a.hash(), b.hash()])))
        return out

    # -- rounds ---------------------------------------------------------------------------
    def run_round(self, proposer, rng, *, proposals_per_niche=2, breed=True, grow_generators=True):
        """One Red Queen round: grow opponents, show counterexamples, charged repairs, breeding."""
        summary = {'accepted': 0, 'rejected': 0, 'invalid': 0, 'neutral': 0, 'budget': 0}
        if grow_generators:
            for niche in sorted(self.generators):
                self.add_generator(self.generators[niche][-1].mutate(rng))
        for niche in sorted(self.champions):
            shown = self.collect_counterexamples(niche)
            if not shown:
                continue
            for _ in range(proposals_per_niche):
                prop = proposer(self.champions[niche], shown, rng)
                if prop is None:
                    break
                o = self.submit(niche, prop)
                summary[o.status] += 1
                if o.status == 'accepted':
                    shown = self.collect_counterexamples(niche) or shown
        if breed:
            for a, b in itertools.combinations(sorted(self.champions), 2):
                for o in self.breed_cross_niche(a, b, rng):
                    summary[o.status] += 1
        self._log('round', **summary)
        return summary

    def stats(self):
        props = [r for r in self.ledger if r['event'] == 'proposal']
        acc = [r for r in props if r['status'] == 'accepted']
        return {'charged': self.budget.used, 'budget': self.budget.total, 'proposals': len(props),
                'accepted': len(acc), 'accepted_per_100': 100.0 * len(acc) / max(1, len(props)),
                'niches': sorted(self.champions), 'counterexamples': len(self.counterexample_log),
                'champions': {n: g.hash()[:12] for n, g in self.champions.items()},
                'opponents': sum(len(v) for v in self.generators.values())}

    def champion_scores(self, n=200, seed=987654321):
        """Report-only: champion solved fraction on a large untouched sample of each niche's first generator."""
        out = {}
        for niche, champ in self.champions.items():
            tasks = self.generators[niche][0].sample(seed, n)
            out[niche] = evaluate(champ, tasks).solved / n
        return out

    # -- persistence ----------------------------------------------------------------------
    def save(self, root=None):
        root = root or self.root
        os.makedirs(os.path.join(root, 'genomes'), exist_ok=True)
        for h, g in self.genomes.items():
            with open(os.path.join(root, 'genomes', h + '.json'), 'w') as f:
                json.dump(g.to_json(), f, sort_keys=True)
        state = {'champions': {n: g.hash() for n, g in self.champions.items()}, 'history': self.history,
                 'generators': {n: [g.spec() for g in v] for n, v in self.generators.items()},
                 'budget': [self.budget.total, self.budget.used],
                 'seed_counters': self.seeds.counters, 'run_seed': self.seeds.run_seed,
                 'anchor_shas': {n: sha(t) for n, t in self.anchors.items()},
                 'retired_sigs': {n: sorted(v) for n, v in self.retired_sigs.items()}}
        state['state_sha256'] = sha({k: v for k, v in state.items()})
        with open(os.path.join(root, 'archive.json'), 'w') as f:
            json.dump(state, f, sort_keys=True, indent=1)
        with open(os.path.join(root, 'ledger.jsonl'), 'w') as f:
            for r in self.ledger:
                f.write(canon(r) + '\n')
        with open(os.path.join(root, 'counterexamples.jsonl'), 'w') as f:
            for r in self.counterexample_log:
                f.write(canon(r) + '\n')

    @classmethod
    def load(cls, root, anchors=None, **kw):
        with open(os.path.join(root, 'archive.json')) as f:
            state = json.load(f)
        claimed = state.pop('state_sha256')
        if sha(state) != claimed:
            raise ValueError('archive.json hash mismatch')
        a = cls(root, budget=state['budget'][0], run_seed=state['run_seed'], anchors=anchors, **kw)
        a.budget.used = state['budget'][1]
        a.seeds.counters = {k: v for k, v in state['seed_counters'].items()}
        for k, n in a.seeds.counters.items():       # rebuild issued seeds so disjointness still holds
            for i in range(n):
                a.seeds.issued.setdefault(k, set()).add(int(sha([a.seeds.run_seed, k, i])[:12], 16))
        for h in os.listdir(os.path.join(root, 'genomes')):
            with open(os.path.join(root, 'genomes', h)) as f:
                g = WeightGenome.from_json(json.load(f))
            a.genomes[g.hash()] = g
        for n, h in state['champions'].items():
            a.champions[n] = a.genomes[h]
        a.history = state['history']
        a.generators = {n: [GridStepGenerator.from_spec(s) for s in v] for n, v in state['generators'].items()}
        a.retired_sigs = {n: set(v) for n, v in state['retired_sigs'].items()}
        for n, t in (anchors or {}).items():
            if state['anchor_shas'].get(n) not in (None, sha(t)):
                raise ValueError(f'anchor for {n} differs from the frozen one')
        with open(os.path.join(root, 'ledger.jsonl')) as f:
            a.ledger = [json.loads(line) for line in f if line.strip()]
        with open(os.path.join(root, 'counterexamples.jsonl')) as f:
            a.counterexample_log = [json.loads(line) for line in f if line.strip()]
        return a


def _anchor(genome, tasks):
    return evaluate(genome, tasks).solved / len(tasks) if tasks else 0.0


# ----------------------------------------------------------------------------------------
# Seeding from the inventory and offline proposers

def load_inventory_genomes(path):
    with open(path) as f:
        inv = json.load(f)
    out = []
    for c in inv['candidates']:
        g = WeightGenome({k: v for k, v in c['parameters'].items()}, TEMPLATE, [], 'seed',
                         {'source_path': c['path'], 'source_hash': c['hash'], 'provenance': c['provenance'],
                          'inventory_status': c['status']})
        g.validate()
        out.append(g)
    return out


def builtin_seed_genomes():
    """The four toy candidates, reconstructed from their recorded gate values (used when the
    inventory file is unavailable). Same tensors as the inventory entries."""
    specs = [((1, 1, 1, 1), -1.5), ((0, 0, 1, 1), -1.5), ((2, -1, 1, 0), -2.5), ((1, 2, 2, 0), -1.5)]
    return [toy_genome(gate_w=w, gate_b=b, origin='seed', meta={'provenance': 'builtin copy of inventory'})
            for w, b in specs]


def make_anchor(generator, seed=0xA11C0, n=24):
    """Frozen 'real-like' anchor for a niche: same rule, shifted distribution (bigger, sparser grids).
    A stand-in for real ARC tasks; Codex should pass real dev tasks here."""
    g = GridStepGenerator.from_spec({**generator.spec(), 'density': 0.15, 'near_miss': 0.2})
    return g.sample(seed, n)


def random_edit_proposer(vocab=(-1.0, 0.0, 1.0, 2.0)):
    """Control: one random literal edit of the gate cell or target/source cell (same vocabulary)."""
    def prop(champ, shown, rng):
        if rng.random() < 0.8:
            j = rng.randrange(4)
            return Proposal('random', 'offline', 'random gate weight', [{'param': 'neighbor_gate.W', 'index': [0, j],
                            'value': rng.choice(vocab)}])
        return Proposal('random', 'offline', 'random gate bias', [{'param': 'neighbor_gate.b', 'index': [0],
                        'value': rng.choice((-0.5, -1.5, -2.5, -3.5))}])
    return prop


def scripted_repair_proposer(W_vocab=(0.0, 1.0), b_vocab=(-0.5, -1.5, -2.5, -3.5)):
    """Offline stand-in for the maker: reads the counterexamples (wrong pixels with their
    fired neighbour evidence) and writes the gate weights/bias that explain the most of them.
    One proposal = one charged edit set. Not an LLM; replace by a real maker via the same signature."""
    def prop(champ, shown, rng):
        obs = []
        for ce in shown:
            for px in ce['wrong_pixels']:
                nb = [1 if x != 0 else 0 for x in px['neighbors']]
                obs.append((nb, px['kind'] == 'miss'))   # miss -> gate should be >0
        if not obs:
            return None
        cur = champ.params['neighbor_gate.W'][0]
        curb = champ.params['neighbor_gate.b'][0]
        best, bestfit = None, -1
        for w in itertools.product(W_vocab, repeat=4):
            for b in b_vocab:
                fit = sum((sum(a * x for a, x in zip(w, nb)) + b > 0) == want for nb, want in obs)
                fit = fit * 100 - sum(abs(a - c) for a, c in zip(w, cur)) - abs(b - curb)  # prefer small edits
                if fit > bestfit:
                    best, bestfit = (w, b), fit
        w, b = best
        edits = [{'param': 'neighbor_gate.W', 'index': [0, j], 'value': float(w[j])} for j in range(4)
                 if w[j] != cur[j]]
        if b != curb:
            edits.append({'param': 'neighbor_gate.b', 'index': [0], 'value': float(b)})
        return Proposal('scripted', 'offline-heuristic', f'fit {len(obs)} wrong pixels', edits)
    return prop


# ----------------------------------------------------------------------------------------
# End-to-end smoke round

def smoke(out_dir=None, *, inventory=None, budget=96, rounds=3, seed=7, proposer=None, run_seed=1):
    arch = NicheArchive(out_dir, budget=budget, run_seed=run_seed)
    for g in default_generators():
        arch.add_generator(g)
        arch.freeze_anchor(g.niche, make_anchor(g))
    seeds = load_inventory_genomes(inventory) if inventory and os.path.exists(inventory) else builtin_seed_genomes()
    arch.seed(seeds)
    before = arch.champion_scores()
    rng = random.Random(seed)
    proposer = proposer or scripted_repair_proposer()
    t0 = time.time()
    rounds_out = []
    for _ in range(rounds):
        if arch.budget.left < 1:
            break
        rounds_out.append(arch.run_round(proposer, rng))
    after = arch.champion_scores()
    if out_dir:
        arch.save()
    return {'before': before, 'after': after, 'rounds': rounds_out, 'stats': arch.stats(),
            'seconds': round(time.time() - t0, 2), 'archive': arch}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='Red Queen niche archive smoke/run (CPU only, no API)')
    ap.add_argument('--out', help='directory to persist the archive (must not exist)')
    ap.add_argument('--inventory', default='notes/WEIGHT_SEED_INVENTORY_2026-09-30.json')
    ap.add_argument('--budget', type=int, default=96)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--proposer', choices=['scripted', 'random'], default='scripted')
    a = ap.parse_args(argv)
    if a.out and os.path.exists(a.out):
        ap.error('--out exists; archives are never overwritten')
    prop = scripted_repair_proposer() if a.proposer == 'scripted' else random_edit_proposer()
    res = smoke(a.out, inventory=a.inventory, budget=a.budget, rounds=a.rounds, seed=a.seed, proposer=prop)
    res.pop('archive')
    print(json.dumps(res, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
