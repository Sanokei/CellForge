"""Continuous head-center driver: niche stages (maker writes literal-weight cells) -> breeding rounds (chains of earned cells) -> ARC scoring.

State lives in ROOT (REGISTRY.json, STATE.json, HISTORY.jsonl, LEADERBOARD.json, STATUS.json, AUTHOR-*/). Safe to stop (STOP file) and resume.
Maker: server `codex exec -m gpt-6.1-sol -c model_reasoning_effort=low` (owner-configured ChatGPT-subscription route; read-only; no API key).
Scoreboard: 100 DEV tasks and 579 held-out public-training tasks (audit population and evaluation split excluded); the maker sees only generated
niche tasks. Controls: matched random cells (same templates, same number of literal positions) and a random-chain control for breeding.
"""
import argparse
import copy
import hashlib
import os
import json
import pathlib
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import head_arc_cells as H  # noqa: E402
import niches as N  # noqa: E402
import skillcore as S  # noqa: E402
import evolution_policy as EP  # noqa: E402

RUN_SRC = pathlib.Path(H.RUN_SRC)
DATA = pathlib.Path(H.DATA)
TILE_RUN = pathlib.Path("/srv/cellforge-inbox/head-20261001/runs/weights-head-niche-tile")
QUEUE = [(n, N.LIMITS[n]) for n in ("tiling", "association", "sequence", "glyph", "neighbor", "object", "crop", "ray",
                                    "recall", "induction", "seqops", "selcopy", "grammar", "successor", "agree",
                                    "gravity", "fillenc", "symmetry", "scale", "halfbool", "denoise",
                                    "objmove", "floodm", "raycross", "context2", "objmove2", "floodthru", "snapline", "ocrdigit", "ocrseq", "legend", "grammar2", "analogy", "tooluse", "lineconn", "cropcolor", "blockpick", "seqcrop", "countout", "outline", "objmove3", "blockred", "selfkron")]
SEED = 11
QUEUE += [(name, N.LIMITS[name]) for name in ('wallpruned', 'corridormarker', 'anchorassign', 'frameinfer', 'referencepaint')]


def now():
    return H.now()


def rand_cell(template, rng):
    return S.make_cell(template, {k: np.array([rng.uniform(-9, 9) for _ in range(int(np.prod(s)))]).reshape(s).tolist()
                                  for k, s in S.TEMPLATES[template]["shapes"].items()})


def parse_cells(raw, template):
    cells = []
    for entry in raw if isinstance(raw, list) else []:
        params = entry.get("params")
        if S.valid_params(template, params):
            cells.append(S.make_cell(template, params, task_id=entry.get("task_id"), rationale=str(entry.get("rationale", ""))[:300],
                                     **({'worker_receipt': entry['_worker_receipt']} if '_worker_receipt' in entry else {})))
    return cells


def json_task(task):
    return json.dumps(task)


# ------------------------------------------------------------------ prompts
def prompt_instance(niche, items):
    t = S.TEMPLATES[niche.template]
    return ("You write literal weight tensors for a grid cell (no code, no tools, no files). Template %s, tensors %s. Semantics: %s\n"
            "For EACH task below, infer from the demonstrations how the output is built from the input and write complete tensors (finite numbers in "
            "[-9, 9], every entry present; use 0 for unused) so the cell reproduces EVERY demonstration and the same rule on a new input. If "
            "previous_failed_attempt is present, read its predicted vs expected and write a genuinely different revision. Return ONLY a JSON array, "
            "one object per task: [{\"task_id\": str, \"params\": {tensor_name: nested lists}, \"rationale\": str naming concrete demonstration evidence}]."
            "\nTasks:\n%s" % (niche.template, {k: list(v) for k, v in t["shapes"].items()}, t["doc"], json.dumps(items)))


def prompt_general(niche, tasks, failing):
    t = S.TEMPLATES[niche.template]
    body = {"example_tasks": [{"demonstrations": x["train"]} for x in tasks]}
    if failing:
        body["previous_attempt"] = failing
    return ("You write the literal weight tensors of ONE grid cell (no code, no tools, no files) that must solve EVERY task of a family. Template %s, "
            "tensors %s. Semantics: %s\nThe example tasks below are samples of the family: each has its own demonstration pairs, and the same cell must "
            "solve all of them (and unseen ones of the family) given only its own demonstrations. If previous_attempt is present it lists your last "
            "tensors and failing tasks with predicted vs expected: write a genuinely different, better revision. Return ONLY a JSON array with ONE "
            "object: [{\"task_id\": \"general\", \"params\": {tensor_name: nested lists}, \"rationale\": str naming concrete evidence}]. Finite numbers "
            "in [-9, 9], every entry present.\n%s" % (niche.template, {k: list(v) for k, v in t["shapes"].items()}, t["doc"], json.dumps(body)))


def counterexamples(cell, tasks, limit=2):
    out = []
    for task in tasks:
        if len(out) >= limit:
            break
        if S.passes(cell, task):
            continue
        t = task["test"][0]
        try:
            pred = S.run(cell, t["input"], task["train"])
        except Exception as error:
            pred = "error:" + type(error).__name__
        out.append({"demonstrations": task["train"], "query": t["input"], "expected": t["output"], "predicted": pred})
    return out


# ------------------------------------------------------------------ driver
class Driver:
    def __init__(self, root, fake=False, workers=3, evolution=None, phase_context=None, fresh_inputs=None):
        self.root, self.fake, self.workers = root, fake, workers
        context_file = root / 'PHASE_CONTEXT.json'
        saved_context = H.load(context_file) if context_file.exists() else None
        self.phase_context = copy.deepcopy(phase_context if phase_context is not None else saved_context)
        if saved_context is not None and self.phase_context != saved_context:
            raise ValueError('phase context cannot change on resume')
        self._phase_incumbent = None
        self._phase_initial_router = None
        if self.phase_context is not None:
            import checkpoint_transitions as transitions
            context = self.phase_context
            if (set(context) != {'schema', 'phase', 'predecessor'}
                    or context['schema'] != 'cellforge-phase-context/1' or fake):
                raise ValueError('exact real-execution phase context required')
            phase = context['phase']
            if (set(phase) != {'plan_hash', 'index', 'epoch', 'kind', 'rounds'}
                    or type(phase['index']) is not int or not 0 <= phase['index'] < 16
                    or type(phase['epoch']) is not int or phase['epoch'] != phase['index'] // 2
                    or phase['kind'] != ('cells' if phase['index'] % 2 == 0 else 'router')
                    or type(phase['rounds']) is not int or not 1 <= phase['rounds'] <= 12
                    or not isinstance(phase['plan_hash'], str) or len(phase['plan_hash']) != 64
                    or any(c not in '0123456789abcdef' for c in phase['plan_hash'])):
                raise ValueError('invalid bounded alternating phase descriptor')
            if root.parent.name != 'phases' or root.name != 'weights-phase-%04d-%s' % (phase['index'], phase['kind']):
                raise ValueError('phase must use its explicit campaign phase directory')
            predecessor = context['predecessor']
            if predecessor is None:
                if phase['index'] != 0:
                    raise ValueError('only the initial cell phase may omit its predecessor')
            else:
                context['predecessor'] = transitions.validate_checkpoint(predecessor)
                previous = context['predecessor']['phase']
                if previous['index'] + 1 != phase['index'] or previous['plan_hash'] != phase['plan_hash']:
                    raise ValueError('phase must follow its exact predecessor in the same frozen plan')
        policy_file = root / 'EVOLUTION_POLICY.json'
        saved_policy = H.load(policy_file) if policy_file.exists() else None
        self.evolution = EP.configuration(evolution if evolution is not None else saved_policy)
        if saved_policy is not None and self.evolution != EP.configuration(saved_policy):
            raise ValueError('resume requires the saved evolution policy; use a separately authorized run for changes')
        # Presence is the opt-in boundary: an explicit all-disabled policy must
        # never fall back to the legacy full queue or legacy maker route.
        self.evolution_active = evolution is not None or saved_policy is not None
        if self.phase_context is not None:
            router_phase = self.phase_context['phase']['kind'] == 'router'
            if not self.evolution_active or self.evolution['router']['enabled'] != router_phase:
                raise ValueError('phase kind differs from its explicit evolution policy')
            if router_phase and self.evolution['niches']:
                raise ValueError('router phase cannot run founder stages')
        if any(name not in N.NICHES for name in self.evolution['niches']):
            raise ValueError('configured niche is not registered')
        self.worker_pool = None
        self._wave_rows = None
        self._phase = None
        self._parent_evidence = None
        self._maker_lock = threading.Lock()
        if fake and (self.evolution['workers']['enabled'] or self.evolution['router']['enabled']):
            raise ValueError('oracle mode cannot invoke the real worker pool')
        if self.evolution_active and not root.name.startswith('weights-'):
            raise ValueError('versioned evolution requires a runs/weights-* experiment directory')
        if self.evolution_active and saved_policy is None and (root / 'STATE.json').exists():
            raise ValueError('cannot retrofit a new policy onto an existing unversioned run')
        fresh_file = root / 'FRESH_INPUTS.json'
        saved_fresh = H.load(fresh_file) if fresh_file.exists() else None
        fresh_config = fresh_inputs if fresh_inputs is not None else saved_fresh
        self.fresh_inputs = None
        if fresh_config is not None:
            import fresh_inputs as F
            if not self.evolution_active or fake:
                raise ValueError('fresh baseline requires an explicit real evolution policy')
            if saved_fresh is not None and fresh_config != saved_fresh:
                raise ValueError('fresh input configuration cannot change on resume')
            if any((root / name).exists() for name in ('pop1_core.py', 'parent.json')):
                raise ValueError('fresh baseline cannot inherit historical seed artifacts')
            self.fresh_inputs = F.load(fresh_config)
        fresh = self.fresh_inputs
        root.mkdir(parents=True, exist_ok=True)
        pending_files = {}
        def provision(path, data):
            if self.evolution_active:
                pending_files[path] = data
            else:
                path.write_bytes(data)
        if fresh is None:
            for name in ("pop1_core.py", "parent.json"):
                if not (root / name).exists():
                    provision(root / name, (RUN_SRC / name).read_bytes())
        else:
            provision(fresh_file, (json.dumps(fresh['config'], sort_keys=True) + '\n').encode('utf-8'))
            provision(root / 'FRESH_BASELINE.json', fresh['manifest_bytes'])
            provision(root / 'DATA_SOURCE.json', fresh['source_bytes'])
        for name in ("skillcore.py", "cell_contracts.py", "wall_skill.py", "wall_avoid_skill.py", "wall_pruned_skill.py", "anchor_skill.py", "frame_skill.py", "reference_skill.py", "anchor_clip_skill.py", "anchor_assign_skill.py", "frame_infer_skill.py", "reference_paint_skill.py", "corridor_skill.py", "structcode.py", "structlib.py"):
            provision(root / name, (pathlib.Path(__file__).resolve().parent / name).read_bytes())
        evaluator = ("import importlib.util\nimport sys\nfrom pathlib import Path\n_h = Path(__file__).parent\nsys.path.insert(0, str(_h))\n_s = importlib.util.spec_from_file_location('skillcore', _h / 'skillcore.py')\n"
                     "_m = importlib.util.module_from_spec(_s)\n_s.loader.exec_module(_m)\n")
        if fresh is None:
            evaluator += "_m.load_legacy(_h / 'pop1_core.py')\n"
        if self.evolution_active:
            evaluator += ("import json\n\ndef predict(weights, task):\n"
                          "    bank = json.loads((_h / 'SKILL_BANK.json').read_text(encoding='utf-8'))\n"
                          "    return _m.predict_bank(bank, task, tensors=weights)\n")
        else:
            evaluator += "\ndef predict(weights, task):\n    return _m.predict(weights, task)\n"
        provision(root / 'arc_eval.py', evaluator.encode('utf-8'))
        if fresh is None:
            legacy_path = (root if (root / 'pop1_core.py').exists() else RUN_SRC) / 'pop1_core.py'
            parent_path = (root if (root / 'parent.json').exists() else RUN_SRC) / 'parent.json'
            parent = H.load(parent_path)['tensors']
            dev_ids = {x["id"] for x in H.load(RUN_SRC / "TASKS.json")["development"]}
            audit = lambda t: any(int(hashlib.sha256(s.encode()).hexdigest(), 16) % 5 == 0 for s in (t, t + ".json"))  # noqa: E731
            dev_rows = [{"id": i, "task": H.load(DATA / (i + ".json"))} for i in sorted(dev_ids)]
            held_rows = [{"id": p.stem, "task": H.load(p)} for p in sorted(DATA.glob("*.json")) if p.stem not in dev_ids and not audit(p.stem)]
            in_val = lambda r: int(hashlib.sha256(("val:" + r["id"]).encode()).hexdigest(), 16) % 2 == 0  # noqa: E731
            self.rows_dev, self.rows_val = dev_rows, [r for r in held_rows if in_val(r)]
            self.rows_rep = [r for r in held_rows if not in_val(r)]
            self.report_ids = [r['id'] for r in self.rows_rep]
        else:
            self.rows_dev = fresh['rows']['search_fitness']
            self.rows_val = fresh['rows']['development_selection']
            self.rows_rep, self.report_ids = [], fresh['report_ids']
        self.rows_sel = self.rows_dev + self.rows_val
        if self.phase_context is not None and (not self.rows_dev or not self.rows_val):
            raise ValueError('alternating phases require nonempty search and development-selection panels')
        self.all = S.Scorer(self.rows_dev + self.rows_val + self.rows_rep)
        self.cache_path = root.resolve().parent.parent / "ERRCACHE.pkl"  # shared by every run in this directory: restarts are warm
        if not self.evolution_active:
            self.all.load_cache(self.cache_path)
        self.registry = H.load(root / "REGISTRY.json") if (root / "REGISTRY.json").exists() else []
        self.state = H.load(root / "STATE.json") if (root / "STATE.json").exists() else {"done": [], "tried_pairs": [], "breed_round": 0, "makers_calls": 0}
        self._legacy_maker_calls = self.state['makers_calls'] - self.state.get('worker_calls_reserved', 0)
        if self.evolution_active:
            import evolution_manifest as manifest
            sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(pathlib.Path(__file__).resolve().parent.glob('*.py'))}
            if fresh is None:
                sources['legacy/pop1_core.py'] = hashlib.sha256(legacy_path.read_bytes()).hexdigest()
                sources['legacy/parent.json'] = hashlib.sha256(parent_path.read_bytes()).hexdigest()
            else:
                sources['fresh/input-manifest'] = hashlib.sha256(fresh['manifest_bytes']).hexdigest()
                sources['fresh/public-training-source'] = hashlib.sha256(fresh['source_bytes']).hexdigest()
            structures = pathlib.Path(S.__file__).parent / 'STRUCTURES.json'
            if structures.exists():
                provision(root / 'STRUCTURES.json', structures.read_bytes())
            elif (root / 'STRUCTURES.json').exists():
                raise ValueError('run-local structures differ from the live evaluator library')
            sources['live/STRUCTURES.json'] = hashlib.sha256(structures.read_bytes() if structures.exists() else b'absent').hexdigest()
            # Bind the actual saved dependencies as well as intended copies.
            # A modified resume artifact is rejected, never silently repaired.
            if saved_policy is not None:
                for path, data in pending_files.items():
                    if not path.exists() or path.read_bytes() != data:
                        raise ValueError('saved runtime dependency changed: ' + path.name)
            runtime = {p.name: p.read_bytes() for p in sorted(root.glob('*.py'))}
            runtime.update({p.name: data for p, data in pending_files.items() if p.suffix == '.py'})
            if fresh is None:
                runtime['pop1_core.py'] = legacy_path.read_bytes()
            if structures.exists():
                runtime['STRUCTURES.json'] = structures.read_bytes()
            sources.update({'runtime/' + name: hashlib.sha256(data).hexdigest() for name, data in runtime.items()})
            sources['search-fitness-task-content'] = EP.digest(self.rows_dev)
            sources['development-selection-task-content'] = EP.digest(self.rows_val)
            if self.phase_context is not None:
                sources['phase/context.json'] = EP.digest(self.phase_context)
            self.phase_scope_hash = EP.digest({'sources': {key: value for key, value in sources.items()
                                                           if not key.startswith('phase/')},
                                               'partitions': {'search_fitness': sorted(r['id'] for r in self.rows_dev),
                                                              'development_selection': sorted(r['id'] for r in self.rows_val),
                                                              'report': sorted(self.report_ids)}})
            if (self.phase_context is not None and self.phase_context['predecessor'] is not None
                    and self.phase_context['predecessor']['scope_hash'] != self.phase_scope_hash):
                raise ValueError('phase evaluator/task scope differs from its predecessor')
            self.manifest = manifest.freeze_manifest(root, self.evolution, sources, {
                'search_fitness': [r['id'] for r in self.rows_dev],
                'development_selection': [r['id'] for r in self.rows_val],
                'report': self.report_ids,
            })
            # Resume validation precedes overwriting any saved evaluator source.
            H.write(policy_file, self.evolution)
            if self.phase_context is not None:
                H.write(context_file, self.phase_context)
            for path, data in pending_files.items():
                path.write_bytes(data)
            if not (root / 'STANDARD.json').exists():
                H.write(root / 'STANDARD.json', {
                    'arms': {'llm': {}, 'random': {}}, 'seeds': [SEED], 'seeds_declared_before_run': True,
                    'budget_per_arm': self.evolution['workers']['max_calls'], 'niches': [],
                    'founders': [], 'genomes': [], 'proposals': [], 'scientific_acceptance': False,
                    'limitations': ['Operational evolution integration; not ONE_STANDARD qualification',
                                    'Detailed attempts and controls are in the worker and evolution journals',
                                    'No independent report evaluation or multi-seed scientific claim'],
                })
        # No run-local module or worker is started before the frozen inputs pass.
        if fresh is None:
            S.load_legacy(legacy_path)
            self.legacy, _ = S.from_tensors(parent)
            for cell in self.legacy:
                cell['niche'] = 'legacy'
            self.legacy_router = (parent['route.W'], parent['route.b'])
        else:
            # Keep the existing immutable-prefix mechanics, without claiming
            # these new random literals are historical or LLM-written cells.
            self.legacy = copy.deepcopy(fresh['baseline']['cells'])
            router = fresh['baseline']['bank']['tensors']
            self.legacy_router = (router['route.W'], router['route.b'])
        if self.evolution_active and len({cell['id'] for cell in self.legacy}) != len(self.legacy):
            raise ValueError('legacy router roster contains duplicate numerical expert identities')
        if self.phase_context is not None:
            self.import_phase_seed()
        if self.evolution['router']['enabled']:
            self.validate_router_state()
        if self.evolution['workers']['enabled']:
            from weight_workers import WeightWorkers
            extra_stops = (root.parent.parent / 'STOP',) if self.phase_context is not None else ()
            self.worker_pool = WeightWorkers(root, self.evolution['workers'], stop_paths=extra_stops)
        self.base = None

    def import_phase_seed(self):
        """Bind a predecessor archive before workers, keeping its incumbent separate."""
        predecessor = self.phase_context['predecessor']
        seed_hash = EP.digest(self.phase_context)
        if predecessor is None:
            self._phase_incumbent = S.export_bank(self.legacy, S.default_router(self.legacy, legacy=self.legacy_router))
        else:
            inherited = copy.deepcopy(predecessor['archive_cells'])
            archive = predecessor['archive_bank']
            router = {name: archive['tensors'][name] for name in ('route.W', 'route.b')}
            if S.export_bank(inherited, router) != archive:
                raise ValueError('inherited archive differs from the frozen typed templates or weights')
            if inherited[:len(self.legacy)] != self.legacy:
                raise ValueError('inherited archive does not preserve this phase legacy prefix')
            self._phase_incumbent = copy.deepcopy(predecessor['retained_bank'])
            # Validate the retained bank against real templates as well as the
            # pure transition schema; it may be a smaller prefix of the archive.
            self.phase_bank_cells(self._phase_incumbent, inherited)
            if self.phase_context['phase']['kind'] == 'router':
                self._phase_initial_router = copy.deepcopy(router)
            if 'phase_seed_hash' not in self.state:
                if self.registry or (self.root / 'STATE.json').exists():
                    raise ValueError('cannot retrofit phase inheritance onto existing unbound state')
                self.registry = inherited[len(self.legacy):]
                self.state.update(copy.deepcopy(predecessor['cell_state']))
            elif self.cells()[:len(inherited)] != inherited:
                raise ValueError('resumed phase altered or removed an inherited expert')
        if self.state.get('phase_seed_hash', seed_hash) != seed_hash:
            raise ValueError('saved phase state has a different seed binding')
        self.state['phase_seed_hash'] = seed_hash
        self.save()

    @staticmethod
    def phase_bank_cells(bank, available):
        by_id = {cell['id']: cell for cell in available}
        cells = [by_id[cid] for cid in bank['router']['expert_ids']]
        router = {name: bank['tensors'][name] for name in ('route.W', 'route.b')}
        if S.export_bank(cells, router) != bank:
            raise ValueError('phase bank differs from the available typed archive')
        return cells

    def phase_outcomes(self, bank, available):
        cells = self.phase_bank_cells(bank, available)
        router = {name: bank['tensors'][name] for name in ('route.W', 'route.b')}
        outcomes = {}
        for row in self.rows_sel:
            if self.shutdown_requested():
                raise RuntimeError('phase STOP during checkpoint scoring; no completion receipt may be issued')
            outcomes.update(self.all.score(cells, router, rows=[row])['outcomes'])
        return {'search_fitness': {row['id']: outcomes[row['id']] for row in self.rows_dev},
                'development_selection': {row['id']: outcomes[row['id']] for row in self.rows_val}}

    def run_phase(self):
        """Execute one explicit bounded phase and return evidence for its owner to publish."""
        import checkpoint_transitions as transitions
        import evolution_manifest as manifest
        if self.phase_context is None:
            raise RuntimeError('run_phase requires a frozen explicit epoch context')
        if (self.root / 'PHASE_CHECKPOINT.json').exists():
            raise RuntimeError('a published phase cannot be run again')
        if self.shutdown_requested():
            self.status(state='stopped', note='campaign or phase STOP before work')
            return None
        predecessor = self.phase_context['predecessor']
        incumbent_outcomes = self.phase_outcomes(self._phase_incumbent, self.cells())
        if predecessor is not None and incumbent_outcomes != predecessor['retained_outcomes']:
            raise ValueError('recomputed incumbent outcomes differ from predecessor evidence')
        self.run(self.phase_context['phase']['rounds'])
        if self.phase_worker_unsettled():
            self.status(state='stopped', note='worker safety halt or in-flight call blocks phase completion; reconcile receipts')
            return None
        status = H.load(self.root / 'STATUS.json')
        if (status.get('state') not in ('rounds_complete', 'plateau', 'router_disabled')
                or self.shutdown_requested() or any(self.state.get(key) is not None
                    for key in ('active_stage', 'active_wave', 'active_router_wave'))):
            return None
        journal = manifest.read_manifest_status(self.root)
        if journal['sealed'] or journal['pending_mutation_waves'] or journal['pending_router_waves']:
            raise RuntimeError('phase must settle all development work without a report before transition')
        cells = self.cells()
        bank = S.export_bank(cells, self.router_for(cells))
        cell_state = {key: copy.deepcopy(self.state[key]) for key in
                      ('done', 'tried_pairs', 'breed_round', 'mutate_round', 'evolution_seed', 'evolution_rng', 'curriculum')
                      if key in self.state}
        cell_state.setdefault('mutate_round', 0)
        # Normalize Random tuples to their recorded JSON representation.
        cell_state = json.loads(json.dumps(cell_state, allow_nan=False))
        checkpoint = transitions.make_checkpoint(self.phase_context['phase'], predecessor, {
            'manifest_hash': self.manifest['manifest_hash'], 'journal_head_hash': journal['head_event_hash'],
            'scope_hash': self.phase_scope_hash, 'archive_cells': cells, 'archive_bank': bank,
            'archive_outcomes': self.phase_outcomes(bank, cells), 'incumbent_bank': self._phase_incumbent,
            'incumbent_outcomes': incumbent_outcomes, 'cell_state': cell_state,
            'worker_calls_reserved': self.worker_pool.calls if self.worker_pool is not None else 0})
        if self.shutdown_requested() or self.phase_worker_unsettled():
            self.status(state='stopped', note='STOP or worker safety halt before phase publication')
            return None
        return checkpoint

    def phase_worker_unsettled(self):
        if self.worker_pool is None:
            return False
        status = self.worker_pool.status()
        # A halt blocks publication even when its final journal write failed.
        # Exhaustion alone is normal; a still-running call is never settled.
        return status['safety_halt'] or status['in_flight'] != 0

    # -- persistence
    def save(self):
        H.write(self.root / "REGISTRY.json", self.registry)
        if self.evolution_active:
            temporary = self.root / 'STATE.json.tmp'
            temporary.write_text(json.dumps(self.state, indent=2), encoding='utf-8')
            temporary.replace(self.root / 'STATE.json')
        else:
            H.write(self.root / "STATE.json", self.state)
        if not self.evolution_active:
            self.all.save_cache(self.cache_path)

    def status(self, **kw):
        H.write(self.root / "STATUS.json", dict(kw, utc=now(), cells=len(self.registry), pid=os.getpid()))

    def log(self, rec):
        with open(self.root / "HISTORY.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({k: v for k, v in rec.items() if not isinstance(v, (list, dict)) or k in ("held_out_gain_ids",)}), flush=True)

    # -- scoring against real ARC
    def cells(self):
        cells = self.legacy + self.registry
        if self.evolution_active:
            unique = {}
            for cell in cells:
                unique.setdefault(cell['id'], cell)
            return list(unique.values())
        return cells

    def split(self, outcomes):
        count = lambda rows: sum(outcomes[r["id"]] for r in rows)  # noqa: E731
        return {"dev": count(self.rows_dev), "val": count(self.rows_val), "report": count(self.rows_rep)}

    def arc(self, cells=None):
        if self.fresh_inputs is not None:
            raise RuntimeError('fresh report tasks are reserved; use development-only scoring')
        cells = cells if cells is not None else self.cells()
        router = S.default_router(cells, legacy=self.legacy_router)
        return self.all.score(cells, router, rows=self.rows_dev + self.rows_val + self.rows_rep)

    def sel(self, cells):
        """Selection score: dev + validation tasks only."""
        router = self.router_for(cells)
        return self.all.score(cells, router, rows=self.rows_sel)

    def router_for(self, cells):
        """Keep a learned router bound to its exact immutable expert order."""
        if not self.evolution['router']['enabled'] or 'router_genome' not in self.state:
            return self.reference_router(cells)
        import router_evolution as R
        saved = self.state['router_genome']
        ids = [cell['id'] for cell in cells]
        actual = R.make_genome(ids, saved['params'], saved['parents'])
        if actual != saved or ids != self.state.get('router_roster'):
            raise ValueError('active router genome differs from the frozen expert order or literals')
        return actual['params']

    def reference_router(self, cells):
        if self._phase_initial_router is not None:
            if [cell['id'] for cell in cells] != self.phase_context['predecessor']['archive_bank']['router']['expert_ids']:
                raise ValueError('inherited router requires the exact preceding cell-phase roster')
            return copy.deepcopy(self._phase_initial_router)
        return S.default_router(cells, legacy=self.legacy_router)

    def validate_router_state(self):
        """Reject uncertain progress before any worker is constructed on resume."""
        import evolution_manifest as manifest
        import router_evolution as R
        status = manifest.read_manifest_status(self.root)
        if status['sealed']:
            raise RuntimeError('sealed report forbids further router search')
        if (status['pending_router_waves'] or status['pending_mutation_waves']
                or self.state.get('active_router_wave') is not None):
            raise RuntimeError('interrupted router wave requires receipt reconciliation; no automatic paid replay')
        ids = [cell['id'] for cell in self.cells()]
        roster = self.state.get('router_roster')
        journal_roster = status['router_expert_ids']
        if roster is None:
            if journal_roster is not None or self.state.get('router_genome') is not None:
                raise ValueError('router state missing despite recorded router work')
            return
        if ids != roster or (journal_roster is not None and journal_roster != roster):
            raise ValueError('frozen router expert order differs from saved cells or journal')
        current = self.state['router_genome']
        self.router_for(self.cells())
        reference = R.make_genome(ids, self.reference_router(self.cells()))
        expected = status['latest_accepted_router_id'] or reference['hash']
        if current['hash'] != expected or self.state.get('router_round', 0) != (status['latest_router_wave'] or 0):
            raise ValueError('saved router checkpoint differs from settled journal evidence')

    def baseline(self):
        if self.base is None:
            self.base = (self.all.score(self.legacy, S.default_router(self.legacy, legacy=self.legacy_router),
                                        rows=self.rows_sel) if self.evolution_active else self.arc(self.legacy))
        return self.base

    def measure(self, stage, **extra):
        if self.evolution_active:
            return self.measure_development(stage, **extra)
        base = self.baseline()
        res = self.arc()
        now_split, base_split = self.split(res["outcomes"]), self.split(base["outcomes"])
        ids = sorted(k for k, v in res["outcomes"].items() if v and k not in {r["id"] for r in self.rows_dev})
        regress = [t for t, v in base["outcomes"].items() if v and not res["outcomes"][t]]
        rec = dict(stage=stage, utc=now(), cells=len(self.registry), depth2=sum(1 for c in self.registry if S.depth(c) > 1),
                   arc_dev=now_split["dev"], arc_val=now_split["val"], arc_report=now_split["report"], arc_held_out=now_split["val"] + now_split["report"],
                   baseline_dev=base_split["dev"], baseline_val=base_split["val"], baseline_report=base_split["report"],
                   baseline_held_out=base_split["val"] + base_split["report"], baseline_regressions=regress, held_out_solved_ids=ids, **extra)
        self.log(rec)
        board = H.load(self.root / "LEADERBOARD.json") if (self.root / "LEADERBOARD.json").exists() else []
        board.append({k: rec[k] for k in ("stage", "utc", "cells", "depth2", "arc_dev", "arc_val", "arc_report", "arc_held_out")})
        H.write(self.root / "LEADERBOARD.json", board)
        tensors = S.to_tensors(self.cells(), S.default_router(self.cells(), legacy=self.legacy_router))
        H.write(self.root / "FINAL_WEIGHTS.json", {"tensors": tensors})
        return rec

    def measure_development(self, stage, **extra):
        """New policy modes never score the report partition during search."""
        base, result = self.baseline(), self.sel(self.cells())
        rec = dict(stage=stage, utc=now(), cells=len(self.registry),
                   depth2=sum(S.depth(c) > 1 for c in self.registry),
                   arc_dev=sum(result['outcomes'][r['id']] for r in self.rows_dev),
                   arc_val=sum(result['outcomes'][r['id']] for r in self.rows_val),
                   arc_report=None, arc_held_out=None, report_evaluated=False,
                   baseline_dev=sum(base['outcomes'][r['id']] for r in self.rows_dev),
                   baseline_val=sum(base['outcomes'][r['id']] for r in self.rows_val),
                   baseline_regressions=[tid for tid, ok in base['outcomes'].items() if ok and not result['outcomes'][tid]],
                   scientific_acceptance=False,
                   worker_calls_reserved=self.worker_pool.calls if self.worker_pool is not None else 0,
                   control_scope='existing development controls; not a qualified equal-search-budget study', **extra)
        self.log(rec)
        board = H.load(self.root / 'LEADERBOARD.json') if (self.root / 'LEADERBOARD.json').exists() else []
        board.append(rec)
        H.write(self.root / 'LEADERBOARD.json', board)
        cells = self.cells()
        router = self.router_for(cells)
        H.write(self.root / 'FINAL_WEIGHTS.json', {'tensors': S.to_tensors(cells, router)})
        # Policy-mode arc_eval consumes this typed artifact for every checkpoint,
        # including runs that never opt into chaining or router evolution.
        H.write(self.root / 'SKILL_BANK.json', S.export_bank(cells, router))
        return rec

    # -- maker
    def ask(self, text, tag, chosen, oracle=None, templates=None):
        if self.evolution_active and not self.fake and self.worker_pool is None:
            raise RuntimeError('live proposals are disabled until a bounded worker allowance is explicitly configured')
        if self.worker_pool is None:
            self.state["makers_calls"] += 1
        if self.fake:  # fake = ORACLE: the true tensors, to test plumbing and measure the ceiling of the niches (never a maker result)
            (self.root / ("AUTHOR-" + tag)).mkdir(parents=True, exist_ok=True)
            return oracle()
        if self.worker_pool is not None:
            if templates is None or set(templates) != {row['id'] for row in chosen}:
                raise ValueError('worker requests require exact trusted task/template bindings')
            if self.worker_pool.attempts_for(tag):
                raise RuntimeError('worker tag already has durable attempts; reconcile receipts instead of dispatching it again')
            positions = sum(sum(int(np.prod(shape)) for shape in S.TEMPLATES[name]['shapes'].values())
                            for name in templates.values())
            try:
                return self.worker_pool.propose(text, tag, templates, S.valid_params, positions)
            finally:
                with self._maker_lock:
                    self.state['worker_calls_reserved'] = self.worker_pool.calls
                    self.state['makers_calls'] = self._legacy_maker_calls + self.worker_pool.calls
        return H.ask_maker(text, self.root / ("AUTHOR-" + tag), False, None, chosen)

    def stop_requested(self):
        return self.shutdown_requested() or (self.worker_pool is not None and self.worker_pool.stopped())

    def shutdown_requested(self):
        paths = [self.root / 'STOP']
        if self.phase_context is not None:
            paths.append(self.root.parent.parent / 'STOP')
        for path in paths:
            try:
                path.lstat()  # A dangling link named STOP is still a stop request.
            except FileNotFoundError:
                continue
            except OSError:
                return True  # An unreadable stop boundary cannot authorize work.
            return True
        return False

    def development_evidence(self, parents):
        """Per-cell execution, aligned on declared DEV tasks only; crashes are unknown."""
        rows = self.rows_dev
        evidence = {'schema': 'cellforge-parent-success/1', 'partition': 'search_fitness',
                    'task_ids': [r['id'] for r in rows], 'evaluator_hash': self.manifest['source_hashes_hash'],
                    'scope_hash': EP.digest({'partition': 'search_fitness', 'rows': rows}),
                    'outcomes': {}, 'case_evaluations': 0}
        for cell in parents:
            evidence['outcomes'][cell['id']] = {r['id']: S.task_outcome(cell, r['task']) for r in rows}
            evidence['case_evaluations'] += len(rows)
        return evidence

    def prepare_wave(self):
        parents = [c for c in self.registry if c['kind'] == 'cell' and c.get('niche') in N.NICHES]
        # Equal numeric identities are one parent, irrespective of niche aliases.
        parents = list({c['id']: c for c in parents}.values())
        self._parent_evidence = None
        self._wave_rows = None
        self._phase = None
        if self.evolution['parent_selection'] == 'complementary_success' or self.evolution['curriculum']['enabled']:
            evidence = self.development_evidence(parents)
            self._parent_evidence = evidence
            H.write(self.root / ('PARENT_EVIDENCE-%04d.json' % self.state['breed_round']), evidence)
            if parents:
                import evolution_manifest as manifest
                manifest.append_event(self.root, 'fitness_evaluation', {
                    key: evidence[key] for key in ('partition', 'task_ids', 'evaluator_hash', 'scope_hash', 'outcomes')})
            if self.evolution['curriculum']['enabled']:
                state = self.state.setdefault('curriculum', {'wave': 0, 'exposure': {}, 'recovery_anchors': []})
                ids, self._phase = EP.curriculum_tasks(self.evolution['curriculum'], state, evidence)
                by_id = {r['id']: r for r in self.rows_dev}
                self._wave_rows = [by_id[tid] for tid in ids]
                self._parent_evidence = dict(evidence, task_ids=ids,
                    scope_hash=EP.digest({'source_scope': evidence['scope_hash'], 'task_ids': ids}),
                    outcomes={cid: {tid: row.get(tid) for tid in ids} for cid, row in evidence['outcomes'].items()})
            self.save()
        return parents

    def record_competition(self, stage, cells, tasks_for):
        """Host-scored proposal competition; records losses as well as champions."""
        groups, records = {}, []
        for cell in cells:
            tasks = tasks_for(cell)
            outcomes = [S.task_outcome(cell, task) for task in tasks]
            record = {'cell_id': cell['id'], 'task_id': cell.get('task_id'), 'outcomes': outcomes,
                      'scope_hash': EP.digest(tasks), 'valid': all(x is not None for x in outcomes),
                      'successes': sum(x is True for x in outcomes), 'worker_receipt': cell.get('worker_receipt')}
            records.append(record)
            groups.setdefault(cell.get('task_id'), []).append((record, cell))
        champions = {}
        for tid, rows in groups.items():
            valid = [(r, c) for r, c in rows if r['valid']]
            if valid:
                _, champion = min(valid, key=lambda pair: (-pair[0]['successes'], pair[1]['id']))
                champions[tid] = champion
        self.log({'stage': stage + ':competition', 'utc': now(), 'partition': 'search_fitness',
                  'candidates': records, 'champions': {tid: c['id'] for tid, c in champions.items()},
                  'claim': 'measured search-fitness champion only'})
        return champions

    # -- import of the finished tiling bank (earlier run, same provenance)
    def import_tiling(self):
        tensors = H.load(TILE_RUN / "FINAL_WEIGHTS.json")["tensors"]
        cells, _ = S.from_tensors(tensors)
        niche, patterns = N.NICHES["tiling"], N.NICHES["tiling"].patterns(60, SEED)
        tasks = {p["id"]: niche.task(p, SEED * 1000 + i) for i, p in enumerate(patterns)}
        imported = 0
        for c in cells:
            if c["kind"] == "cell" and c["template"] == "tile_orient":
                for p in patterns:
                    if S.passes(c, tasks[p["id"]]):
                        c.update(niche="tiling", pattern=p["id"], origin="maker gpt-6.1-sol low, tiling run 2026-10-01 (imported)", generation=1)
                        self.registry.append(c)
                        imported += 1
                        break
        self.state["done"].append("tiling")
        return imported

    # -- niche stages
    def stage(self, name, limit):
        if self.evolution['router']['enabled'] and 'router_roster' in self.state:
            raise RuntimeError('founder stages cannot change a frozen router roster')
        niche = N.NICHES[name]
        if self.evolution_active:
            if self.state.get('active_stage') is not None:
                raise RuntimeError('interrupted niche stage requires receipt reconciliation before resume')
            self.state['active_stage'] = {'name': name, 'limit': limit, 'stage': 'inflight'}
            self.save()
        self.status(stage=name, phase="authoring")
        if name == "tiling" and (TILE_RUN / "FINAL_WEIGHTS.json").exists() and not self.fake and self.phase_context is None:
            n = self.import_tiling()
            self.log({"stage": "niche:tiling", "utc": now(), "imported_cells": n, "note": "imported finished bank; no new maker calls"})
            if self.evolution_active:
                self.state.pop('active_stage')
                self.save()
            return self.measure("niche:tiling", niche_cells=n)
        (self.instance_stage if not niche.general else self.general_stage)(niche, limit)
        if self.evolution_active and self.stop_requested():
            self.save()
            return
        self.state["done"].append(name)
        if self.evolution_active:
            self.state.pop('active_stage')
        self.save()

    def instance_stage(self, niche, limit):
        patterns, tasks, fresh = [], {}, []
        for i, p in enumerate(niche.patterns(limit, SEED)):
            try:
                tasks[p["id"]] = niche.task(p, SEED * 1000 + i)
                fresh.append({"id": p["id"], "task": niche.task(p, SEED * 1000 + 500 + i)})
                patterns.append(p)
            except RuntimeError:  # a pattern that cannot yield a verifiable task is dropped, never faked
                self.log({"stage": "niche:%s:skip" % niche.name, "utc": now(), "skipped_pattern": p["id"]})
        by_id = {p["id"]: p for p in patterns}
        attempts, bank, charged_total, rand_total = {}, {}, 0, [0, 0, 0]
        for gen in range(1, 4):
            if self.stop_requested():
                break
            todo = [pid for pid in by_id if pid not in bank]
            if not todo:
                break
            batches = [todo[i:i + 10] for i in range(0, len(todo), 10)]

            def work(args):
                b, ids = args
                items = []
                for pid in ids:
                    item = {"task_id": pid, "demonstrations": tasks[pid]["train"]}
                    if attempts.get(pid):
                        cell = attempts[pid][-1]
                        item["previous_failed_attempt"] = (counterexamples(cell, [tasks[pid]], 1) or [{"note": "passed"}])[0]
                        item["previous_failed_attempt"]["params"] = cell["params"]
                    items.append(item)
                return self.ask(prompt_instance(niche, items), "%s-%d-%d" % (niche.name, gen, b), [{"id": i} for i in ids],
                                lambda: [{"task_id": pid, "params": by_id[pid]["truth"], "rationale": "oracle"} for pid in ids],
                                templates={pid: niche.template for pid in ids})

            with ThreadPoolExecutor(self.workers) as pool:
                raws = list(pool.map(work, list(enumerate(batches))))
            submitted = [c for raw in raws for c in parse_cells(raw, niche.template) if c.get("task_id") in by_id]
            control_tasks = [c['task_id'] for c in submitted]
            if self.worker_pool is not None:
                reservations = [record for b in range(len(batches))
                                for record in self.worker_pool.attempts_for('%s-%d-%d' % (niche.name, gen, b))]
                control_tasks = [tid for record in reservations for tid in record['templates']]
                charged = sum(record['charged_positions'] for record in reservations)
            else:
                charged = sum(S.positions(c) for c in submitted)
            charged_total += charged
            champions = (self.record_competition('niche:%s:gen%d' % (niche.name, gen), submitted,
                                                 lambda c: [tasks[c['task_id']]]) if self.worker_pool is not None else None)
            for c in submitted:
                attempts.setdefault(c["task_id"], []).append(c)
                if champions is not None and champions.get(c['task_id']) is not c:
                    continue
                if S.passes(c, tasks[c["task_id"]]):
                    author = ('maker %s %s' % (self.evolution['workers']['model'], self.evolution['workers']['reasoning'])
                              if self.worker_pool is not None else 'maker gpt-6.1-sol low')
                    c.update(niche=niche.name, pattern=c["task_id"], origin=author, generation=gen)
                    bank[c["task_id"]] = c
            for k, seed in enumerate((501, 502, 503)):
                r = random.Random(seed * 100 + gen)
                rand_total[k] += sum(S.passes(rand_cell(niche.template, r), tasks[tid]) for tid in control_tasks)
            fresh_hit = S.Scorer(fresh).score(list(bank.values()))["solved"] if bank else 0
            self.log({"stage": "niche:%s:gen%d" % (niche.name, gen), "utc": now(), "submitted": len(submitted), "eligible_total": len(bank),
                      "patterns": len(patterns), "fresh_transfer": fresh_hit, "charged_positions_per_arm": charged,
                      "random_eligible_total": list(rand_total)})
        self.registry.extend(bank.values())
        self.measure("niche:" + niche.name, niche_cells=len(bank), niche_patterns=len(patterns), charged_positions_per_arm=charged_total,
                     random_eligible=list(rand_total))

    def general_stage(self, niche, limit):
        patterns = niche.patterns(limit, SEED)
        tasks = [niche.task(p, SEED * 1000 + i) for i, p in enumerate(patterns)]
        fresh = [niche.task(p, SEED * 1000 + 500 + i) for i, p in enumerate(patterns)]
        best, failing, charged_total, rand_rates = None, None, 0, []
        for gen in range(1, 4):
            if self.stop_requested():
                break
            shown = tasks[:3] if gen == 1 else tasks[gen: gen + 3]
            raw = self.ask(prompt_general(niche, shown, failing), "%s-%d" % (niche.name, gen), [{"id": "general"}],
                           lambda: [{"task_id": "general", "params": patterns[0]["truth"], "rationale": "oracle"}],
                           templates={'general': niche.template})
            cells = parse_cells(raw, niche.template)
            if self.worker_pool is not None:
                reservations = self.worker_pool.attempts_for('%s-%d' % (niche.name, gen))
                charged = sum(record['charged_positions'] for record in reservations)
                control_count = len(reservations)
            else:
                charged = sum(S.positions(c) for c in cells)
                control_count = max(1, len(cells))
            charged_total += charged
            r = random.Random(900 + gen)
            random_groups = []
            for arm in range(3):
                candidates = [rand_cell(niche.template, r) for _ in range(control_count)]
                # A single random candidate is evaluated unchanged on every task.
                rows = [{'cell_id': c['id'], 'params': c['params'],
                         'outcomes': [S.task_outcome(c, task) for task in tasks]} for c in candidates]
                rates = [sum(value is True for value in row['outcomes']) / len(tasks) for row in rows] if tasks else []
                random_groups.append({'arm': arm, 'candidates': rows, 'best_rate': max(rates, default=0.0)})
            rand_rates.append([group['best_rate'] for group in random_groups])
            if self.worker_pool is not None:
                self.log({'stage': 'niche:%s:gen%d:random_competition' % (niche.name, gen), 'utc': now(),
                          'partition': 'search_fitness', 'scope_hash': EP.digest(tasks), 'groups': random_groups,
                          'charged_positions_per_arm': charged, 'reserved_proposals': control_count})
            if not cells:
                self.log({"stage": "niche:%s:gen%d" % (niche.name, gen), "utc": now(), "submitted": 0})
                continue
            if self.worker_pool is not None:
                champion = self.record_competition('niche:%s:gen%d' % (niche.name, gen), cells, lambda c: tasks).get('general')
                if champion is None:
                    continue
                cell = champion
            else:
                cell = cells[0]
            rate = float(np.mean([S.passes(cell, t) for t in tasks]))
            fresh_rate = float(np.mean([S.passes(cell, t) for t in fresh]))
            self.log({"stage": "niche:%s:gen%d" % (niche.name, gen), "utc": now(), "submitted": 1, "pass_rate": rate, "fresh_pass_rate": fresh_rate,
                      "charged_positions_per_arm": charged, "random_pass_rate": rand_rates[-1]})
            if best is None or rate > best[0]:
                best = (rate, cell)
            if rate >= 0.9:
                break
            failing = {"params": cell["params"], "failing_tasks": counterexamples(cell, tasks, 2)}
        added = 0
        if best and best[0] >= 0.9:
            author = ('maker %s %s' % (self.evolution['workers']['model'], self.evolution['workers']['reasoning'])
                      if self.worker_pool is not None else 'maker gpt-6.1-sol low')
            best[1].update(niche=niche.name, pattern="*", origin=author, generation=gen, general=True)
            self.registry.append(best[1])
            added = 1
        self.measure("niche:" + niche.name, niche_cells=added, best_pass_rate=best[0] if best else 0.0, charged_positions_per_arm=charged_total)

    # -- breeding
    def pattern_of(self, cell, rng):
        niche = N.NICHES[cell["niche"]]
        patterns = niche.patterns(N.LIMITS[cell["niche"]], SEED)
        if cell.get("pattern") not in (None, "*"):
            return next(p for p in patterns if p["id"] == cell["pattern"])
        return rng.choice(patterns)

    def sample_input(self, pattern, rng):
        if pattern["niche"] == "association":
            colors = sorted(pattern["map"]) + pattern["ident"][:3]
            h, w = rng.randint(3, 6), rng.randint(3, 6)
            return [[rng.choice(colors) for _ in range(w)] for _ in range(h)]
        if pattern["niche"] == "glyph":
            return N.Glyph._pair(rng)["input"]
        return pattern["sample_input"](rng)

    def composite_tasks(self, a, b, rng, n=3):
        pa, pb = self.pattern_of(a, rng), self.pattern_of(b, rng)
        tasks = []
        for _ in range(n):
            pairs = []
            for _ in range(4):
                try:
                    x = self.sample_input(pa, rng)
                    mid = pa["apply"](x)
                    y = pb["apply"](mid)
                except (ValueError, IndexError, TypeError, KeyError):  # e.g. a transposing tile transform meeting a non-square grid: this composition does not exist
                    return None
                if max(len(y), len(y[0])) > 30 or y == mid or y == x:
                    return None
                pairs.append({"input": x, "output": y})
            tasks.append({"train": pairs[:3], "test": pairs[3:]})
        return tasks

    def breed(self, n_pairs=300, cap=40):
        rng = random.Random(1000 + self.state["breed_round"])
        parents = self.prepare_wave() if self.evolution_active else [c for c in self.registry if c["kind"] == "cell" and c.get("niche") in N.NICHES]
        if self._phase is not None:
            n_pairs = self._phase['pairs']
        tried = {tuple(p) for p in self.state["tried_pairs"]}
        if self.evolution['parent_selection'] == 'complementary_success':
            self.state.setdefault('evolution_seed', 1000 + self.state['breed_round'])
            if self.state.get('evolution_rng') is not None:
                rng.setstate(EP.tuple_tree(self.state['evolution_rng']))
            rng_before = rng.getstate()
            compatible = []
            import cell_contracts as contracts
            references = {c['id']: S.cell_reference(c) for c in parents}
            for a in parents:
                for b in parents:
                    if a['id'] == b['id'] or (a['id'], b['id']) in tried:
                        continue
                    try:
                        contracts.require_chain(references[a['id']], references[b['id']])
                    except ValueError:
                        continue
                    compatible.append((a['id'], b['id']))
            selected, receipt = EP.complementary_pairs([c['id'] for c in parents], compatible,
                self._parent_evidence, rng, n_pairs, self._phase['alpha'] if self._phase else self.evolution['alpha'])
            receipt.update(wave=self.state['breed_round'], seed=self.state['evolution_seed'],
                           rng_before_selection=rng_before, rng_after_selection=rng.getstate())
            H.write(self.root / ('PARENT_SELECTION-%04d.json' % self.state['breed_round']), receipt)
            by_id = {c['id']: c for c in parents}
            pairs = [(by_id[a], by_id[b]) for a, b in selected]
        else:
            pairs = [(a, b) for a in parents for b in parents if a["niche"] != b["niche"] and (a["id"], b["id"]) not in tried]
            rng.shuffle(pairs)
        admitted, evaluated, solved, ctrl, rejected_useless = [], 0, 0, 0, 0
        base_cells = self.cells()
        current = self.sel(base_cells)
        for a, b in pairs[:n_pairs]:
            if self.stop_requested():
                break
            self.state["tried_pairs"].append([a["id"], b["id"]])
            tasks = self.composite_tasks(a, b, rng)
            if tasks is None:
                continue
            evaluated += 1
            compose = S.typed_chain_cell if self.evolution['typed_chains'] else S.chain_cell
            child = compose(a, b, niche="chain", origin="crossover chain(%s,%s)" % (a["niche"], b["niche"]), round=self.state["breed_round"] + 1)
            ok = all(S.passes(child, t) for t in tasks) and sum(S.passes(a, t) or S.passes(b, t) for t in tasks) <= 1
            r = random.Random(rng.random())
            ctrl += all(S.passes(S.chain_cell(rand_cell(a["template"], r), rand_cell(b["template"], r)), t) for t in tasks)
            if not ok:
                continue
            solved += 1
            if len(admitted) >= cap:
                continue
            cur_cells = base_cells + admitted
            router = S.default_router(cur_cells, legacy=self.legacy_router)
            unsolved = [r for r in self.rows_sel if not current["outcomes"][r["id"]]]
            if not self.all.contends(child, cur_cells, router, unsolved):  # cannot be routed to on any unsolved task: no gain possible, skip the full pass
                rejected_useless += 1
                continue
            trial = self.sel(base_cells + admitted + [child])  # SELECTION on real ARC dev+validation: must strictly gain, never regress
            lost = [t for t, v in current["outcomes"].items() if v and not trial["outcomes"][t]]
            if trial["solved"] > current["solved"] and not lost:
                child.update(composite_fitness=1.0, a=a["niche"] + ":" + str(a.get("pattern")), b=b["niche"] + ":" + str(b.get("pattern")),
                             arc_selection_gain=trial["solved"] - current["solved"])
                admitted.append(child)
                current = trial
            else:
                rejected_useless += 1
        self.registry.extend(admitted)
        if self.evolution['parent_selection'] == 'complementary_success':
            self.state['evolution_rng'] = rng.getstate()
        if self.evolution_active:
            import evolution_manifest as manifest
            manifest.append_event(self.root, 'breeding_wave', {
                'wave': self.state['breed_round'], 'operator': 'chain',
                'parents': [c['id'] for c in parents], 'seed': self.state.get('evolution_seed', 1000 + self.state['breed_round']),
                'scope_hash': self._parent_evidence['scope_hash'] if self._parent_evidence else None,
                'children': [c['id'] for c in admitted], 'rng_state_after': rng.getstate(),
            })
        self.state["breed_round"] += 1
        self.save()
        combos = {}
        for c in admitted:
            combos[c["origin"]] = combos.get(c["origin"], 0) + 1
        return self.measure("breed:round%d" % self.state["breed_round"], pairs_evaluated=evaluated, children_valid=solved,
                            children_admitted=len(admitted), children_rejected_no_arc_gain=rejected_useless, random_chain_valid=ctrl,
                            admitted_by_pair=combos)

    # -- maker mutation of near-miss cells, selected on real ARC dev+validation, with a matched random-edit control
    def mutation_targets(self, cells, cur, n_targets):
        seen, targets = set(), []
        rows = self._wave_rows if self._wave_rows is not None else self.rows_dev if self.evolution_active else self.rows_sel
        for row in rows:
            if cur["outcomes"][row["id"]]:
                continue
            errs = [(self.all.demo_error(c, row), i) for i, c in enumerate(cells) if c["kind"] == "cell" and c.get("niche") != "legacy"]
            if not errs:
                continue
            e, i = min(errs)
            ident = float(np.mean([1.0 if np.asarray(d["input"]).shape != np.asarray(d["output"]).shape
                                   else float(np.mean(np.asarray(d["input"]) != np.asarray(d["output"]))) for d in row["task"]["train"]]))
            # a REAL near-miss explains at least half of the change the task needs; a small pixel error on a task that barely changes is just the identity
            broad = self._phase is not None and self._phase['name'] in ('exploration', 'recovery')
            if 0 < e <= (1.0 if broad else 0.5 * ident) and ident > 0 and cells[i]["id"] not in seen:
                seen.add(cells[i]["id"])
                targets.append((e, row, cells[i]))
        return sorted(targets, key=lambda t: t[0])[:n_targets]

    def mutation_prompt(self, targets):
        items = []
        for e, row, cell in targets:
            task = row["task"]
            d = task["train"][0]
            try:
                pred = S.run(cell, d["input"], task["train"][1:])
            except Exception as error:
                pred = "error:" + type(error).__name__
            items.append({"task_id": row["id"], "template": cell["template"], "current_params": cell["params"], "demonstrations": task["train"],
                          "counterexample": {"input": d["input"], "expected": d["output"], "predicted": pred}})
        template_docs = {i["template"]: S.TEMPLATES[i["template"]]["doc"] for i in items}
        text = ("You revise literal weight tensors of grid cells (no code, no tools, no files). For EACH item, the current cell nearly fits the task but fails. "
                "Read the counterexample (expected vs predicted), and write revised complete tensors of the SAME template and shapes (finite numbers in [-9, 9]) "
                "that reproduce every demonstration and generalise. Template semantics: %s\nReturn ONLY a JSON array: [{\"task_id\": str, \"params\": "
                "{tensor_name: nested lists}, \"rationale\": str}].\nItems:\n%s" % (json.dumps(template_docs), json.dumps(items)))
        return items, text

    def mutate(self, n_targets=6):
        if self._phase is not None:
            n_targets = self._phase['mutations']
        cells = self.cells()
        cur = self.sel(cells)
        targets = self.mutation_targets(cells, cur, n_targets)
        if not targets:
            return self.measure("mutate:round%d" % (self.state.get("mutate_round", 0) + 1), targets=0, note="no near-miss cells on the selection tasks")
        items, text = self.mutation_prompt(targets)
        rnd = self.state.get("mutate_round", 0) + 1
        if self.evolution['curriculum']['enabled']:
            exposure = self.state['curriculum']['exposure']
            for _, row, _ in targets:
                exposure[row['id']] = exposure.get(row['id'], 0) + 1
            self.save()
        if self.worker_pool is not None:
            return self.mutate_population(targets, text, rnd)
        raw = self.ask(text, "mutate-%d" % rnd, [{"id": r["id"]} for _, r, _ in targets], lambda: [],
                       templates={row['id']: cell['template'] for _, row, cell in targets})
        by_task = {e.get("task_id"): e for e in raw if isinstance(raw, list)} if isinstance(raw, list) else {}
        admitted, tried, rand_admitted, charged, declined = [], 0, 0, 0, 0
        current = cur
        rng = random.Random(7000 + rnd)
        for e, row, cell in targets:
            entry = by_task.get(row["id"])
            if not entry or not S.valid_params(cell["template"], entry.get("params")):
                declined += 1
                continue
            same = np.allclose(np.concatenate([np.asarray(v, float).ravel() for v in entry["params"].values()]),
                               np.concatenate([np.asarray(cell["params"][k], float).ravel() for k in entry["params"]]))
            if same or str(entry.get("rationale", "")).lower().startswith(("no valid", "no satisfying", "no exact", "no revision", "no tensors", "cannot", "no such")):
                declined += 1  # the maker states that no tensors of this template can express the rule: a structural limit, not a weight problem
                continue
            tried += 1
            child = S.make_cell(cell["template"], entry["params"], niche=cell["niche"], pattern=cell.get("pattern"), parents=[cell["id"]],
                                origin="maker mutation round %d" % rnd, generation=rnd,
                                **({'worker_receipt': entry['_worker_receipt']} if entry.get('_worker_receipt') else {}))
            charged += S.positions(child)
            old = np.concatenate([np.asarray(v, float).ravel() for v in cell["params"].values()])
            new = np.concatenate([np.asarray(v, float).ravel() for v in child["params"].values()])
            changed = max(1, int((old != new).sum()))
            flat = old.copy()
            for k in rng.sample(range(len(flat)), min(changed, len(flat))):
                flat[k] = rng.uniform(-9, 9)
            rparams, at = {}, 0
            for k, v in cell["params"].items():
                size = int(np.asarray(v).size)
                rparams[k] = flat[at:at + size].reshape(np.asarray(v).shape).tolist()
                at += size
            rchild = S.make_cell(cell["template"], rparams, niche=cell["niche"])
            rtrial = self.sel(cells + admitted + [rchild])
            rand_admitted += rtrial["solved"] > current["solved"]
            trial = self.sel(cells + admitted + [child])
            lost = [t for t, v in current["outcomes"].items() if v and not trial["outcomes"][t]]
            if trial["solved"] > current["solved"] and not lost:
                child["arc_selection_gain"] = trial["solved"] - current["solved"]
                admitted.append(child)
                current = trial
        self.registry.extend(admitted)
        self.state["mutate_round"] = rnd
        self.save()
        return self.measure("mutate:round%d" % rnd, targets=len(targets), mutations_tried=tried, mutations_declined_structural=declined, mutations_admitted=len(admitted),
                            random_edits_that_would_gain=rand_admitted, charged_positions_per_arm=charged)

    def mutate_population(self, targets, text, rnd):
        """Compare every reserved proposal and paired full-template random edit.

        A proposal replaces a complete tensor set, so both arms are charged that
        entire vocabulary even for neutral/invalid/omitted responses. Group
        ranking happens only after all candidate and control outcomes are saved.
        """
        import evolution_manifest as manifest
        tag = 'mutate-%d' % rnd
        by_task = {row['id']: (row, parent) for _, row, parent in targets}
        templates = {tid: parent['template'] for tid, (_, parent) in by_task.items()}
        raw = self.ask(text, tag, [{'id': tid} for tid in by_task], lambda: [], templates=templates)
        entries = {(entry['_worker_receipt']['call'], entry['task_id']): entry for entry in raw}
        reservations = self.worker_pool.attempts_for(tag)
        base_cells, current = self.cells(), self.sel(self.cells())
        trials, controls, records, charged, invalid = [], [], [], 0, 0
        partition_ids = {'search_fitness': [r['id'] for r in self.rows_dev],
                         'development_selection': [r['id'] for r in self.rows_val]}
        def partitioned(outcomes):
            return {partition: {tid: outcomes[tid] for tid in ids} for partition, ids in partition_ids.items()}
        for reservation in reservations:
            call = reservation['call']
            if reservation['templates'] != templates:
                raise ValueError('worker reservation differs from the frozen mutation scope')
            for tid, (row, parent) in by_task.items():
                count = S.positions(parent)
                charged += count
                # Seed is stable per reserved call and task, independent of
                # completion order or whether the writer returned valid output.
                seed = int(EP.digest({'wave': rnd, 'call': call, 'task': tid})[:16], 16)
                random_cell = rand_cell(parent['template'], random.Random(seed))
                random_cell.update(niche=parent['niche'], pattern=parent.get('pattern'), parents=[parent['id']],
                                   origin='paired random full-template proposal', generation=rnd)
                existing_ids = {cell['id'] for cell in base_cells}
                random_trial = current if random_cell['id'] in existing_ids else self.sel(base_cells + [random_cell])
                controls.append((random_cell, random_trial))
                entry = entries.get((call, tid))
                candidate = None
                candidate_trial = None
                outcome = 'invalid_or_missing'
                if entry is not None:
                    candidate = S.make_cell(parent['template'], entry['params'], niche=parent['niche'],
                        pattern=parent.get('pattern'), task_id=tid, parents=[parent['id']], generation=rnd,
                        origin='maker mutation round %d' % rnd, worker_receipt=entry['_worker_receipt'])
                    candidate_trial = current if candidate['id'] in existing_ids else self.sel(base_cells + [candidate])
                    outcome = 'neutral' if candidate['id'] == parent['id'] else 'valid'
                    trials.append((candidate, candidate_trial))
                else:
                    invalid += 1
                records.append({'call': call, 'task_id': tid, 'parent_id': parent['id'], 'seed': seed,
                    'operator': 'complete_tensor_replacement', 'charged_positions_per_arm': count,
                    'outcome': outcome, 'candidate_id': candidate['id'] if candidate else None,
                    'candidate_outcomes': partitioned(candidate_trial['outcomes']) if candidate_trial else None,
                    'random_id': random_cell['id'], 'random_outcomes': partitioned(random_trial['outcomes']),
                    'random_params': random_cell['params'],
                    'candidate_params': candidate['params'] if candidate else None})
        manifest.append_event(self.root, 'mutation_competition', {'wave': rnd, 'operator': 'complete_tensor_replacement',
            'partition': 'development', 'task_ids': partition_ids,
            'baseline_outcomes': partitioned(current['outcomes']), 'attempts': records,
            'charged_positions_per_arm': charged, 'invalid_or_missing': invalid})

        def eligible(trial):
            return selection_score(trial) > selection_score(current) and not any(
                ok and not trial['outcomes'][tid] for tid, ok in current['outcomes'].items())

        def selection_score(trial):
            return sum(trial['outcomes'][tid] for tid in partition_ids['development_selection'])

        # Rank within each arm against exactly the same frozen incumbent bank.
        # Both champions are preserved; only the LLM arm enters its live archive.
        ranked = sorted(((c, r) for c, r in trials if eligible(r)), key=lambda item: (-selection_score(item[1]), item[0]['id']))
        random_ranked = sorted(((c, r) for c, r in controls if eligible(r)), key=lambda item: (-selection_score(item[1]), item[0]['id']))
        champion = ranked[0][0] if ranked else None
        random_champion = random_ranked[0][0] if random_ranked else None
        if champion is not None:
            champion['arc_selection_gain'] = ranked[0][1]['solved'] - current['solved']
            self.registry.append(champion)
        manifest.append_event(self.root, 'mutation_champions', {'wave': rnd,
            'candidate_id': champion['id'] if champion else None,
            'random_id': random_champion['id'] if random_champion else None,
            'claim': 'group-relative development selection; not PPO/GRPO or independent generalisation'})
        self.state['mutate_round'] = rnd
        self.save()
        return self.measure('mutate:round%d' % rnd, targets=len(targets), mutations_tried=len(records),
                            mutations_admitted=int(champion is not None), invalid_or_missing=invalid,
                            charged_positions_per_arm=charged, random_champion_id=random_champion['id'] if random_champion else None)

    # -- fixed-roster literal router evolution (explicit policy only)
    def run_router(self, max_rounds):
        import router_evolution as R
        cells = self.cells()
        ids = [cell['id'] for cell in cells]
        if len(ids) > self.evolution['router']['max_experts']:
            raise ValueError('router roster exceeds the configured expert/parameter bound')
        if 'router_roster' not in self.state:
            reference = R.make_genome(ids, self.reference_router(cells))
            # Validate every typed expert before freezing the live roster.
            S.export_bank(cells, reference['params'])
            self.state.update(router_roster=ids, router_genome=reference, router_round=0)
            self.save()
        self.router_for(cells)
        self.measure('router:reference', router_id=self.state['router_genome']['hash'])
        if self.worker_pool is None:
            self.status(state='router_disabled', note='bounded worker allowance is disabled; no proposal submitted')
            return
        for _ in range(max_rounds):
            if self.stop_requested():
                self.status(state='stopped', note='router STOP or worker allowance exhausted')
                return
            self.evolve_router()
        self.status(state='rounds_complete', note='bounded fixed-roster router phase completed')

    def evolve_router(self):
        import evolution_manifest as manifest
        import router_evolution as R
        if self.worker_pool is None:
            raise RuntimeError('router proposals require an explicitly enabled bounded worker pool')
        self.validate_router_state()
        cells = self.cells()
        ids = self.state['router_roster']
        incumbent = R.make_genome(ids, self.router_for(cells), self.state['router_genome']['parents'])
        wave = self.state.get('router_round', 0) + 1
        tag = 'router-%d' % wave
        if self.worker_pool.attempts_for(tag):
            raise RuntimeError('router tag has durable attempts; reconcile receipts before retrying')
        # Unknown expert execution is not a negative teaching label. Skip that
        # entire search row; the complete selection panels remain unchanged.
        examples = []
        for row in self.rows_dev[:self.evolution['router']['task_limit']]:
            if self.shutdown_requested():
                self.status(state='stopped', note='STOP before router proposal')
                return
            outcomes = [S.task_outcome(cell, row['task']) for cell in cells]
            if any(outcome is None for outcome in outcomes):
                continue
            examples.append({'task_id': row['id'],
                'features': [-self.all.demo_error(cell, row) for cell in cells],
                'successful_expert_ids': [cell['id'] for cell, ok in zip(cells, outcomes) if ok]})
        if not examples:
            raise RuntimeError('router has no fully observed search-fitness examples')
        prompt = R.proposal_prompt(ids, incumbent, examples)
        count = R.positions(ids)
        templates = {'router': 'cellforge-router-genome/1'}
        self.state['active_router_wave'] = {'wave': wave, 'stage': 'proposal_inflight',
                                           'incumbent_id': incumbent['hash'], 'expert_ids': ids}
        self.save()
        self.status(stage='router', phase='authoring', round=wave, charged_positions_per_proposal=count)
        try:
            raw = self.worker_pool.propose(prompt, tag, templates,
                lambda template, params: template == templates['router'] and R.validate_params(params, ids), count)
        finally:
            self.state['worker_calls_reserved'] = self.worker_pool.calls
            self.state['makers_calls'] = self._legacy_maker_calls + self.worker_pool.calls
            self.save()
        reservations = self.worker_pool.attempts_for(tag)
        if not reservations:
            self.state.pop('active_router_wave')
            self.save()
            self.status(state='stopped', note='router allowance refused before any reservation')
            return
        entries = {entry['_worker_receipt']['call']: entry for entry in raw}
        task_ids = {'search_fitness': [row['id'] for row in self.rows_dev],
                    'development_selection': [row['id'] for row in self.rows_val]}
        def evaluate(genome):
            outcomes = {}
            for row in self.rows_sel:
                if self.shutdown_requested():
                    raise RuntimeError('router STOP during scoring; reconcile reserved receipts before resume')
                outcomes.update(self.all.score(cells, genome['params'], rows=[row])['outcomes'])
            return {partition: {tid: outcomes[tid] for tid in tids} for partition, tids in task_ids.items()}
        baseline = evaluate(incumbent)
        n = len(ids)
        zero = R.make_genome(ids, {'route.W': [[0] * n for _ in range(n)], 'route.b': [0] * n})
        zero_outcomes = evaluate(zero)
        trials, controls, records = [], [], []
        for reservation in reservations:
            if reservation['templates'] != templates or reservation['charged_positions'] != count:
                raise ValueError('router reservation differs from the frozen roster/position charge')
            call = reservation['call']
            seed = int(EP.digest({'router_wave': wave, 'call': call, 'roster': ids})[:16], 16)
            random_genome = R.make_genome(ids, R.random_params(ids, seed), parents=[incumbent['hash']])
            random_outcomes = evaluate(random_genome)
            controls.append({'genome': random_genome, 'outcomes': random_outcomes})
            entry = entries.get(call)
            candidate, candidate_outcomes = None, None
            outcome = 'invalid_or_missing'
            if entry is not None:
                if entry['task_id'] != 'router':
                    raise ValueError('router worker returned a different task binding')
                candidate = R.make_genome(ids, entry['params'], parents=[incumbent['hash']])
                candidate_outcomes = baseline if candidate['hash'] == incumbent['hash'] else evaluate(candidate)
                trials.append({'genome': candidate, 'outcomes': candidate_outcomes})
                outcome = 'neutral' if candidate['hash'] == incumbent['hash'] else 'valid'
            records.append({'call': call, 'seed': seed, 'charged_positions_per_arm': count,
                'outcome': outcome, 'candidate_id': candidate['hash'] if candidate else None,
                'candidate_params': candidate['params'] if candidate else None,
                'candidate_outcomes': candidate_outcomes, 'random_id': random_genome['hash'],
                'random_params': random_genome['params'], 'random_outcomes': random_outcomes})
        invalid = sum(record['outcome'] == 'invalid_or_missing' for record in records)
        manifest.append_event(self.root, 'router_competition', {
            'wave': wave, 'operator': 'complete_router_replacement', 'partition': 'development',
            'expert_ids': ids, 'features': 'negative-leave-one-out-demo-error/1',
            'baseline_id': incumbent['hash'], 'baseline_params': incumbent['params'],
            'task_ids': task_ids, 'baseline_outcomes': baseline,
            'zero_id': zero['hash'], 'zero_params': zero['params'], 'zero_outcomes': zero_outcomes,
            'attempts': records, 'charged_positions_per_arm': count * len(records),
            'invalid_or_missing': invalid})
        champion = R.select_champion(baseline, trials)
        random_champion = R.select_champion(baseline, controls)
        accepted = None
        if champion is not None:
            score = next(sum(trial['outcomes']['development_selection'].values())
                         for trial in trials if trial['genome']['hash'] == champion['hash'])
            control_ceiling = max([sum(zero_outcomes['development_selection'].values())] +
                                  [sum(trial['outcomes']['development_selection'].values()) for trial in controls])
            if score > control_ceiling:
                accepted = champion
        manifest.append_event(self.root, 'router_champions', {'wave': wave, 'expert_ids': ids,
            'candidate_id': champion['hash'] if champion else None,
            'random_id': random_champion['hash'] if random_champion else None,
            'accepted_id': accepted['hash'] if accepted else None,
            'claim': 'development-only router selection; no independent generalisation claim'})
        # Only a settled journal result may replace the live literal router.
        if accepted is not None:
            self.state['router_genome'] = accepted
        self.state['router_round'] = wave
        self.state.pop('active_router_wave')
        self.save()
        self.measure('router:round%d' % wave, router_id=self.state['router_genome']['hash'],
            router_admitted=int(accepted is not None), router_attempts=len(records),
            invalid_or_missing=invalid, charged_positions_per_arm=count * len(records))

    # -- main loop
    def run(self, max_rounds):
        if self.evolution_active and self.state.get('active_stage') is not None:
            raise RuntimeError('interrupted niche stage requires receipt reconciliation; no automatic paid replay')
        if self.evolution_active and self.state.get('active_wave', {}).get('stage') in ('breeding_inflight', 'mutation_inflight'):
            raise RuntimeError('interrupted wave requires receipt reconciliation; no automatic operator or maker retry')
        if self.evolution_active and self.state.get('active_wave', {}).get('stage') == 'complete':
            completed = self.state['active_wave']
            if self.evolution['curriculum']['enabled']:
                if self.state['curriculum']['wave'] != completed['wave']:
                    raise ValueError('completed wave/phase checkpoint mismatch')
                self.state['curriculum']['wave'] += 1
            self.state.pop('active_wave')
            self.save()
        if self.stop_requested():
            self.status(state='stopped', note='no work started: STOP or worker allowance exhausted')
            return
        self.baseline()
        if self.evolution_active:
            counts = {'dev': sum(self.base['outcomes'][r['id']] for r in self.rows_dev),
                      'val': sum(self.base['outcomes'][r['id']] for r in self.rows_val)}
        else:
            counts = self.split(self.base['outcomes'])
        self.log(dict(stage="baseline", utc=now(), **{"baseline_" + k: v for k, v in counts.items()}))
        queue = [(name, N.LIMITS[name]) for name in self.evolution['niches']] if self.evolution_active else QUEUE
        for name, limit in queue:
            if self.stop_requested():
                break
            if name not in self.state["done"]:
                self.stage(name, limit)
                self.save()
        if self.evolution['router']['enabled']:
            if self.stop_requested():
                self.status(state='stopped', note='stopped before fixed-roster router phase')
                return
            if any(name not in self.state['done'] for name, _ in queue):
                raise RuntimeError('complete configured founder stages before freezing router roster')
            return self.run_router(max_rounds)
        idle = 0
        last = None
        for _ in range(max_rounds):
            if self.stop_requested():
                self.status(state="stopped", note="STOP requested or bounded worker allowance exhausted")
                self.save()
                return
            active = self.state.get('active_wave') if self.evolution_active else None
            if active is None:
                if self.evolution_active:
                    active = {'wave': self.state.get('curriculum', {}).get('wave', self.state['breed_round']),
                              'stage': 'breeding_inflight'}
                    self.state['active_wave'] = active
                    self.save()
                self.status(stage="breeding", phase="running", round=self.state["breed_round"] + 1)
                rec = self.breed()
                if active is not None:
                    active.update(stage='mutation_ready', breed_result=rec)
                    self.save()
            else:
                if active['stage'] not in ('mutation_ready', 'complete'):
                    raise RuntimeError('unknown persisted wave stage')
                rec = active['breed_result']
                if self.evolution['curriculum']['enabled']:
                    curriculum = self.state['curriculum']
                    self._phase = EP.phase_at(self.evolution['curriculum'], curriculum['wave'])
                    by_id = {r['id']: r for r in self.rows_dev}
                    self._wave_rows = [by_id[tid] for tid in curriculum['selected_task_ids']]
            if self.stop_requested():
                self.status(state='stopped', note='stopped before mutation')
                self.save()
                return
            if active is not None and active['stage'] == 'complete':
                rec2 = active['mutation_result']
            else:
                if active is not None:
                    active['stage'] = 'mutation_inflight'
                    self.save()
                self.status(stage="mutation", phase="running", round=self.state.get("mutate_round", 0) + 1)
                rec2 = self.mutate()
                if active is not None:
                    active.update(stage='complete', mutation_result=rec2)
                    self.save()
            progress_score = rec2['arc_dev'] + rec2['arc_val'] if self.evolution_active else rec2['arc_held_out']
            progressed = rec["children_admitted"] or rec2.get("mutations_admitted") or progress_score != last
            idle = 0 if progressed else idle + 1
            last = progress_score
            if self.evolution['curriculum']['enabled']:
                self.state['curriculum']['wave'] += 1
            if active is not None:
                self.state.pop('active_wave')
                self.save()
            if idle >= 3:
                self.status(state="plateau", note="3 rounds of breeding and mutation with no admission and no held-out change; needs new niches")
                return
        self.status(state="rounds_complete")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=pathlib.Path)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed-from", type=pathlib.Path, help="earlier run dir: keep its maker-written cells, drop its chain children")
    ap.add_argument('--evolution-policy', type=pathlib.Path,
                    help='explicit opt-in policy JSON; does not authorize reopening a closed campaign')
    ap.add_argument('--fresh-inputs', type=pathlib.Path,
                    help='explicit hash-bound fresh baseline/local training config; never a legacy seed')
    a = ap.parse_args()
    if a.fresh_inputs and a.seed_from:
        ap.error('fresh inputs and historical --seed-from cannot be combined')
    d = Driver(a.root, a.fake, a.workers, H.load(a.evolution_policy) if a.evolution_policy else None,
               fresh_inputs=H.load(a.fresh_inputs) if a.fresh_inputs else None)
    try:
        if a.seed_from and d.fresh_inputs is not None:
            raise RuntimeError('fresh inputs and historical --seed-from cannot be combined, including on resume')
        if d.phase_context is not None:
            raise RuntimeError('epoch phases require the explicit evolution_epochs entry point')
        if a.seed_from and not d.registry:
            old = H.load(a.seed_from / "REGISTRY.json")
            d.registry = [c for c in old if (c["kind"] == "cell" and c.get("origin", "").startswith("maker")) or c.get("arc_selection_gain")]
            d.state["done"] = sorted({c["niche"] for c in d.registry if c.get("niche") in N.NICHES})
            d.save()
        d.run(a.rounds)
    finally:
        if d.worker_pool is not None:
            d.worker_pool.close()  # prevents new work; does not claim thread cancellation


if __name__ == "__main__":
    main()
