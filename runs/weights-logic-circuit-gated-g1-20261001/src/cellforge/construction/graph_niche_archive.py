"""Finite living CellForge graph population with literal maker mutations.

Offline proposer signature: proposer(graph, train_diagnostics, rng) -> literal
DirectWeightMaker proposal. No toy interpreter, model service, or optimizer.
The evaluator supplies the full paired development/opponent/native protocol.
"""
from dataclasses import dataclass
from pathlib import Path
import copy
import math
import numpy as np
from .direct_weight_maker import DirectWeightMaker
from .repair_acceptance import repair_admission
from ..ir.graph import Graph


@dataclass
class LivingGraph:
    graph: object
    metrics: dict
    niche: str
    born: int
    parent_hash: str | None = None
    parents: tuple = ()


@dataclass(frozen=True)
class ArchivedGraph:
    path: Path
    niche: str
    born: int
    parents: tuple


class GraphNicheArchive:
    def __init__(self, ledger, *, budget, capacity=8, lifespan=8, max_parameters=1024):
        if capacity<2 or lifespan<1:raise ValueError('finite population and positive lifespan required')
        self.maker=DirectWeightMaker(ledger,budget=budget,max_parameters=max_parameters)
        if self.maker.operator.rows:raise ValueError('population resumption requires restored ages and living state; do not silently reseed')
        self.archive_root=Path(ledger).with_name(Path(ledger).name+'-graphs')
        self.capacity=capacity;self.lifespan=lifespan;self.max_parameters=max_parameters
        self.tick=0;self.living={};self.archive={}

    def seed(self, graph, metrics, *, niche):
        if graph.meta.get('gradients') is not False:raise ValueError('explicit no-gradient provenance required')
        graph.validate()
        if graph.stats()['unique_param_storage']>self.max_parameters:raise ValueError('graph exceeds device cap')
        if not niche:raise ValueError('named niche required')
        # Self-comparison checks manifests, per-task populations, FP and caps.
        verdict=repair_admission(metrics,metrics)
        if verdict['reasons']!=[{'gate':'strict_fresh_gain'}]:
            raise ValueError('seed requires complete defined admission evidence')
        if not metrics.get('fixtures_passed') or not metrics.get('caps_passed'):
            raise ValueError('seed requires evaluated fixtures and caps')
        for label in ('fresh','archive','anchor'):
            section=metrics.get(label,{})
            if section.get('partition')!='development' or not section.get('manifest') or not section.get('outcomes'):
                raise ValueError('paired development metrics required')
            if not math.isfinite(section.get('false_positive_given_consistency',float('nan'))):
                raise ValueError('defined false-positive rate required')
        h=graph.hash()
        if h in self.archive:raise ValueError('duplicate seed cannot reset age')
        for member in self.living.values():
            for label in ('fresh','archive','anchor'):
                a,b=member.metrics[label],metrics[label]
                if (a['manifest'],a.get('source'),set(a['outcomes']))!=(b['manifest'],b.get('source'),set(b['outcomes'])):
                    raise ValueError('population food sources must be exactly paired')
        member=LivingGraph(graph.copy(),copy.deepcopy(metrics),niche,self.tick)
        self.living[h]=member;self._retain(member);self._prune()

    def _retain(self, member):
        """Keep donor tensors on disk; only the finite living set stays in RAM."""
        h=member.graph.hash();path=self.archive_root/h
        if path.exists():
            if Graph.load(path).hash()!=h:raise ValueError('archived donor hash mismatch')
        else:member.graph.save(path)
        self.archive[h]=ArchivedGraph(path,member.niche,member.born,member.parents)

    def donor(self, graph_hash):
        graph=Graph.load(self.archive[graph_hash].path)
        if graph.hash()!=graph_hash:raise ValueError('archived donor hash mismatch')
        return graph

    def shared_rewards(self):
        """Each solved task is one food unit split among its living solvers."""
        coverage={}
        for h,m in self.living.items():
            for task,solved in m.metrics['fresh']['outcomes'].items():
                if solved:coverage.setdefault(task,[]).append(h)
        return {h:sum(1/len(solvers) for solvers in coverage.values() if h in solvers) for h in self.living}

    def _prune(self):
        self.living={h:m for h,m in self.living.items() if self.tick-m.born<self.lifespan}
        while len(self.living)>self.capacity:
            rewards=self.shared_rewards()
            victim=min(self.living,key=lambda h:(rewards[h],self.living[h].born,h))
            del self.living[victim]

    def mutate(self, parent_hash, proposer, *, diagnostics, fixtures, evaluate, rng):
        if parent_hash not in self.living:raise ValueError('parent must be a living contemporary')
        parent=self.living[parent_hash]
        if diagnostics.get('graph_hash')!=parent_hash:raise ValueError('exact parent diagnostics required')
        return self._submit(parent.graph,parent,proposer,diagnostics=diagnostics,fixtures=fixtures,
            evaluate=evaluate,rng=rng,parents=(parent_hash,))

    def breed_and_mutate(self, parent_hash, mate_hash, breeder, proposer, *,
                         diagnostics_for, fixtures_for, evaluate, rng, breeding_seed):
        """One charged offspring from two living contemporaries, then maker.

        The unselected bred graph is never admitted as a living individual.
        Its literal edit is evaluated against the target-niche parent's full
        paired protocol. Diagnostic inference must be accounted by the caller.
        """
        if parent_hash==mate_hash or any(h not in self.living for h in (parent_hash,mate_hash)):
            raise ValueError('two distinct living contemporaries required')
        parent=self.living[parent_hash];mate=self.living[mate_hash]
        try:
            base=breeder(parent.graph.copy(),mate.graph.copy(),breeding_seed)
            if base.meta.get('gradients') is not False:raise ValueError('no-gradient offspring required')
            if base.meta.get('expanded_genome',{}).get('parent_hashes')!=[parent_hash,mate_hash]:
                raise ValueError('exact two-parent offspring lineage required')
            base.validate()
            diagnostics=diagnostics_for(base.copy())
            if diagnostics.get('graph_hash')!=base.hash():raise ValueError('exact offspring diagnostics required')
            fixtures=fixtures_for(base.copy())
        except Exception as exc:
            # A broken reproduction consumes a proposal, without scorer calls.
            diagnostics=dict(partition='development',evidence='train-pairs',graph_hash=parent_hash,
                reproduction_error=f'{type(exc).__name__}: {exc}')
            return self._submit(parent.graph,parent,lambda *args:{},diagnostics=diagnostics,
                fixtures=[],evaluate=None,rng=rng,parents=(parent_hash,mate_hash))
        diagnostics=copy.deepcopy(diagnostics)
        diagnostics['reproduction']=dict(parent_hashes=[parent_hash,mate_hash],bred_hash=base.hash(),
            breeding_seed=breeding_seed,operator='sexual recombination followed by literal maker mutation')
        return self._submit(base,parent,proposer,diagnostics=diagnostics,fixtures=fixtures,
            evaluate=evaluate,rng=rng,parents=(parent_hash,mate_hash))

    def _submit(self, base, parent, proposer, *, diagnostics, fixtures, evaluate, rng, parents):
        before=len(self.maker.operator.rows)
        try:proposal=proposer(base.copy(),copy.deepcopy(diagnostics),rng)
        except Exception as exc:proposal=dict(rationale=f'proposer failed: {type(exc).__name__}')
        result=self.maker.submit(base,proposal if isinstance(proposal,dict) else {},
            diagnostics=diagnostics,fixtures=fixtures,evaluate=evaluate,admission_parent=parent.metrics)
        if len(self.maker.operator.rows)!=before+1:raise ValueError('one proposal must consume one attempt')
        self.tick+=1
        candidate=result['candidate'];receipt=result['receipt']
        if candidate is not None:
            h=candidate.hash()
            # Failed donors remain immutable crossover material; accepted
            # offspring alone enter the living population. Duplicate hashes
            # never gain a second life or reset mortality.
            if h not in self.archive:
                member=LivingGraph(candidate.copy(),copy.deepcopy(receipt.get('development_metrics',{})),parent.niche,self.tick,parents[0],parents)
                self._retain(member)
                if receipt['status']=='accepted':self.living[h]=member
        self._prune()
        return result

    def sample_parent(self, rng, *, niche=None):
        eligible=sorted(h for h,m in self.living.items() if niche is None or m.niche==niche)
        if not eligible:raise ValueError('niche extinct; requires an explicitly evaluated new seed')
        rewards=self.shared_rewards()
        weights=np.asarray([1+rewards[h] for h in eligible]);weights/=weights.sum()
        return eligible[int(rng.choice(len(eligible),p=weights))]
