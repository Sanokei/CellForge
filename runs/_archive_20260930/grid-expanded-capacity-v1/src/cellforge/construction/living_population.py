"""Bounded living cohort mechanics from supplied Boolean solve vectors.

No tasks, labels, inference, recombination, optimizer or gradient estimators
exist here. This helper defines population mechanics, not evidence of efficacy.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class Member:
    genotype_hash: str
    age: int
    task_manifest: str
    solves: tuple[bool, ...]


def jaccard_novelty(members):
    """Mean solve-set Jaccard distance to contemporary other genotypes."""
    members=tuple(members); shared_rewards(members)
    sets={m.genotype_hash:{i for i,value in enumerate(m.solves) if value} for m in members}
    novelty={}
    for member in members:
        a=sets[member.genotype_hash]; distances=[]
        for other in members:
            if other.genotype_hash==member.genotype_hash: continue
            b=sets[other.genotype_hash]; union=a|b
            distances.append(1-len(a&b)/len(union) if union else 0.)
        novelty[member.genotype_hash]=sum(distances)/len(distances) if distances else 0.
    return novelty


def shared_rewards(members):
    """One reward per solved task divided among its current living solvers."""
    members=tuple(members)
    if not members: return {}
    if any(not isinstance(m,Member) for m in members): raise ValueError('typed cohort members required')
    n=len(members[0].solves); manifest=members[0].task_manifest
    if not n or len({m.genotype_hash for m in members})!=len(members) or any(len(m.solves)!=n or m.task_manifest!=manifest or any(type(x) is not bool for x in m.solves) for m in members):
        raise ValueError('one equal task manifest and vector length per cohort required')
    counts=[sum(m.solves[i] for m in members) for i in range(n)]
    return {m.genotype_hash:sum(1/counts[i] for i,solved in enumerate(m.solves) if solved) for m in members}


class LivingPopulation:
    """Elites also die at age>=max_age; archive is never a parent pool.

    Admission survivor ranking is cohort shared reward descending, novelty
    descending, age ascending, hash ascending. Rankings are recomputed for
    each admission; sampling uses the current surviving cohort only.
    """
    def __init__(self,*,capacity,max_age,task_manifest,n_tasks,novelty_weight=.25,sampling_floor=.05):
        if type(capacity) is not int or capacity<1 or type(max_age) is not int or max_age<1:
            raise ValueError('positive integer capacity/max_age required')
        if not isinstance(task_manifest,str) or not task_manifest or type(n_tasks) is not int or n_tasks<1:
            raise ValueError('nonempty frozen task manifest and positive task count required')
        if not math.isfinite(novelty_weight) or novelty_weight<0 or not math.isfinite(sampling_floor) or sampling_floor<=0:
            raise ValueError('finite nonnegative novelty weight and positive sampling floor required')
        self.capacity=capacity; self.max_age=max_age; self.task_manifest=task_manifest; self.n_tasks=n_tasks
        self.novelty_weight=novelty_weight; self.sampling_floor=sampling_floor
        self.generation=0; self._living={}; self._archive=[]; self._events=[]

    @property
    def members(self): return tuple(self._living[key] for key in sorted(self._living))

    @property
    def archive(self):
        # Fresh dictionaries prevent external edits of evaluated-history records.
        return tuple(dict(record) for record in self._archive)

    @property
    def events(self): return tuple(dict(record) for record in self._events)

    def _validated(self,evaluation):
        if not isinstance(evaluation,dict): raise ValueError('evaluation dictionary required')
        key=evaluation.get('genotype_hash'); manifest=evaluation.get('task_manifest'); solves=evaluation.get('solves')
        if not isinstance(key,str) or not key: raise ValueError('nonempty genotype hash required')
        if manifest!=self.task_manifest: raise ValueError('frozen cohort task manifest mismatch')
        if not isinstance(solves,(tuple,list)) or len(solves)!=self.n_tasks or any(type(x) is not bool for x in solves):
            raise ValueError('one strict Boolean per frozen task required')
        failed=evaluation.get('failed',False)
        if type(failed) is not bool: raise ValueError('Boolean failed flag required')
        return key,tuple(solves),failed

    def admit(self,evaluations):
        """Archive all valid evaluation records; failed genotypes cannot live.

        Re-evaluating a current genotype preserves its age, never rejuvenates
        an elite. A new externally supplied birth may share an archived hash;
        reproduction and ancestry authorization are outside this helper.
        """
        validated=[self._validated(e) for e in evaluations]
        if len({key for key,_,_ in validated})!=len(validated): raise ValueError('duplicate genotype in admission batch')
        cohort=dict(self._living)
        for key,solves,failed in validated:
            age=cohort[key].age if key in cohort else 0
            self._archive.append({'genotype_hash':key,'age_at_evaluation':age,'task_manifest':self.task_manifest,
                                  'solves':solves,'failed':failed,'generation':self.generation})
            if failed:
                cohort.pop(key,None); self._events.append({'event':'failed','genotype_hash':key,'generation':self.generation})
            else: cohort[key]=Member(key,age,self.task_manifest,solves)
        rewards=shared_rewards(cohort.values()); novelty=jaccard_novelty(cohort.values())
        ranked=sorted(cohort.values(),key=lambda m:(-rewards[m.genotype_hash],-novelty[m.genotype_hash],m.age,m.genotype_hash))
        kept=ranked[:self.capacity]
        for m in ranked[self.capacity:]:
            self._events.append({'event':'capacity_eviction','genotype_hash':m.genotype_hash,'age':m.age,'generation':self.generation})
        self._living={m.genotype_hash:m for m in kept}
        return self.metrics()

    def advance_age(self):
        """A single synchronous generation tick; no elite immortality."""
        self.generation+=1; survivors={}
        for member in self.members:
            age=member.age+1
            if age>=self.max_age:
                self._events.append({'event':'age_eviction','genotype_hash':member.genotype_hash,'age':age,'generation':self.generation})
            else: survivors[member.genotype_hash]=Member(member.genotype_hash,age,member.task_manifest,member.solves)
        self._living=survivors
        return self.metrics()

    def metrics(self):
        members=self.members
        return {'generation':self.generation,'living':len(members),'capacity':self.capacity,'max_age':self.max_age,
                'task_manifest':self.task_manifest,'shared_rewards':shared_rewards(members),'novelty':jaccard_novelty(members),
                'archived_evaluations':len(self._archive),'archive_genotypes':len({e['genotype_hash'] for e in self._archive}),
                'survivor_rule':'shared reward desc, mean Jaccard novelty desc, age asc, hash asc',
                'recombination_implemented':False,'gradient_steps':0,'gradient_estimators':0,
                'limitations':'mechanics only; no experimental efficacy, dominance, task-content or hidden-label access'}

    def sample_parents(self,*,seed,count=2):
        """Seeded sampling with replacement from contemporaries, never archive."""
        if type(seed) is not int or seed<0 or type(count) is not int or count<0: raise ValueError('nonnegative integer seed/count required')
        members=self.members
        if count and not members: raise ValueError('no living contemporary parents')
        rewards=shared_rewards(members); novelty=jaccard_novelty(members)
        weights=[self.sampling_floor+rewards[m.genotype_hash]+self.novelty_weight*novelty[m.genotype_hash] for m in members]
        total=sum(weights)
        probabilities={m.genotype_hash:w/total for m,w in zip(members,weights)} if total else {}
        selected=random.Random(seed).choices(members,weights=weights,k=count) if count else []
        return {'seed':seed,'generation':self.generation,'parent_hashes':[m.genotype_hash for m in selected],
                'probabilities':probabilities,'pool':'living contemporaries only; sampling with replacement'}
