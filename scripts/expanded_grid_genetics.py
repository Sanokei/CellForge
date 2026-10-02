# SPDX-License-Identifier: MIT-0
"""Discrete homologous breeding and validated functional identity for seven slots."""
import copy,hashlib,json
import numpy as np
from cellforge.ir.graph import Limits
from cellforge.ir.profiles import check_profile
from cellforge.construction.transactions import MAX_GRID_BANK_GENES


def validate_genome(g):
    if g.meta.get('gradients') is not False:raise ValueError('primary no-gradient contract required')
    record=g.meta['expanded_genome'];bank=record['available_genes'];slots=record['slots'];alleles=record['alleles']
    if slots!=list(range(8,8+len(alleles))) or not 7<=len(alleles)<=9 or not 7<=len(bank)<=MAX_GRID_BANK_GENES or len(set(bank))!=len(bank):raise ValueError('fixed seven-slot contract required')
    for slot,gene in zip(slots,alleles):
        if gene not in bank or g.defs[f'expanded_slot_{slot}']!=g.defs[gene]:raise ValueError('declared allele disagrees with slot')
    g.validate(Limits(strict_microcells=True));check_profile(g,'ladder-micro/1')


def compatible(left,right):
    validate_genome(left);validate_genome(right)
    for key in ('available_genes','slots','donor_hash'):
        if left.meta['expanded_genome'][key]!=right.meta['expanded_genome'][key]:raise ValueError('shared bank contract differs')
    for key in ('grid_interface','grid_attempts','state_dim','profile'):
        if left.meta.get(key)!=right.meta.get(key):raise ValueError('runtime contract differs')
    if left.params.keys()!=right.params.keys() or any(not np.array_equal(left.params[k],right.params[k]) for k in left.params):raise ValueError('immutable parameters differ')
    mutable={f'expanded_slot_{i}' for i in left.meta['expanded_genome']['slots']}
    if left.defs.keys()!=right.defs.keys() or any(left.defs[k]!=right.defs[k] for k in left.defs if k not in mutable):raise ValueError('shared wiring differs')
    if left.entries!=right.entries:raise ValueError('entries differ')


def canonical(base,g):
    compatible(base,g)
    body=dict(base_hash=base.hash(),alleles=g.meta['expanded_genome']['alleles'])
    return hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()


def instantiate(parent,alleles,parent_hashes=None,seed=None):
    validate_genome(parent);alleles=list(alleles);bank=parent.meta['expanded_genome']['available_genes']
    if len(alleles)!=len(parent.meta['expanded_genome']['slots']) or any(x not in bank for x in alleles):raise ValueError('seven declared alleles required')
    child=parent.copy()
    # These fields describe the immediate operator, not inherited history.
    # Historical imports remain in slot_gene_imports with their own hashes.
    for field in ('donor_per_slot','mutated_slot','repaired_slot','old_allele','new_allele','no_change'):
        child.meta['expanded_genome'].pop(field,None)
    for slot,gene in zip(parent.meta['expanded_genome']['slots'],alleles):child.defs[f'expanded_slot_{slot}']=copy.deepcopy(child.defs[gene])
    label='instantiate seven alleles' if len(alleles)==7 else f'instantiate {len(alleles)} alleles'
    child.meta['expanded_genome'].update(alleles=alleles,parent_hashes=list(parent_hashes or [parent.hash()]),operator_seed=seed,operator=label)
    validate_genome(child);return child


def crossover(left,right,seed):
    compatible(left,right);donors=list(map(int,np.random.default_rng(seed).integers(2,size=len(left.meta['expanded_genome']['slots']))))
    choices=[left.meta['expanded_genome']['alleles'],right.meta['expanded_genome']['alleles']]
    child=instantiate(left,[choices[d][i] for i,d in enumerate(donors)],[left.hash(),right.hash()],seed)
    label='homologous uniform seven-slot crossover' if len(donors)==7 else f'homologous uniform {len(donors)}-slot crossover'
    child.meta['expanded_genome'].update(operator=label,donor_per_slot=donors)
    return child


def mutate(parent,seed):
    validate_genome(parent);rng=np.random.default_rng(seed);slot=int(rng.integers(len(parent.meta['expanded_genome']['slots'])))
    bank=parent.meta['expanded_genome']['available_genes'];choice=bank[int(rng.integers(len(bank)))]
    alleles=parent.meta['expanded_genome']['alleles'].copy();old=alleles[slot];alleles[slot]=choice
    child=instantiate(parent,alleles,seed=seed)
    label='one-slot uniform seven-allele mutation' if len(bank)==7 else f'one-slot uniform {len(bank)}-allele mutation'
    child.meta['expanded_genome'].update(operator=label,mutated_slot=slot,old_allele=old,new_allele=choice,no_change=old==choice)
    return child
