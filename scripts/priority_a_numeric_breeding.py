# SPDX-License-Identifier: MIT-0
"""Opt-in whole-cell numerical inheritance for later real-graph populations.

Not used by the frozen Priority A comparison. Authoring and evaluating a
weight mutation remains the maker's job; this only recombines literal tensors.
"""
import numpy as np
from priority_a_breeding import breed_champion
from expanded_grid_genetics import validate_genome


def breed_numeric_champion(champion,mate,seed):
    validate_genome(champion);validate_genome(mate)
    shared=sorted(mate.params)
    for name in shared:
        if name not in champion.params or champion.params[name].shape!=mate.params[name].shape:
            raise ValueError('homologous numeric tensor shapes required')
        if not np.isfinite(champion.params[name]).all() or not np.isfinite(mate.params[name]).all():
            raise ValueError('finite literal numeric parents required')
    # The strict existing breeder certifies compatible authored structure.
    # This temporary structural witness is not presented as a biological parent.
    witness=champion.copy()
    for name in shared:witness.params[name]=mate.params[name].copy()
    child=breed_champion(witness,mate,seed)
    representative={name:name for name in shared}
    def find(name):
        while representative[name]!=name:
            representative[name]=representative[representative[name]];name=representative[name]
        return name
    for name,definition in mate.defs.items():
        if definition['kind']!='cell':continue
        parameters=sorted(set(mate.params_of(name))&set(shared))
        for param in parameters[1:]:representative[find(param)]=find(parameters[0])
    groups={}
    for name in shared:groups.setdefault(find(name),[]).append(name)
    rng=np.random.default_rng(seed+91000000);provenance=[]
    for names in sorted(groups.values()):
        donor=int(rng.integers(2));parent=(champion,mate)[donor]
        for name in names:child.params[name]=parent.params[name].copy()
        provenance.append(dict(parameters=names,donor=donor,parent_hash=parent.hash()))
    child.meta['expanded_genome'].update(parent_hashes=[champion.hash(),mate.hash()],
        operator='homologous slot and whole-cell tensor crossover',numeric_donors=provenance,
        numerical_inheritance='all parameters tied through shared cells inherit together; new maternal parameters retained')
    validate_genome(child)
    return child
