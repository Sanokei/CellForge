# SPDX-License-Identifier: MIT-0
"""Breed an evolved champion with an unchanged compatible archive mate.

The champion owns newly authored cells and their weights. Homologous slots
choose either parent's gene; original mate genes must retain exact semantics.
This is a two-parent repair protocol, not a large biological population.
"""
import numpy as np
from expanded_grid_genetics import instantiate,validate_genome


def breed_champion(champion,mate,seed):
    validate_genome(champion);validate_genome(mate)
    slots=champion.meta['expanded_genome']['slots']
    if slots!=mate.meta['expanded_genome']['slots'] or champion.entries!=mate.entries:
        raise ValueError('archive mate runtime/slot interface differs')
    for key in ('state_dim','grid_interface','grid_attempts','profile'):
        if champion.meta.get(key)!=mate.meta.get(key):raise ValueError('archive mate runtime differs')
    if not set(mate.meta['expanded_genome']['available_genes'])<=set(champion.meta['expanded_genome']['available_genes']):
        raise ValueError('archive mate genes unavailable in champion')
    mutable={f'expanded_slot_{slot}' for slot in slots}
    ported=set(champion.meta['expanded_genome']['available_genes'])|mutable
    for name,definition in mate.defs.items():
        if name in mutable or champion.defs.get(name)==definition:continue
        current=champion.defs.get(name)
        # Maternal context wiring is inherited as a blueprint trait. Old
        # primitive bodies may receive an unused mode port without changing
        # their numerical semantics. Certify the exact transformation only.
        if 'context_mode_gate' in champion.defs and current is not None:
            if name==mate.entries['grid_step'] or name.startswith('expert_'):
                from demonstration_mode_blueprint import controller_with_mode
                if current==controller_with_mode(definition,ported):continue
            if name in ported and current.get('inputs')==definition.get('inputs',[])+['mode']:
                import copy
                stripped=copy.deepcopy(current);stripped['inputs']=definition['inputs']
                if stripped==definition:continue
        raise ValueError('archive mate shared cell/wiring semantics differ')
    for name,value in mate.params.items():
        if name not in champion.params or not np.array_equal(champion.params[name],value):
            raise ValueError('archive mate shared numerical semantics differ')
    donors=np.random.default_rng(seed).integers(0,2,size=len(slots)).tolist()
    choices=[champion.meta['expanded_genome']['alleles'],mate.meta['expanded_genome']['alleles']]
    child=instantiate(champion,[choices[d][i] for i,d in enumerate(donors)],
        parent_hashes=[champion.hash(),mate.hash()],seed=seed)
    child.meta['expanded_genome'].update(operator='homologous champion/archive-mate crossover',donor_per_slot=donors)
    return child
