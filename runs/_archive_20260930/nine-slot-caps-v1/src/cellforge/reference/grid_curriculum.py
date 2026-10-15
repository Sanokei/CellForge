# SPDX-License-Identifier: MIT-0
"""Original, open synthetic grid tasks for offline evaluation only.

Generator targets never enter inference as query features. No ARC files,
pretrained corpus, benchmark items or network resources are read here.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

PROTOCOL='original-grid-curriculum/1'
SUPPORTED_FAMILIES=('identity','color_permutation','reflect_horizontal','reflect_vertical','transpose','bbox_crop_bg0')
CHALLENGE_FAMILIES=('tiling2x','local_fill')
FAMILIES=SUPPORTED_FAMILIES+CHALLENGE_FAMILIES
DIFFICULTY_BINS={'tiny':(3,5),'small':(6,9)}
PARTITION_BASES={'selection':100000,'validation':900000}
FAMILY_DEFINITIONS={
    'identity':'Copy the full input unchanged.',
    'color_permutation':'One task-wide bijection of colors0..9; all ten source colors appear across demonstrations.',
    'reflect_horizontal':'Reverse column order in every row.',
    'reflect_vertical':'Reverse row order.',
    'transpose':'Exchange row and column axes.',
    'bbox_crop_bg0':'Crop to the smallest bounding rectangle containing every nonzero pixel; background is0.',
    'tiling2x':'Repeat the complete input twice along both axes, without changing individual pixel size.',
    'local_fill':'One simultaneous four-neighbor dilation into zero cells; maximum neighboring color wins ties; nonzero cells persist.'}


def classify_train_families(train):
    """Offline multi-label annotation from demonstrations ONLY, never routing.

    Partial color bijections annotate observed consistency, not identification
    of any unobserved query colors. Overlap and unclassified tasks are valid.
    """
    from .arc_grid_operator import validate_grid
    if not isinstance(train,list) or not train: raise ValueError('nonempty demonstrations required')
    pairs=[]
    for pair in train:
        validate_grid(pair['input']); validate_grid(pair['output'])
        pairs.append((np.asarray(pair['input']),np.asarray(pair['output'])))
    members=[]
    for family in FAMILIES:
        if family=='color_permutation':
            mapping={}; reverse={}; valid=True
            for x,y in pairs:
                if x.shape!=y.shape: valid=False; break
                for a,b in zip(x.flat,y.flat):
                    a,b=int(a),int(b)
                    if (a in mapping and mapping[a]!=b) or (b in reverse and reverse[b]!=a): valid=False; break
                    mapping[a]=b; reverse[b]=a
                if not valid: break
            if valid: members.append(family)
        else:
            try: consistent=all(np.array_equal(_target(x,family),y) for x,y in pairs)
            except ValueError: consistent=False
            if consistent: members.append(family)
    return members


def task_io_fingerprint(record):
    """Exact I/O commitment for cross-split duplicate auditing, evaluator only."""
    return _hash(record['task'])


def _hash(value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()


def _asymmetric(grid):
    return (not np.array_equal(grid,grid[:,::-1]) and
            not np.array_equal(grid,grid[::-1,:]) and
            (grid.shape[0]!=grid.shape[1] or not np.array_equal(grid,grid.T)))


def _random_grid(rng,height,width):
    for _ in range(100):
        grid=rng.integers(0,10,size=(height,width))
        if _asymmetric(grid): return grid
    raise ValueError('failed bounded accidental-symmetry exclusion')


def _target(grid,family,permutation=None):
    """Offline ground-truth construction; never called by grid runtime."""
    if family=='identity': return grid.copy()
    if family=='color_permutation': return permutation[grid]
    if family=='reflect_horizontal': return grid[:,::-1].copy()
    if family=='reflect_vertical': return grid[::-1,:].copy()
    if family=='transpose': return grid.T.copy()
    if family=='bbox_crop_bg0':
        rows,columns=np.where(grid!=0)
        if not len(rows): raise ValueError('crop fixture requires nonbackground pixels')
        return grid[rows.min():rows.max()+1,columns.min():columns.max()+1].copy()
    if family=='tiling2x': return np.tile(grid,(2,2))
    if family=='local_fill':
        # One simultaneous four-neighbor dilation into background0. On ties,
        # the largest neighboring color wins; existing nonzero cells persist.
        output=grid.copy()
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if grid[y,x]==0:
                    output[y,x]=max([int(grid[yy,xx]) for yy,xx in
                        ((y-1,x),(y+1,x),(y,x-1),(y,x+1))
                        if 0<=yy<grid.shape[0] and 0<=xx<grid.shape[1]],default=0)
        return output
    raise ValueError('unknown original family')


def make_grid_task(family,seed,*,difficulty='tiny'):
    """Return three demonstrations and one labeled evaluator-only query."""
    if family not in FAMILIES or difficulty not in DIFFICULTY_BINS:
        raise ValueError('known family and difficulty required')
    if type(seed) is not int or seed<0: raise ValueError('nonnegative integer seed required')
    rng=np.random.Generator(np.random.PCG64(seed)); low,high=DIFFICULTY_BINS[difficulty]
    grids=[]
    for _ in range(4):
        height,width=map(int,rng.integers(low,high+1,size=2))
        if family=='bbox_crop_bg0':
            grid=np.zeros((height,width),dtype=int)
            inner_h=int(rng.integers(1,height)); inner_w=int(rng.integers(1,width))
            y=int(rng.integers(0,height-inner_h+1)); x=int(rng.integers(0,width-inner_w+1))
            grid[y:y+inner_h,x:x+inner_w]=rng.integers(1,10,size=(inner_h,inner_w))
        elif family=='local_fill':
            grid=np.zeros((height,width),dtype=int)
            locations=rng.choice(height*width,size=max(1,height*width//8),replace=False)
            grid.reshape(-1)[locations]=rng.integers(1,10,size=len(locations))
        else: grid=_random_grid(rng,height,width)
        grids.append(grid)
    permutation=None
    if family=='color_permutation':
        permutation=rng.permutation(10)
        while np.array_equal(permutation,np.arange(10)): permutation=rng.permutation(10)
        # Force all ten source colors into the three demonstrations, shuffled
        # over positions; thus the query mapping is fully identifiable.
        positions=[(i,j) for i,g in enumerate(grids[:3]) for j in range(g.size)]
        chosen=rng.choice(len(positions),size=10,replace=False)
        for color,position in enumerate(chosen):
            i,j=positions[int(position)]; grids[i].reshape(-1)[j]=color
    pairs=[{'input':g.tolist(),'output':_target(g,family,permutation).tolist()} for g in grids]
    task={'train':pairs[:3],'test':pairs[3:]}
    provenance={'protocol':PROTOCOL,'family':family,'seed':seed,'difficulty':difficulty,
                'coverage_role':'initial-gene-target' if family in SUPPORTED_FAMILIES else 'challenge',
                'rng':'NumPy PCG64; explicit independent per-task seed','benchmark_data_read':False,
                'source':'original synthetic construction, no corpus','query_target_scope':'offline evaluator only',
                'input_symmetry_exclusion':'bounded horizontal/vertical/square-transpose rejection for random dense families; crop/local-fill exempt',
                'all10colors_in_demonstrations':family=='color_permutation',
                'challenge_note':'challenge label does not assert that every frozen gene fails every generated item'}
    return {'id':_hash({'protocol':PROTOCOL,'family':family,'seed':seed,'difficulty':difficulty}),
            'family':family,'difficulty':difficulty,'seed':seed,'task':task,
            'task_sha256':_hash(task),'train_family_annotations':classify_train_families(task['train']),
            'annotation_scope':'offline demonstrations only; frozen before query scoring; never a runtime feature',
            'provenance':provenance}


def generate_task(family,seed,*,difficulty=None):
    """Operator-compatible task; default bins alternate by independent seed.

    Metadata/annotations are evaluator-only. Predictors receive train and each
    query input through the usual grid interface, not this complete dictionary.
    """
    if difficulty is None: difficulty='tiny' if seed%2==0 else 'small'
    record=make_grid_task(family,seed,difficulty=difficulty)
    task=record['task']
    task['metadata']={'family':family,'seed':seed,'difficulty':difficulty,'task_sha256':record['task_sha256'],
                      'provenance':record['provenance'],'train_family_annotations':record['train_family_annotations'],
                      'annotation_scope':record['annotation_scope'],'family_definition':FAMILY_DEFINITIONS[family]}
    return task


def build_curriculum(partition,*,per_family_bin=25,families=FAMILIES):
    """Fixed disjoint seed ranges; both task content and source are committed."""
    if partition not in PARTITION_BASES or type(per_family_bin) is not int or not 1<=per_family_bin<=1000:
        raise ValueError('fixed selection/validation partition and count1..1000 required')
    if not families or len(set(families))!=len(families) or any(f not in FAMILIES for f in families):
        raise ValueError('unique known families required')
    records=[]
    for family in families:
        for difficulty_index,difficulty in enumerate(DIFFICULTY_BINS):
            for index in range(per_family_bin):
                seed=PARTITION_BASES[partition]+FAMILIES.index(family)*10000+difficulty_index*1000+index
                records.append(make_grid_task(family,seed,difficulty=difficulty))
    manifest={'protocol':PROTOCOL,'partition':partition,'seed_base':PARTITION_BASES[partition],
              'seed_recipe':'base+canonical_family_index*10000+difficulty_index*1000+item_index',
              'per_family_bin':per_family_bin,'families':list(families),'difficulty_bins':DIFFICULTY_BINS,
              'family_definitions':FAMILY_DEFINITIONS,
              'family_schema':{'initial_gene_targets':list(SUPPORTED_FAMILIES),'challenges':list(CHALLENGE_FAMILIES),'count':8,'claim':'target categories, not measured solver coverage'},
              'tasks':len(records),'items':records,'numpy_version':np.__version__,
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'provenance':'original synthetic; no ARC/data/model downloads or reads; targets evaluator-only'}
    manifest['manifest_sha256']=_hash(manifest)
    return manifest


def validate_partition_disjoint(selection,validation):
    if selection.get('partition')!='selection' or validation.get('partition')!='validation':
        raise ValueError('selection and validation manifests required')
    for field in ('id','seed','task_sha256'):
        if {i[field] for i in selection['items']} & {i[field] for i in validation['items']}:
            raise ValueError('partition overlap: '+field)
    for manifest in (selection,validation):
        fingerprints=[task_io_fingerprint(i) for i in manifest['items']]
        if len(set(fingerprints))!=len(fingerprints): raise ValueError('exact I/O duplicates inside partition')
        if any(i['task_sha256']!=task_io_fingerprint(i) for i in manifest['items']): raise ValueError('stale task I/O commitment')
    return True


def runtime_inputs(record):
    """Transport view excludes query target, task IDs, seeds and generator rules."""
    import copy
    return copy.deepcopy(record['task']['train']),copy.deepcopy(record['task']['test'][0]['input'])
