"""New clipped rigid rotation. anchor_spin2 retains its rejection semantics."""
import numpy as np
import skillcore as S
import anchor_skill as A

TEMPLATE,NICHENAME='anchor_spin3','anchorclip'
SHAPES=dict(A.SHAPES,bounds=(2,))
ORACLE_PARAMS=dict(A.ORACLE_PARAMS,bounds=[0,1])
schema_rows=A.schema_rows
generated=A.generated
truth=A.truth
score=A.score


def predict(p,grid,demos=None):
    if int(np.argmax(p['bounds']))==0:return A.predict(p,grid,demos)
    a=np.asarray(grid);margin=max(a.shape)*2;bg=int(np.argmax(p['bg']))
    padded=np.pad(a,margin,constant_values=bg)
    out=np.asarray(A.predict(p,padded.tolist(),demos))
    return out[margin:margin+a.shape[0],margin:margin+a.shape[1]].tolist()


S.TEMPLATES[TEMPLATE]={'shapes':SHAPES,'fn':predict,'aware':False,'doc':S.TEMPLATES[A.TEMPLATE]['doc']+
    ' New bounds[2]: argmax0 rejects out-of-bounds rotations; argmax1 clips rotated source pixels outside the original grid after erasing originals.'}
