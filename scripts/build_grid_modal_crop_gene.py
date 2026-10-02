# SPDX-License-Identifier: MIT-0
import argparse,copy,json
from build_grid_repeat_gene import build_repeat_gene
from build_grid_bbox_gene import bbox_cluster

def build_modal_crop():
 g=build_repeat_gene(2);bbox_cluster(g,'modal');g.defs['repeat.gene']=copy.deepcopy(g.defs['bbox.geometry'])
 g.meta.update(construction='authored modal-background bounding-box crop/1',provenance='authored hypothesis from existing zero-background limitation; no query-derived fitting',limitations='most frequent input color is background; ties lowest color; empty foreground identity; no semantic object selection')
 g.meta.pop('factor',None);g.validate();return g
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();g=build_modal_crop();print(json.dumps(dict(hash=g.save(a.out),stats=g.stats())))
