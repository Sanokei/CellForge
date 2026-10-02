"""Build graph-native authored NCA without benchmark access."""
import argparse,json
from pathlib import Path
from cellforge.construction.nca_gene import build_nca_gene

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--iterations',type=int,default=2); p.add_argument('--seed',choices=('identity','neighbor_majority'),default='identity')
    args=p.parse_args(); graph=build_nca_gene(args.iterations,args.seed); graph.save(args.out)
    print(json.dumps({'hash':graph.hash(),'stats':graph.stats(),'recipe':graph.meta['nca_recipe']}))
if __name__=='__main__': main()
