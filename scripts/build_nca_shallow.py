"""Separate authored shallow artifact; no benchmark access or measurement."""
import argparse,json
from cellforge.construction.nca_shallow import build_shallow_nca

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--seed',choices=('identity','swap01'),default='identity'); p.add_argument('--architecture',choices=('residual','direct'),default='residual')
    args=p.parse_args(); graph=build_shallow_nca(args.seed,args.architecture); graph.save(args.out)
    print(json.dumps({'hash':graph.hash(),'stats':graph.stats(),'limitations':graph.meta['limitations']}))
if __name__=='__main__':main()
