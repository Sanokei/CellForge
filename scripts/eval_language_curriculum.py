"""Run a fixed selection curriculum; private report data belongs to operator."""
import argparse
import json
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.reference.language_curriculum import evaluate_language_curriculum


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--items',required=True);p.add_argument('--out',required=True)
    p.add_argument('--max-new-tokens',type=int,default=24);p.add_argument('--aggregate-only',action='store_true')
    args=p.parse_args();out=Path(args.out)
    if out.exists():raise SystemExit('preserve existing receipt')
    manifest=json.loads(Path(args.items).read_text(encoding='utf-8'))
    if manifest.get('partition')!='selection':raise SystemExit('this author CLI accepts selection only; report items require operator frozen-winner evaluation')
    report=evaluate_language_curriculum(Graph.load(args.model),manifest['items'],max_new_tokens=args.max_new_tokens,include_raw=not args.aggregate_only)
    report['partition']='selection';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='items'},indent=2))


if __name__=='__main__':main()
