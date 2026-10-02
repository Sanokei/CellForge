"""Export object genes, literal tensors and typed contracts without data access."""
import argparse
import json
from pathlib import Path

from cellforge.construction.object_gene_bank import build_object_gene_bank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise ValueError('Choose a fresh output directory.')
    graph = build_object_gene_bank()
    digest = graph.save(out)
    receipt = {'hash': digest, 'stats': graph.stats(),
               'genes': graph.meta['port_contracts'],
               'cells': graph.meta['cell_port_contracts'],
               'literal_tensors': {k: v.tolist() for k, v in graph.params.items()},
               'benchmark_tasks_read': 0, 'efficacy_measured': False}
    (out / 'GENE_BANK.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'hash': digest, 'genes': len(receipt['genes']), 'stats': receipt['stats']}))


if __name__ == '__main__':
    main()
