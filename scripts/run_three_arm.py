"""File-supervised paired gene repair study (seeds 11,22,33).

Run again after writing inbox/<seed>/round<k>.json. No model API is called.
Custom adapters use module:function and support genes from any session/domain.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cellforge.ir.graph import Graph
from cellforge.construction.niche_weight_bridge import run_comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gene', action='append', required=True, help='graph artifact directory; repeat for every gene')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--adapter', required=True, help='Python module:function providing generated task contexts')
    parser.add_argument('--rounds', type=int, default=4)
    args = parser.parse_args()
    module, function = args.adapter.split(':', 1)
    adapter = getattr(importlib.import_module(module), function)
    genes = {}
    for path in args.gene:
        graph = Graph.load(Path(path))
        # Hash ids deduplicate identical genes without depending on session names.
        genes[graph.hash()] = graph
    print(json.dumps(run_comparisons(args.out, genes, adapter, rounds=args.rounds,
                                     adapter_id=args.adapter), indent=2))


if __name__ == '__main__':
    main()
