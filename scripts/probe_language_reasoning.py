"""Matched author-visible development probe; no sealed evaluation access."""
import argparse
import hashlib
import json
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.runtime.lm import generate, score_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    raw = (root / 'data/tinyreal/dev_eval.json').read_bytes()
    texts = [s[:128] for s in json.loads(raw)[:3]]
    # Predeclared novel assignments: exact copying from context is necessary.
    cases = [(f'Remember: {key} means {value}. What does {key} mean? Answer:', value)
             for key, value in [('miv', 'red'), ('zop', 'blue'), ('nel', 'green'), ('tav', 'gold')]]
    comparisons = {}
    for label, path in [('parent', 'runs/unified-byte/artifact'), ('memory', 'runs/unified-byte-memory/artifact')]:
        graph = Graph.load(root / path)
        scores, _ = score_texts(graph, texts)
        outputs, info = generate(graph, len(cases), max_len=12, prompts=[p for p, _ in cases])
        comparisons[label] = {'model_hash': graph.hash(), 'bits_per_byte': float(scores['bits'].sum() / scores['n_tokens'].sum()),
                             'correct_tokens': int(scores['correct'].sum()), 'tokens': int(scores['n_tokens'].sum()),
                             'failures': int(scores['failed'].sum()),
                             'reasoning_exact': sum(o.strip() == a for o, (_, a) in zip(outputs, cases)),
                             'reasoning_cases': [{'prompt': p, 'expected': a, 'output': o, 'failure': f}
                                                 for (p, a), o, f in zip(cases, outputs, info['failure'])]}
    report = {'protocol': 'language-memory-development/1', 'source_sha256': hashlib.sha256(raw).hexdigest(),
              'selection': 'first three development texts truncated to 128 characters', 'comparisons': comparisons,
              'scope': 'author-visible bounded development; exact novel-binding probes are smoke evidence, not independent held-out benchmark',
              'numeric_provenance': 'single directly authored strength 2; no fitted weights or parameter sweep',
              'decision': 'retain mutation as experimental only; no understanding or frontier-superiority claim'}
    target = Path(args.out)
    if target.exists():
        raise SystemExit('preserve existing receipt')
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
