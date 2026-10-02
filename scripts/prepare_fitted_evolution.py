"""Operator-only fresh fit/development/report split and fitted seed export."""
import argparse
import hashlib
import json
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.construction.fitted_byte_seed import fitted_byte_seed
from cellforge.governance.ledger import Ledger


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def context_items(batch):
    values = ('ZK7', 'FN4') if batch == 'a' else ('MV2', 'QH8')
    items = []
    for j, value in enumerate(values):
        filler = ' quiet room ' * (2 if j == 0 else 8)
        prefix = f'key={value}; key={value};' + filler + 'key='
        size = len(prefix.encode())
        text = prefix + value
        items.append({'id': f'context-{batch}-{j}', 'kind': 'next_byte', 'family': f'authored-context-{batch}-{j}',
                      'text': text, 'score_byte_ranges': [[size, size+3]],
                      'dependency_distance_bytes': size-4,
                      'scope': 'authored repeated-key continuation, not ordinary QA'})
    return {'partition': 'selection', 'items': items, 'scope': 'conditional target-byte loss; full prefix/state retained'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True); p.add_argument('--founder', required=True)
    p.add_argument('--exclude-private', required=True); p.add_argument('--private', required=True)
    p.add_argument('--public', required=True)
    args = p.parse_args()
    private, public = Path(args.private).resolve(), Path(args.public)
    if private.parent != Path('/srv/cellforge-private') or private.exists() or public.exists():
        raise SystemExit('fresh operator-private partition and public export required')
    excluded = set()
    for name in ('selection.json', 'report-only.json'):
        excluded.update(x['family'] for x in json.loads((Path(args.exclude_private)/name).read_text()))
    raw = Path(args.source).read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    unique = {hashlib.sha256(t.encode()).hexdigest(): t for t in json.loads(raw)}
    ordered = [h for h in sorted(unique) if h not in excluded]
    if len(ordered) < 518: raise SystemExit('insufficient whole-family source')
    fit = {'partition': 'dev_train_selection', 'corpus_sha256': source_sha,
           'source_split': 'dev_train', 'items': [{'family': h, 'text': unique[h]} for h in ordered[:512]]}
    private.mkdir(mode=0o700); public.mkdir()
    write(private/'fit-selection.json', fit)
    hashes = {}
    for label, start in [('a', 512), ('b', 514), ('report', 516)]:
        manifest = {'partition': 'report' if label == 'report' else 'selection',
                    'corpus_sha256': source_sha,
                    'items': [{'id': h, 'family': h, 'kind': 'next_byte', 'text': unique[h][:64]} for h in ordered[start:start+2]],
                    'window_policy': 'first64UTF8bytes of ASCII whole-family stories; fresh state at window start'}
        assert all(x['text'].isascii() and len(x['text'].encode()) == 64 for x in manifest['items'])
        path = private/f'{label}.json'; write(path, manifest)
        hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    for label in ('a', 'b'): write(private/f'context-{label}.json', context_items(label))
    founder = Graph.load(args.founder)
    seed, provenance = fitted_byte_seed(founder, fit)
    digest = seed.save(public/'fitted-seed')
    policy = {'protocol': 'fitted-evolution-partition/1', 'source_sha256': source_sha,
              'fit_file_sha256': hashlib.sha256((private/'fit-selection.json').read_bytes()).hexdigest(),
              'dev_report_file_hashes': hashes, 'fitted_seed_hash': digest, 'founder_hash': founder.hash(),
              'partition_method': 'sorted exact-whole-text SHA256 families; first512 fit, next2 devA, next2 devB, next2 report, prior pilot families excluded',
              'fit_provenance': provenance, 'report_access_before_freeze': False,
              'limitations': 'public dev_train source; exact duplicates excluded, near-family overlap unaudited; pilot report is not unexposed final',
              'context_fitness': 'two disjoint authored cases per batch, target3bytes only; >8 and >64 dependency distances, no QA claim',
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(private/'policy.json', policy); write(public/'policy.json', policy)
    Ledger(public/'ledger.jsonl').append('count_fitted_seed', policy, 'operator')
    print(json.dumps({'fitted_seed_hash': digest, 'fit_documents': provenance['fit_documents'],
                      'fit_bytes': provenance['fit_content_bytes'], 'unique_parameters': seed.stats()['unique_param_storage']}))


if __name__ == '__main__': main()
