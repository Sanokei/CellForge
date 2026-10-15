"""Recover the exact cancelled prompt, failing closed on its recorded SHA256."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / 'PROMPTS-1.json'
data = json.loads(path.read_text())
prefix = ('Do not use tools, run commands, inspect files, search, or write code. Reason from the examples and answer literal tensors only. '
          'A tool invocation invalidates the whole submission. ')
prompt = data['prompt']
original = prompt[len(prefix):] if prompt.startswith(prefix) else prompt
anchor = json.loads((root / 'AUTHOR-1' / 'START.json').read_text())['prompt_sha256']
if hashlib.sha256(original.encode()).hexdigest() != anchor:
    raise RuntimeError('cannot reconstruct original prompt exactly; preserve current files')
(root / 'PROMPTS-1-retry-not-sent.json').write_text(json.dumps(data, indent=2))
data['prompt'] = original
data['recovery'] = 'exact SHA256 match to original AUTHOR-1/START.json; cancelled tool attempt'
path.write_text(json.dumps(data, indent=2))
print('Recovered original prompt with exact SHA256 match')
