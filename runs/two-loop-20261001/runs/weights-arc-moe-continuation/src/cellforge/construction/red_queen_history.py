"""Development-only immutable opponent history around paired repair admission."""
import copy
import hashlib
import json
from cellforge.construction.repair_acceptance import repair_admission

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()

class OpponentHistory:
    def __init__(self, records=()):
        self.records=[]
        for record in records:
            self.register(**record)

    def save(self, path):
        from pathlib import Path
        payload={'records':self.records,'history_hash':digest(self.records)}
        Path(path).write_text(json.dumps(payload,sort_keys=True,indent=2))

    @classmethod
    def load(cls, path):
        from pathlib import Path
        payload=json.loads(Path(path).read_text())
        if payload['history_hash']!=digest(payload['records']):
            raise ValueError('opponent history integrity failure')
        return cls(payload['records'])

    def register(self, *, graph_hash, family, structure_hash, provenance, scores):
        if not all((graph_hash,family,structure_hash,provenance)):
            raise ValueError('graph, structural niche and provenance required')
        for label in ('fresh','archive','anchor'):
            section=scores[label]
            if section.get('partition')!='development':
                raise ValueError('development-only opponent history')
            if not section.get('manifest') or not section.get('outcomes'):
                raise ValueError('frozen manifest and nonempty outcomes required')
            if any(type(v) is not bool for v in section['outcomes'].values()):
                raise ValueError('strict boolean outcomes required')
        if any(r['graph_hash']==graph_hash for r in self.records):
            raise ValueError('opponent already recorded')
        self.records.append(copy.deepcopy(dict(graph_hash=graph_hash,family=family,
            structure_hash=structure_hash,provenance=provenance,scores=scores)))

    def admission(self, parent_scores, candidate_scores):
        result=repair_admission(parent_scores,candidate_scores)
        # Every archived predecessor retains its solved archive challenges.
        losses=[]
        candidate=candidate_scores['archive']
        for record in self.records:
            old=record['scores']['archive']
            if (old['manifest']!=candidate['manifest'] or
                old['outcomes'].keys()!=candidate['outcomes'].keys()):
                raise ValueError('history requires one matched archive challenge contract')
            missing=[k for k,v in old['outcomes'].items() if v and not candidate['outcomes'][k]]
            if missing: losses.append({'opponent':record['graph_hash'],'family':record['family'],'tasks':missing})
        if losses:
            result['accepted']=False
            result['reasons'].append({'gate':'historical_opponent_regression','losses':losses})
        result['history_hash']=digest(self.records)
        result['opponent_count']=len(self.records)
        return result
