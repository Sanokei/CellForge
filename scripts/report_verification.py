"""Fail-closed report consumer for the authoritative bounded-action replay verifier."""
import hashlib
import json
from pathlib import Path

ACTION_RUN = 'weights-action-arguments-v1'
POPULATION_RUN = 'weights-action-population-corrected-g3'
RECEIPT_SCHEMA = 'cellforge-run-verification/1'
REPLAY_SCHEMA = 'cellforge-verified-action-trial/1'
BOUNDED_LABEL = 'bounded lexical argument capability trial; provisional and unadmitted'


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_gate(directory):
    directory = Path(directory)
    if directory.name == POPULATION_RUN:
        return read_population_gate(directory)
    gate = {'run': directory.name, 'scientific_headline_allowed': False,
            'scientific_acceptance': False, 'mechanism_replay_passed': False,
            'status': 'unverified', 'errors': [], 'receipt': None,
            'authoritative_replay': None}
    path = directory / 'VERIFICATION.json'
    if not path.exists():
        gate['errors'].append('missing VERIFICATION.json')
    else:
        try:
            receipt = load_json(path)
            if not isinstance(receipt, dict) or receipt.get('schema') != RECEIPT_SCHEMA:
                raise ValueError('unsupported verification receipt schema')
            gate['receipt'] = receipt
            integrity = load_json(directory / 'INTEGRITY.json')
            if integrity.get('VERIFICATION.json') != hashlib.sha256(path.read_bytes()).hexdigest():
                gate['errors'].append('stale_or_tampered_verification_receipt')
            if receipt.get('scientific_standard', {}).get('acceptance') is not True:
                gate['errors'].append('scientific_standard acceptance missing or false')
            if receipt.get('report_policy', {}).get('scientific_headline_allowed') is not True:
                gate['errors'].append('report_policy scientific headline missing or false')
            if receipt.get('attribution', {}).get('status') != 'passed':
                gate['errors'].append('attribution gate missing or failed')
        except (OSError, ValueError, TypeError, AttributeError) as error:
            gate['errors'].append('missing_or_invalid_receipt: ' + str(error))
    if directory.name != ACTION_RUN:
        gate['status'] = 'unsupported_unverified'
        gate['errors'].append('authoritative replay unsupported for this run; topology/lineage/population gates not verified')
        return gate
    gate['label'] = BOUNDED_LABEL
    try:
        from cellforge.construction.action_verification import verify
        replay = verify(directory)
        gate['authoritative_replay'] = replay
        if not isinstance(replay, dict) or replay.get('schema') != REPLAY_SCHEMA:
            raise ValueError('unsupported authoritative replay schema')
        mechanism = replay.get('mechanism_replay', {})
        if mechanism.get('passed') is not True or mechanism.get('errors') != []:
            gate['errors'].append('authoritative mechanism replay failed')
            gate['errors'].extend(str(x) for x in mechanism.get('errors', []))
        else:
            gate['mechanism_replay_passed'] = True
        if replay.get('scientific_acceptance') is not True:
            gate['errors'].append('authoritative verifier rejects scientific acceptance')
        if replay.get('scientific_headline_allowed') is not True:
            gate['errors'].append('authoritative verifier rejects scientific headline')
        # Receipt claims cannot override independently recomputed policy.
        receipt = gate['receipt'] or {}
        if receipt.get('scientific_standard', {}).get('acceptance') is True and replay.get('scientific_acceptance') is not True:
            gate['errors'].append('receipt acceptance conflicts with authoritative verifier')
    except Exception as error:
        # A broken/unavailable verifier cannot authorize a report headline.
        gate['errors'].append('authoritative verifier unavailable or invalid: ' + str(error))
    gate['scientific_acceptance'] = not gate['errors']
    gate['scientific_headline_allowed'] = not gate['errors']
    gate['status'] = 'accepted' if not gate['errors'] else 'scientific_acceptance_blocked'
    return gate


def component_metrics(directory, gate):
    """Keep recorded bounded metrics even when scientific acceptance is denied."""
    directory = Path(directory)
    if directory.name != ACTION_RUN:
        return None
    try:
        result = load_json(directory / 'RESULT.json')
        metrics = []
        for candidate in result['results']:
            # No normalization into ARC exact-task learning or maker replicates.
            metrics.append({'arm': candidate['arm'], 'seed': candidate['seed'],
                            'scores': candidate['scores']})
        return {'label': BOUNDED_LABEL, 'metrics': metrics,
                'metric_status': 'independently_replayed' if gate['mechanism_replay_passed'] else 'recorded_unverified',
                'scientific_headline_allowed': gate['scientific_headline_allowed']}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {'label': BOUNDED_LABEL, 'metrics': [], 'metric_status': 'missing_or_invalid',
                'error': str(error), 'scientific_headline_allowed': False}


def read_population_gate(directory):
    """Population replay and fixture quality are separate, mandatory report gates."""
    directory=Path(directory)
    gate={'run':directory.name,'status':'scientific_acceptance_blocked',
          'scientific_acceptance':False,'scientific_headline_allowed':False,
          'runtime_acceptance':False,'mechanism_replay_passed':False,
          'population_replay_passed':False,'evidence_quality_passed':False,
          'quality_receipt_valid':False,'verification_receipt_valid':False,'component_metrics_allowed':False,
          'errors':[],'receipt':None,'quality_receipt':None,'authoritative_replay':None}
    try:
        from cellforge.construction.action_verification import verify
        replay=verify(directory);gate['authoritative_replay']=replay
        if replay.get('schema')!=REPLAY_SCHEMA or replay.get('kind')!='independent-population':
            raise ValueError('unsupported independent-population verifier response')
        gate['population_replay_passed']=replay.get('population_replay',{}).get('passed') is True
        gate['mechanism_replay_passed']=replay.get('mechanism_replay',{}).get('passed') is True
        gate['evidence_quality_passed']=replay.get('evidence_quality',{}).get('passed') is True
        gate['errors'].extend(str(x) for x in replay.get('mechanism_replay',{}).get('errors',[]))
        for filename,key in [('VERIFICATION.json','receipt'),('QUALITY_VERIFICATION.json','quality_receipt')]:
            try:
                receipt=load_json(directory/filename);gate[key]=receipt
                if receipt.get('schema')!=REPLAY_SCHEMA or receipt.get('kind')!='independent-population':
                    raise ValueError('unsupported population receipt schema')
                if receipt.get('artifact_binding')!=replay.get('artifact_binding'):
                    raise ValueError('receipt artifact binding differs from independently checked evidence')
                if key=='quality_receipt':
                    if receipt.get('evidence_quality')!=replay.get('evidence_quality'):
                        raise ValueError('quality status differs from independent fixture verification')
                    gate['quality_receipt_valid']=True
                elif receipt.get('scientific_acceptance') is not True and receipt.get('scientific_headline_allowed') is not True:
                    gate['verification_receipt_valid']=True
                if receipt.get('scientific_acceptance') is True and replay.get('scientific_acceptance') is not True:
                    gate['errors'].append(filename+': fabricated scientific acceptance')
                if receipt.get('scientific_headline_allowed') is True and replay.get('scientific_headline_allowed') is not True:
                    gate['errors'].append(filename+': fabricated scientific headline')
            except (OSError,ValueError,TypeError,AttributeError) as error:
                gate['errors'].append(filename+': missing_stale_or_tampered_receipt: '+str(error))
        if not gate['evidence_quality_passed']:
            gate['errors'].append('fixture_quality_failed; no valid accuracy may be claimed')
        if not gate['quality_receipt_valid']:
            gate['errors'].append('quality receipt missing, stale or tampered; metric validity blocked')
        gate['component_metrics_allowed']=(gate['mechanism_replay_passed'] and gate['population_replay_passed']
                                         and gate['evidence_quality_passed'] and gate['quality_receipt_valid'] and gate['verification_receipt_valid'])
        gate['runtime_acceptance']=gate['component_metrics_allowed'] and replay.get('runtime_acceptance') is True
        gate['errors'].append('authoritative population verifier rejects full scientific acceptance')
    except Exception as error:
        gate['errors'].append('authoritative population verifier unavailable or invalid: '+str(error))
    return gate


def population_metrics(directory,gate):
    if Path(directory).name!=POPULATION_RUN:return None
    try:
        result=load_json(Path(directory)/'RESULT.json')
        rows=[{'seed':context['seed'],'raw_recorded_correct':context['report_correct'],
               'raw_recorded_case_count':len(context['reporting'])} for context in result['contexts']]
        valid=gate['component_metrics_allowed']
        return {'label':'bounded independent-population fixture diagnostics; scientific acceptance blocked',
                'metric_status':'quality_verified_component' if valid else 'invalid_or_unverified_fixture_raw_scores_only',
                'raw_scores':rows,'valid_accuracy':None if not valid else [r['raw_recorded_correct']/r['raw_recorded_case_count'] for r in rows],
                'quality_errors':gate.get('authoritative_replay',{}).get('evidence_quality',{}).get('errors',[]) if gate.get('authoritative_replay') else gate['errors'],
                'population_replay_passed':gate['population_replay_passed'],
                'evidence_quality_passed':gate['evidence_quality_passed'],
                'scientific_headline_allowed':False}
    except (OSError,ValueError,KeyError,TypeError) as error:
        return {'metric_status':'missing_or_invalid','raw_scores':[],'valid_accuracy':None,'error':str(error),'scientific_headline_allowed':False}
