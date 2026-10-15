"""Frozen scalar local packet metric; not a BFCL leaderboard-equivalent matcher."""
import native_adapter as A
METRIC='restricted_local_packet_accuracy'
def score(packet,case,annotation):
    result=A.check_against_public_annotation(packet,case,annotation)
    return {'correct':bool(result['joint_exact_match']),'schema_valid':bool(result['schema_valid']),'metric':METRIC,'reason':result.get('reason')}