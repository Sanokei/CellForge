"""Offline fixtures matching parent-reported CLI envelopes, not a live audit.

No invocation, reservation, tensor proposal, secret, or private reasoning text.
The projection mirrors the frozen audit encoding specification; sequence is a
host wrapper line index and hashes cover original bytes including terminators.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import nca_author_intake as A


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def native_stream(text='  {"fixture":"no scientific candidate"}\n'):
    events = [
        {'type': 'thread.started', 'thread_id': 'fixture-only'},
        {'type': 'turn.started'},
        {'type': 'item.completed', 'item': {'id': 'fixture-item', 'type': 'agent_message', 'text': text}},
        {'type': 'turn.completed', 'usage': {'input_tokens': 100, 'cached_input_tokens': 20,
                                           'cache_write_input_tokens': 5, 'output_tokens': 30,
                                           'reasoning_output_tokens': 10}},
    ]
    return b''.join(json.dumps(event, ensure_ascii=False).encode('utf-8') + b'\n' for event in events)


def project(raw):
    metadata, final_bytes, usage = [], None, None
    for index, line in enumerate(raw.splitlines(keepends=True)):
        event = A.strict_json(line)
        validate_reported_envelope(event)
        item = event.get('item', {})
        event_type, item_type = event['type'], item.get('type')
        classification, final_hash = 'unknown', None
        if event_type in {'thread.started', 'turn.started', 'turn.completed'}:
            classification = 'lifecycle'
        elif item_type == 'reasoning':
            classification = 'reasoning_summary'
        elif event_type == 'item.completed' and item_type == 'agent_message':
            classification = 'final_message'
            final_bytes = item['text'].encode('utf-8')
            final_hash = sha(final_bytes)
        if event_type == 'turn.completed':
            # Preserve every native counter. Existing intake uses a deliberately
            # narrower host field; the separate native record is not discarded.
            usage = {'native': copy.deepcopy(event['usage']),
                     'host': {**{key: event['usage'][key] for key in
                                 ('input_tokens', 'cached_input_tokens', 'output_tokens')},
                              'credits': 'unknown'}}
        metadata.append({'sequence': index, 'event_type': event_type, 'item_type': item_type,
                         'classification': classification, 'event_sha256': sha(line),
                         'final_raw_sha256': final_hash})
    return metadata, final_bytes, usage


def validate_reported_envelope(event):
    """Exact observed four-envelope fixture schema, before any projection.

    The parent prospectively verified item.completed reasoning with the same
    id/type/text shape. Other forms need a new allowlist, not silent projection.
    This does not claim live wrapper coverage.
    """
    if type(event) is not dict:
        raise ValueError('event object required')
    kind = event.get('type')
    keys = {'thread.started': {'type', 'thread_id'}, 'turn.started': {'type'},
            'item.completed': {'type', 'item'}, 'turn.completed': {'type', 'usage'}}
    if kind not in keys or set(event) != keys[kind]:
        raise ValueError('unverified event type or extra envelope field')
    if kind == 'thread.started':
        if type(event['thread_id']) is not str or not event['thread_id']:
            raise ValueError('thread identifier required')
    elif kind == 'item.completed':
        item = event['item']
        if (type(item) is not dict or set(item) != {'id', 'type', 'text'}
                or item['type'] not in {'agent_message', 'reasoning'} or type(item['id']) is not str
                or not item['id'] or type(item['text']) is not str):
            raise ValueError('unverified item shape or execution-bearing field')
    elif kind == 'turn.completed':
        usage = event['usage']
        if (type(usage) is not dict
                or set(usage) != {'input_tokens', 'cached_input_tokens', 'cache_write_input_tokens',
                                  'output_tokens', 'reasoning_output_tokens'}
                or any(type(value) is not int or value < 0 for value in usage.values())):
            raise ValueError('unverified usage fields or values')


def audit(raw):
    metadata, final, usage = project(raw)
    reservation = {'parent_identity': 'fixture-parent', 'packet_sha256': 'a' * 64,
                   'route_attestation_reference': 'fixture-route', 'quota_attestation_reference': 'fixture-quota'}
    submission = {'fixture': 'no real invocation'}
    value = dict(reservation, schema='nca-host-response-audit/1', status='completed', host_capture=True,
                 raw_sha256=sha(final or b''), submission_sha256=A.digest(submission),
                 tools_audit_status='observed_zero_tool_use', forbidden_tool_uses=[], audit_reference='fixture-audit',
                 event_stream_status='complete', event_metadata=metadata,
                 restricted_trace_sha256=sha(raw), restricted_trace_reference='fixture-restricted-trace',
                 usage=usage['host'] if usage else None, input_isolation_status='verified_packet_only_no_repo_global_context',
                 preventive_isolation_status='not_claimed', timed_out=False, completion_certain=True)
    return value, reservation, submission, final


class NativeEnvelopeCompatibility(unittest.TestCase):
    def validate(self, raw):
        value, reservation, submission, final = audit(raw)
        A.validate_host_audit(value, reservation, submission, sha(final or b''))
        return value, final

    def test_reported_four_native_envelopes_need_no_native_sequence_or_channel(self):
        value, _ = self.validate(native_stream())
        self.assertEqual([event['sequence'] for event in value['event_metadata']], list(range(4)))
        self.assertEqual(value['usage'], dict(input_tokens=100, cached_input_tokens=20, output_tokens=30, credits='unknown'))

    def test_final_text_is_exact_utf8_without_trim_or_added_newline(self):
        text = ' \n{"fixture":"λ"}  '
        _, final = self.validate(native_stream(text))
        self.assertEqual(final, text.encode('utf-8'))
        self.assertFalse(final.endswith(b'\n'))

    def test_hashes_preserve_original_crlf_and_last_line_without_newline(self):
        raw = native_stream().replace(b'\n', b'\r\n')[:-2]
        value, _ = self.validate(raw)
        lines = raw.splitlines(keepends=True)
        self.assertEqual(value['restricted_trace_sha256'], sha(raw))
        self.assertEqual([event['event_sha256'] for event in value['event_metadata']], [sha(line) for line in lines])
        self.assertNotEqual(sha(lines[0]), sha(A.canonical(A.strict_json(lines[0]))))

    def test_command_execution_started_or_completed_is_rejected(self):
        lines = native_stream().splitlines(keepends=True)
        for kind in ['item.started', 'item.completed']:
            event = {'type': kind, 'item': {'id': 'fixture-tool', 'type': 'command_execution',
                                          'command': 'fixture command never executed', 'output': '', 'status': 'completed'}}
            raw = b''.join(lines[:2]) + json.dumps(event).encode() + b'\n' + b''.join(lines[2:])
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.validate(raw)

    def test_observed_error_reconnect_fallback_retry_is_rejected(self):
        lines = native_stream().splitlines(keepends=True)
        for kind in ['error', 'stream.reconnected', 'provider.fallback', 'request.retry']:
            raw = b''.join(lines[:2]) + json.dumps({'type': kind}).encode() + b'\n' + b''.join(lines[2:])
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.validate(raw)

    def test_missing_terminal_or_duplicate_final_is_rejected(self):
        lines = native_stream().splitlines(keepends=True)
        for raw in [b''.join(lines[:-1]), b''.join(lines[:3] + [lines[2]] + lines[3:])]:
            with self.assertRaises(ValueError):
                self.validate(raw)

    def test_blank_malformed_or_non_utf8_lines_are_rejected(self):
        raw = native_stream()
        for prefix in [b'\n', b'not-json\n', b'\xff\n']:
            with self.subTest(prefix=prefix), self.assertRaises((ValueError, UnicodeError)):
                self.validate(prefix + raw)

    def test_host_gap_timeout_or_uncertainty_is_rejected(self):
        value, reservation, submission, final = audit(native_stream())
        for change in [{'timed_out': True}, {'completion_certain': False}, {'event_stream_status': 'truncated'}]:
            invalid = dict(value, **change)
            with self.assertRaises(ValueError):
                A.validate_host_audit(invalid, reservation, submission, sha(final))
        invalid = copy.deepcopy(value)
        invalid['event_metadata'][2]['sequence'] = 3
        with self.assertRaises(ValueError):
            A.validate_host_audit(invalid, reservation, submission, sha(final))

    def test_extra_top_level_or_nested_fields_are_rejected_before_projection(self):
        events = [json.loads(line) for line in native_stream().splitlines()]
        for index in range(4):
            altered = copy.deepcopy(events)
            altered[index]['command'] = 'unverified execution field'
            raw = b''.join(json.dumps(event).encode() + b'\n' for event in altered)
            with self.subTest(index=index), self.assertRaises(ValueError):
                project(raw)
        for field in ['command', 'tool_calls', 'status', 'channel']:
            altered = copy.deepcopy(events)
            altered[2]['item'][field] = 'unverified'
            raw = b''.join(json.dumps(event).encode() + b'\n' for event in altered)
            with self.subTest(field=field), self.assertRaises(ValueError):
                project(raw)

    def test_unverified_harmless_fields_and_wrong_native_types_also_reject(self):
        events = [json.loads(line) for line in native_stream().splitlines()]
        variants = []
        changed = copy.deepcopy(events); changed[0]['sequence'] = 0; variants.append(changed)
        changed = copy.deepcopy(events); changed[3]['usage']['reasoning_tokens'] = 0; variants.append(changed)
        changed = copy.deepcopy(events); changed[3]['usage']['input_tokens'] = True; variants.append(changed)
        changed = copy.deepcopy(events); changed[2]['item']['type'] = 'unknown_reasoning'; variants.append(changed)
        for changed in variants:
            with self.assertRaises(ValueError):
                project(b''.join(json.dumps(event).encode() + b'\n' for event in changed))

    def test_all_five_native_counters_are_preserved_with_explicit_host_projection(self):
        raw = native_stream()
        _, _, usage = project(raw)
        self.assertEqual(usage['native'], {'input_tokens': 100, 'cached_input_tokens': 20,
            'cache_write_input_tokens': 5, 'output_tokens': 30, 'reasoning_output_tokens': 10})
        self.assertEqual(usage['host'], {'input_tokens': 100, 'cached_input_tokens': 20,
                                       'output_tokens': 30, 'credits': 'unknown'})
        self.validate(raw)

    def test_verified_reasoning_envelope_exports_hash_only(self):
        lines = native_stream().splitlines(keepends=True)
        # Dummy marker is not real model reasoning. The body must not be exported.
        marker = 'fixture-private-body-marker'
        reasoning = {'type': 'item.completed', 'item': {'id': 'fixture-reasoning',
                                                      'type': 'reasoning', 'text': marker}}
        raw = b''.join(lines[:2]) + json.dumps(reasoning).encode() + b'\n' + b''.join(lines[2:])
        value, _ = self.validate(raw)
        self.assertEqual(value['event_metadata'][2]['classification'], 'reasoning_summary')
        self.assertNotIn(marker, json.dumps(value))


if __name__ == '__main__':
    unittest.main()
