"""Worker lifecycle regression fixtures; no model-author calls."""
import io
from pathlib import Path
import queue
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import structcode as X


class Lifecycle(unittest.TestCase):
    def test_close_kills_reaps_closes_stdin_and_joins_reader_once(self):
        box = X.Sandbox()
        proc = box.proc = Mock()
        proc.poll.return_value = None
        box.reader = Mock()
        box.close()
        proc.kill.assert_called_once_with()
        proc.wait.assert_called_once_with(timeout=X.TIMEOUT)
        proc.stdin.close.assert_called_once_with()
        box.reader.join.assert_called_once_with(timeout=X.TIMEOUT)
        self.assertIsNone(box.proc)
        box.close()
        proc.wait.assert_called_once()

    def test_already_exited_worker_is_reaped_without_kill(self):
        box = X.Sandbox()
        proc = box.proc = Mock()
        proc.poll.return_value = 1
        box.close()
        proc.kill.assert_not_called()
        proc.wait.assert_called_once_with(timeout=X.TIMEOUT)
        proc.stdin.close.assert_called_once()

    def test_bounded_reap_timeout_retains_ownership_for_retry(self):
        box = X.Sandbox()
        proc = box.proc = Mock()
        proc.poll.return_value = None
        proc.wait.side_effect = [subprocess.TimeoutExpired('worker', X.TIMEOUT), 0]
        box.close()
        self.assertIs(box.proc, proc)
        proc.stdin.close.assert_called_once()
        box.close()
        self.assertIsNone(box.proc)
        self.assertEqual(proc.wait.call_count, 2)

    def test_call_after_reap_timeout_returns_pending_without_write_or_restart(self):
        box = X.Sandbox()
        proc = box.proc = Mock()
        proc.stdin = io.StringIO()
        proc.poll.return_value = None
        proc.wait.side_effect = subprocess.TimeoutExpired('worker', X.TIMEOUT)
        box.close()
        with patch.object(box, '_start') as start:
            out, error = box.call('def run(params, grid, demos):\n    return grid\n', {'a': [0]}, [[1]], [])
        self.assertIsNone(out)
        self.assertIn('cleanup pending', error)
        self.assertIs(box.proc, proc)
        start.assert_not_called()

    def test_dead_worker_is_cleaned_before_replacement(self):
        box = X.Sandbox()
        old = box.proc = Mock()
        old.poll.return_value = 1
        old.stdin = io.StringIO()
        old_reader = box.reader = Mock()
        new = Mock()
        new.poll.return_value = None
        def start():
            self.assertTrue(old.stdin.closed)
            old.wait.assert_called_once_with(timeout=X.TIMEOUT)
            old_reader.join.assert_called_once_with(timeout=X.TIMEOUT)
            box.proc = new
            box.lines = queue.Queue()
            box.lines.put('{"id": 1, "out": [[1]]}')
            box.reader = Mock()
        with patch.object(box, '_start', side_effect=start) as restart:
            result = box.call('def run(params, grid, demos):\n    return grid\n', {'a': [0]}, [[1]], [])
        self.assertEqual(result, ([[1]], None))
        restart.assert_called_once_with()
        old.kill.assert_not_called()

    def test_stdin_broken_pipe_during_close_does_not_interrupt_cleanup(self):
        box = X.Sandbox()
        proc = box.proc = Mock()
        proc.poll.return_value = 1
        proc.stdin.close.side_effect = BrokenPipeError('buffered flush failed')
        reader = box.reader = Mock()
        box.close()
        proc.wait.assert_called_once_with(timeout=X.TIMEOUT)
        reader.join.assert_called_once_with(timeout=X.TIMEOUT)
        self.assertIsNone(box.proc)

    def test_pending_cleanup_can_recover_before_next_call(self):
        box = X.Sandbox()
        proc = box.proc = Mock()
        proc.stdin = io.StringIO()
        proc.poll.return_value = None
        proc.wait.side_effect = [subprocess.TimeoutExpired('worker', X.TIMEOUT), 0]
        box.close()
        proc.poll.return_value = 0
        def start():
            self.assertIsNone(box.proc)
            box.proc = Mock()
            box.lines = queue.Queue()
            box.lines.put('{"id": 1, "out": [[1]]}')
        with patch.object(box, '_start', side_effect=start):
            result = box.call('def run(params, grid, demos):\n    return grid\n', {'a': [0]}, [[1]], [])
        self.assertEqual(result, ([[1]], None))
        self.assertEqual(proc.wait.call_count, 2)

    def test_pump_closes_stdout_and_reports_end_of_stream(self):
        proc = Mock(stdout=io.StringIO('one\ntwo\n'))
        lines = queue.Queue()
        X._pump(proc, lines)
        self.assertTrue(proc.stdout.closed)
        self.assertEqual([lines.get_nowait() for _ in range(3)], ['one\n', 'two\n', None])

    def test_pump_read_error_still_closes_stream_and_sends_sentinel(self):
        class BrokenRead(io.StringIO):
            def __next__(self):
                raise OSError('read failed')
        proc = Mock(stdout=BrokenRead(''))
        lines = queue.Queue()
        with self.assertRaisesRegex(OSError, 'read failed'):
            X._pump(proc, lines)
        self.assertTrue(proc.stdout.closed)
        self.assertIsNone(lines.get_nowait())

    def test_pump_close_error_still_sends_sentinel(self):
        class BrokenClose(io.StringIO):
            def close(self):
                super().close()
                raise OSError('close failed')
        proc = Mock(stdout=BrokenClose(''))
        lines = queue.Queue()
        with self.assertRaisesRegex(OSError, 'close failed'):
            X._pump(proc, lines)
        self.assertTrue(proc.stdout.closed)
        self.assertIsNone(lines.get_nowait())

    def test_real_worker_close_reaps_and_releases_streams(self):
        box = X.Sandbox(timeout=3)
        self.addCleanup(box.close)
        out, error = box.call('def run(params, grid, demos):\n    return grid\n', {'a': [0]}, [[1]], [])
        self.assertEqual((out, error), ([[1]], None))
        proc, reader = box.proc, box.reader
        box.close()
        self.assertIsNotNone(proc.returncode)
        self.assertTrue(proc.stdin.closed)
        self.assertTrue(proc.stdout.closed)
        self.assertFalse(reader.is_alive())
        self.assertIsNone(box.proc)


if __name__ == '__main__':
    unittest.main()
