import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepublish_scan as ps


def touch(root, rel, data=b"x"):
    p = os.path.join(root, *rel.split("/"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as fh:
        fh.write(data)


class ScanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def levels(self):
        return {(l, r) for l, r, _ in ps.scan(self.root, 1, 64)}

    def test_clean_tree_exits_zero(self):
        touch(self.root, "runs/a/REPORT.md", b"# ok\nsha256 abc")
        self.assertEqual(ps.main([self.root, "--quiet"]), 0)

    def test_hold_dirs(self):
        names = ["anchor", "_accept", "dev", "final_lm", "sealed1", "heldout", "private"]
        for d in names:
            touch(self.root, "runs/x/%s/f.json" % d)
        found = self.levels()
        for d in names:
            self.assertIn(("HOLD", "runs/x/" + d), found)
        self.assertEqual(ps.main([self.root, "--quiet"]), 1)

    def test_archives_pickles_env_keys_aggregates(self):
        files = ["a.tar.gz", "b.pkl", ".env", "id_rsa", "id_rsa.pub", "server-aggregate-1.jsonl"]
        for f in files:
            touch(self.root, f)
        holds = {r for l, r in self.levels() if l == "HOLD"}
        self.assertEqual(holds, set(files))

    def test_secret_strings_warn_only(self):
        touch(self.root, "n.txt", b'api_key = "abc"\nAuthorization: Bearer abcdefgh12345\n')
        touch(self.root, "h.txt", b"secret token " + b"a1" * 40 + b"\n")
        touch(self.root, "ok.json", b'{"sha256": "' + b"ab" * 32 + b'"}')
        found = self.levels()
        self.assertIn(("WARN", "n.txt"), found)
        self.assertIn(("WARN", "h.txt"), found)
        self.assertNotIn(("WARN", "ok.json"), found)
        self.assertEqual(ps.main([self.root, "--quiet"]), 0)

    def test_large_file_and_final_json_warn(self):
        touch(self.root, "big.bin", b"0" * (2 * 1024 * 1024))
        touch(self.root, "exports/FINAL.json")
        found = self.levels()
        self.assertIn(("WARN", "big.bin"), found)
        self.assertIn(("WARN", "exports/FINAL.json"), found)

    def test_held_dir_contents_not_read(self):
        touch(self.root, "anchor/secret.txt", b"api_key=zzz")
        rels = [r for _, r, _ in ps.scan(self.root, 1, 64)]
        self.assertEqual(rels, ["anchor"])

    def test_bad_dir_exit_2(self):
        self.assertEqual(ps.main([os.path.join(self.root, "nope")]), 2)


if __name__ == "__main__":
    unittest.main()
