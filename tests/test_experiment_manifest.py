import unittest,tempfile,pathlib,json
from cellforge.construction.experiment_manifest import load_json_strict,validate_action_manifest,freeze_manifest,assert_manifest_unchanged
class TestExperimentManifest(unittest.TestCase):
 def manifest(self):
  return {k:[{"id":k,"request":"write 'x' to a.txt","initial":{},"expected":{"a.txt":"x"},"action":"write_file"}] for k in ("fresh","archive","safety","reporting")}
 def test_duplicate_keys_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/"m.json";p.write_text('{"reporting":[],"reporting":"note"}')
   with self.assertRaises(ValueError):load_json_strict(p)
 def test_overwritten_case_list_and_wrong_filename_rejected(self):
  m=self.manifest();m["reporting"]="note"
  with self.assertRaises(ValueError):validate_action_manifest(m)
  m=self.manifest();m["reporting"][0]["expected"]={"JOURNAL":"x"}
  with self.assertRaises(ValueError):validate_action_manifest(m)
 def test_label_or_partition_change_breaks_frozen_hash(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/"m.json";m=self.manifest();p.write_text(json.dumps(m))
   receipt=freeze_manifest(p,"action");assert_manifest_unchanged(p,receipt["sha256"])
   m["reporting"][0]["expected"]["a.txt"]="y";p.write_text(json.dumps(m))
   with self.assertRaises(ValueError):assert_manifest_unchanged(p,receipt["sha256"])
 def test_development_report_overlap_rejected(self):
  m=self.manifest();m["reporting"][0]["id"]="fresh"
  with self.assertRaises(ValueError):validate_action_manifest(m)
