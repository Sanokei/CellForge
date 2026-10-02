import tempfile,unittest
from pathlib import Path
from cellforge.construction.action_packet import validate_file_action
WORKSPACE=Path(r'C:\Users\wkeif\Documents\Codex\2026-09-30\task-5').resolve()
class ActionPacketTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='action-boundary-',dir=WORKSPACE)
  self.root=Path(self.tmp.name).resolve()
  self.assertTrue(self.root.is_relative_to(WORKSPACE))
 def tearDown(self):
  if not self.root.is_relative_to(WORKSPACE):raise RuntimeError('cleanup root escaped workspace')
  self.tmp.cleanup()
 def test_literal_parameterized_write_and_edit_on_isolated_fixture(self):
  # These are transport fixtures, not model-authored capability claims.
  write={'type':'call','name':'write_file','arguments':{'path':'note.txt','content':'alpha\nbeta\n'}}
  checked=validate_file_action(write,allowed_root=self.root)
  path=Path(checked['resolved_path']);self.assertFalse(path.exists())
  path.write_text(checked['packet']['arguments']['content'],encoding=checked['encoding'])
  edit={'type':'call','name':'edit_in_place','arguments':{'path':'note.txt','old':'beta','new':'gamma','expected_matches':1}}
  checked=validate_file_action(edit,allowed_root=self.root)
  source=path.read_text(encoding='utf-8');args=checked['packet']['arguments']
  self.assertEqual(source.count(args['old']),args['expected_matches'])
  path.write_text(source.replace(args['old'],args['new']),encoding='utf-8')
  self.assertEqual(path.read_text(),'alpha\ngamma\n')
 def test_validator_alone_never_writes(self):
  validate_file_action({'type':'call','name':'write_file','arguments':{'path':'untouched.txt','content':'x'}},allowed_root=self.root)
  self.assertFalse((self.root/'untouched.txt').exists())
 def test_outside_and_device_paths_rejected(self):
  for path in ('../outside.txt',str(WORKSPACE/'outside.txt'),'C:relative.txt','NUL','other:stream','folder/../../outside.txt','file.txt.'):
   with self.assertRaises(ValueError):validate_file_action({'type':'call','name':'write_file','arguments':{'path':path,'content':'x'}},allowed_root=self.root)
 def test_leaked_fields_unbounded_or_missing_args_rejected(self):
  requests=[{'type':'call','name':'write_file','arguments':{'path':'x.txt','content':'x','target':'secret'}},
   {'type':'call','name':'edit_in_place','arguments':{'path':'missing.txt','old':'a','new':'b','expected_matches':1}},
   {'type':'call','name':'write_file','arguments':{'path':'x.txt','content':'x'*4097}},
   {'type':'call','name':'run_shell','arguments':{'path':'x.txt','content':'x'}}]
  for packet in requests:
   with self.assertRaises(ValueError):validate_file_action(packet,allowed_root=self.root)
if __name__=='__main__':unittest.main(verbosity=2)
