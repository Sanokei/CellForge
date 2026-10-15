
import pathlib,subprocess,sys,time
b=pathlib.Path(__file__).resolve().parent
cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ConnectionAttempts=1','-o','StrictHostKeyChecking=yes','cellforge-1','/srv/arc3-claude/venv/bin/python -B -u -']
p=subprocess.run(cmd,input=(b/'new_linux_probe.py').read_bytes(),capture_output=True,timeout=240)
(b/'new-linux-tests.stdout').write_bytes(p.stdout);(b/'new-linux-tests.stderr').write_bytes(p.stderr)
for line in p.stdout.decode('utf8','replace').splitlines():
 if line.startswith('PRODUCTION_REQUEST:'):(b/'PRODUCTION_RESUME_REQUEST.json').write_bytes((line.split(':',1)[1]+'\n').encode())
 elif line.startswith('FRESH_CHECKPOINT:'):(b/'FRESH_CHECKPOINT.json').write_bytes((line.split(':',1)[1]+'\n').encode())
 elif line.startswith('PROTECTED_AFTER:'):(b/'PROTECTED_AFTER.json').write_bytes((line.split(':',1)[1]+'\n').encode())
 else:print(line)
print(p.stderr.decode('utf8','replace'),file=sys.stderr);sys.exit(p.returncode)
