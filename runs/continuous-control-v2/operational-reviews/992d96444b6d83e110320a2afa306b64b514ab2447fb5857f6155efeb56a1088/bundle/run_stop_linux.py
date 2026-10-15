import pathlib,subprocess,sys
root=pathlib.Path(__file__).resolve().parent
command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ConnectionAttempts=1','-o','StrictHostKeyChecking=yes','cellforge-1','/srv/arc3-claude/venv/bin/python -u -']
p=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
p.stdin.write((root/'stop_linux_stream.py').read_bytes());p.stdin.close()
with (root/'stop-linux-run.log').open('wb') as log:
    for line in p.stdout:
        log.write(line);log.flush()
        if line.startswith(b'REPORT_JSON:'):
            import json
            report=json.loads(line[len(b'REPORT_JSON:'):]);(root/'LINUX_STOP_LIFECYCLE.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n',encoding='utf8')
            print('FINAL LINUX REPORT '+str({k:report.get(k) for k in ['passed','error','production_unchanged','fixture_directory_removed']})+'; cases='+str(len(report['cases'])),flush=True)
        else:print(line.decode('utf8','replace').rstrip(),flush=True)
code=p.wait();print('SSH fixture exit '+str(code),flush=True);sys.exit(code)