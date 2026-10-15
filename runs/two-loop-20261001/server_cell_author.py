import json,pathlib,subprocess,os,hashlib,time
base=pathlib.Path('/srv/cellforge-inbox/two-loop-20261001/author-cell-g3')
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
rows=json.loads((base/'development-prompts.json').read_text())
for row in rows:
 p=row['prompt'];p['supplied_prompt_sha256']=digest(p);p['supplied_feedback_sha256']=digest(p['development_evidence'])
text='Author fresh literal numeric cell revisions using only the enclosed development feedback. Do not use tools, read files, or access reporting. Return ONLY a JSON array of {seed,response}. Each response: model gpt-6.1-sol, reasoning low, parent_id and parent_params exact, tensors with all39literal positions, feature_order and role_order exact, prompt_sha256 supplied_prompt_sha256, feedback_sha256 supplied_feedback_sha256, charged_positions39, interface tool-action-joint/2. Make a justified nonzero weight revision. No guaranteed improvement; no fabricated outcomes.\n'+json.dumps(rows)
env=dict(os.environ)
for k in ('OPENAI_API_KEY','AZURE_OPENAI_API_KEY'):env.pop(k,None)
cmd=['/root/.local/bin/codex','exec','--ignore-user-config','--ephemeral','--skip-git-repo-check','--sandbox','read-only','-m','gpt-6.1-sol','-c','model_reasoning_effort="low"','--json','-C',str(base),'-o',str(base/'response.json'),'-']
(base/'START.json').write_text(json.dumps({'utc':time.time(),'model':'gpt-6.1-sol','reasoning':'low','auth':'ChatGPT subscription','prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'paid_api_fallback':False}))
with (base/'events.jsonl').open('w') as log:
 result=subprocess.run(cmd,input=text,text=True,stdout=log,stderr=log,env=env,timeout=300)
(base/'EXIT.json').write_text(json.dumps({'returncode':result.returncode,'utc':time.time()}))
print('author_returncode',result.returncode)
