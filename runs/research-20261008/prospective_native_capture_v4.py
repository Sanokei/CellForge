"""Prospective capture only. Importing this module never dispatches a process."""
import hashlib,json,math,os,pathlib,select,signal,subprocess,time
USAGE_KEYS={'input_tokens','cached_input_tokens','cache_write_input_tokens','output_tokens','reasoning_output_tokens'}
def exact(obj,keys):
    if type(obj) is not dict or set(obj)!=set(keys): raise ValueError('unexpected/missing native fields')
def text(value):
    if type(value) is not str or not value: raise ValueError('nonempty string required')
def pairs(items):
    result={}
    for key,value in items:
        if key in result: raise ValueError('duplicate JSON key')
        result[key]=value
    return result
def invalid_constant(value): raise ValueError('nonstandard JSON number')
def native(raw):
    if not raw.endswith(b'\n'): raise ValueError('unterminated native line')
    obj=json.loads(raw.decode('utf-8'),object_pairs_hook=pairs,parse_constant=invalid_constant)
    if type(obj) is not dict: raise ValueError('native envelope must be object')
    kind=obj.get('type')
    if kind=='thread.started':
        exact(obj,{'type','thread_id'});text(obj['thread_id'])
    elif kind=='turn.started': exact(obj,{'type'})
    elif kind=='item.completed':
        exact(obj,{'type','item'});item=obj['item'];exact(item,{'id','type','text'})
        text(item['id'])
        if item['type'] not in {'agent_message','reasoning'}: raise ValueError('unapproved item type')
        if type(item['text']) is not str: raise ValueError('message text must be string')
    elif kind=='turn.completed':
        exact(obj,{'type','usage'});exact(obj['usage'],USAGE_KEYS)
        for value in obj['usage'].values():
            if type(value) is not int or value<0: raise ValueError('usage must be nonnegative integer')
    else: raise ValueError('unknown/disallowed native event')
    return obj
class Validator:
    def __init__(self): self.phase=0;self.final=None;self.indices=[];self.message_ids=set()
    def add(self,raw):
        # Validate complete object BEFORE metadata projection.
        obj=native(raw);kind=obj['type']
        if self.phase==0 and kind=='thread.started':self.phase=1
        elif self.phase==1 and kind=='turn.started':self.phase=2
        elif self.phase==2 and kind=='item.completed':
            item=obj['item']
            if item['id'] in self.message_ids:raise ValueError('duplicate completed item ID')
            self.message_ids.add(item['id'])
            if item['type']=='agent_message':
                if self.final is not None:raise ValueError('exactly one final agent_message required')
                self.final=item['text'].encode('utf-8')
            elif self.final is not None:raise ValueError('reasoning after final agent_message rejected')
        elif self.phase==2 and kind=='turn.completed' and self.final is not None:self.phase=3
        else:raise ValueError('invalid event order or event after terminal completion')
        self.indices.append({'raw_line_index':len(self.indices),'raw_line_bytes':len(raw),'raw_line_sha256':hashlib.sha256(raw).hexdigest(),'type':kind})
    def finish(self):
        if self.phase!=3:raise ValueError('missing successful terminal completion')
        return self.final

def validate_saved(path):
    v=Validator()
    with pathlib.Path(path).open('rb') as source:
        for line in source:v.add(line)
    return v.finish(),v.indices

def capture(argv,prompt,destination,*,dispatch_authorized=False,wall_seconds=120,rss_limit_kib=2097152):
    """Only a separately authorized controller may call this; quota gate remains controller-owned."""
    if dispatch_authorized is not True:raise PermissionError('No separately authorized dispatch')
    if type(prompt) is not bytes or not argv:raise ValueError('exact prompt bytes and argv required')
    out=pathlib.Path(destination);out.mkdir(exist_ok=False)
    (out/'prompt.bin').write_bytes(prompt)
    (out/'command.json').write_text(json.dumps(argv,indent=2)+'\n')
    env=dict(os.environ)
    for key in ['OPENAI_API_KEY','AZURE_OPENAI_API_KEY']:env.pop(key,None)
    def owned_rss(pid):
        rows={}
        for d in pathlib.Path('/proc').iterdir():
            if not d.name.isdigit():continue
            try:
                s={a.split(':',1)[0]:a.split(':',1)[1].strip() for a in (d/'status').read_text().splitlines() if ':' in a}
                rows[int(d.name)]=(int(s.get('PPid','0')),int(s.get('VmRSS','0 kB').split()[0]))
            except (OSError,ValueError):continue
        owned={pid}
        for unused in range(16):
            expanded=owned|{pid for pid,(ppid,rss) in rows.items() if ppid in owned}
            if expanded==owned:break
            owned=expanded
        return sum(rows.get(pid,(0,0))[1] for pid in owned)
    start=time.monotonic();v=Validator();reject=None;buffer=b'';peak=0;eof=set();term_at=None;raw_line_audit=[]
    p=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,start_new_session=True)
    (out/'start.json').write_text(json.dumps({'pid':p.pid,'attempt_charged':1,'wall_seconds':wall_seconds,'rss_limit_kib':rss_limit_kib,'virtual_address_space_cap':None})+'\n')
    streams={p.stdout:(out/'stdout.jsonl').open('wb'),p.stderr:(out/'stderr.bin').open('wb')}
    try:
        p.stdin.write(prompt);p.stdin.close()
        while streams:
            now=time.monotonic();rss=owned_rss(p.pid);peak=max(peak,rss)
            if now-start>wall_seconds:reject=reject or 'wall timeout'
            if rss>rss_limit_kib:reject=reject or 'RSS limit'
            if reject and term_at is None:
                term_at=now
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            if term_at is not None and now-term_at>5:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            for stream in select.select(list(streams),[],[],0.2)[0]:
                data=os.read(stream.fileno(),65536)
                if not data:
                    eof.add('stdout' if stream is p.stdout else 'stderr');streams[stream].close();del streams[stream];continue
                streams[stream].write(data);streams[stream].flush()
                if stream is p.stderr:
                    reject=reject or 'nonempty stderr; recovery/error uncertainty'
                else:
                    buffer+=data
                    while b'\n' in buffer:
                        line,buffer=buffer.split(b'\n',1)
                        raw_line_audit.append({'raw_line_index':len(raw_line_audit),'raw_line_bytes':len(line)+1,'raw_line_sha256':hashlib.sha256(line+b'\n').hexdigest()})
                        if reject is None:
                            try:v.add(line+b'\n')
                            except (ValueError,UnicodeError,TypeError) as error:reject=str(error)
    except Exception as error:
        reject=reject or ('capture exception '+type(error).__name__)
    finally:
        for stream,f in streams.items():f.close()
        if p.poll() is None:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        try:p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            p.wait()
    if buffer:reject=reject or 'unterminated stdout tail'
    if eof!={'stdout','stderr'}:reject=reject or 'missing EOF'
    if p.returncode!=0:reject=reject or 'nonzero exit'
    if reject is None:
        try:final=v.finish()
        except ValueError as error:reject=str(error)
    if reject is None:(out/'final-text.bin').write_bytes(final)
    result={'accepted':reject is None,'rejection':reject,'exit_code':p.returncode,'stdout_EOF':'stdout' in eof,'stderr_EOF':'stderr' in eof,'wall_seconds':round(time.monotonic()-start,3),'peak_owned_RSS_KiB':peak,'raw_stdout_sha256':hashlib.sha256((out/'stdout.jsonl').read_bytes()).hexdigest(),'raw_stderr_sha256':hashlib.sha256((out/'stderr.bin').read_bytes()).hexdigest(),'raw_line_audit':raw_line_audit,'unterminated_tail_bytes':len(buffer),'validated_prefix_metadata':v.indices,'charged_attempts':1,'no_automatic_wrapper_retry':True}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result
