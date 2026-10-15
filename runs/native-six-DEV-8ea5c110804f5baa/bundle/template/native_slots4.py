"""Versioned numeric function/span ranking; no ID/gold access or tool execution."""
import ast,math,re
from decimal import Decimal,InvalidOperation
import native_adapter as A
VERSION='native-tool-slots4/1'
SHAPES={'route':16,'binding':24,'null_gate':8}
class InvalidGenome(ValueError):pass

def validate(weights):
    if set(weights)!=set(SHAPES):raise InvalidGenome('exact tensor vocabulary required')
    for name,n in SHAPES.items():
        values=weights[name]
        if len(values)!=n or any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>4 for v in values):raise InvalidGenome('invalid literal coefficients')

def words(text):
    text=re.sub(r'([a-z])([A-Z])',r'\1 \2',text)
    return set(re.findall(r'[a-z]+',text.lower()))
def overlap(a,b):return len(a&b)/max(1,len(a|b))
def dot(w,f):return sum(a*b for a,b in zip(w,f))
# Generic surface grammar and finite Boolean type domain. No entity/unit lookup.
SPAN=re.compile(r"(?<!\w)(?:\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*')|(?<!\w)(?:true|false)(?!\w)",re.I)
NUMBER=re.compile(r'(?<![\w./])(?P<number>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)(?P<unit>(?:[^\W\d_]|[%\u00b0])[\w%\u00b0]*)?(?=$|\s|[,;!?()\[\]{}]|[.:](?=\s|$))')
WRAPPERS_LEFT='"\'([{'
WRAPPERS_RIGHT='"\'.,;!?)]}'

def spans(request):
    if type(request) is not str or len(request)>8192:raise A.InvalidCall('request exceeds fixed limit')
    result=[];quoted=[];seen=set()
    def add(value,start,end,kind,evidence_end=None):
        key=(type(value).__name__,repr(value),start,end,kind)
        if key in seen:return
        seen.add(key);result.append({'value':value,'start':start,'end':end,'kind':kind,'evidence_start':start,'evidence_end':end if evidence_end is None else evidence_end})
        if len(result)>512:raise A.InvalidCall('span capacity exceeded; no truncation or case replacement')
    for match in SPAN.finditer(request):
        token=match.group();kind='quoted' if token[:1] in ('"',"'") else 'boolean_literal'
        try:value=ast.literal_eval(token) if kind=='quoted' else token.lower()=='true'
        except (SyntaxError,ValueError):continue
        add(value,match.start(),match.end(),kind)
        if kind=='quoted':quoted.append((match.start(),match.end()))
    def in_quote(start):return any(a<=start<b for a,b in quoted)
    for match in NUMBER.finditer(request):
        if in_quote(match.start()):continue
        token=match.group('number')
        if len(token)>128:continue
        try:
            decimal=Decimal(token);number=float(decimal)
            if not math.isfinite(number):continue
            value=int(decimal) if decimal==decimal.to_integral_value() else number
        except (InvalidOperation,OverflowError,ValueError):continue
        add(value,match.start('number'),match.end('number'),'unit_magnitude' if match.group('unit') else 'number',match.end())
    tokens=[]
    for match in re.finditer(r'\S+',request):
        if in_quote(match.start()):continue
        raw=match.group();left=len(raw)-len(raw.lstrip(WRAPPERS_LEFT));right=len(raw.rstrip(WRAPPERS_RIGHT));start=match.start()+left;end=match.start()+right
        if start>=end:continue
        tokens.append((start,end));add(request[start:end],start,end,'unquoted')
        if (start,end)!=(match.start(),match.end()):add(raw,match.start(),match.end(),'raw_surface')
    for i,first in enumerate(tokens):
        for width in (2,3):
            group=tokens[i:i+width]
            if len(group)!=width or any(not re.fullmatch(r'[\s,-]+',request[a[1]:b[0]]) for a,b in zip(group,group[1:])):continue
            add(request[first[0]:group[-1][1]],first[0],group[-1][1],'unquoted')
    result.sort(key=lambda x:(x['start'],x['end'],x['kind']))
    return result

def typed_candidates(request,spec,literals=None):
    literals=spans(request) if literals is None else literals
    compatible=[dict(s) for s in literals if A._typed(s['value'],spec['type'])]
    if spec['type']=='boolean':
        # Both values of this finite type are eligible; neither is a supplied default.
        compatible += [{'value':value,'start':0,'end':0,'evidence_start':0,'evidence_end':0,'kind':'boolean_domain'} for value in (False,True)]
    if len(compatible)>514:raise A.InvalidCall('typed capacity exceeded')
    return compatible

STOPWORDS=set('a an the and or to of for in at on from with as is are was were be by it this that whether if please'.split())
def lexemes(text):
    result=set()
    for token in words(text)-STOPWORDS:
        result.add(token)
        for suffix in ('ing','ed','es','s','ly'):
            if token.endswith(suffix) and len(token)-len(suffix)>=3:result.add(token[:-len(suffix)])
    return result

def schema_signal(request,key,spec):
    vocabulary=lexemes(key+' '+spec.get('description',''));neg=False;hits=[]
    for match in re.finditer(r"(?u)[^\W\d_]\w*(?:'\w+)?|[;,.!?]",request):
        token=match.group().lower()
        if token in ';,.!?' or token in ('but','however','rather','instead'):neg=False;continue
        if token in ('not','never','without','avoid',"don't",'dont'):neg=True;continue
        if lexemes(token)&vocabulary:hits.append(-1 if neg else 1)
    return sum(hits)/max(1,len(hits))

def clause_signals(request,function,functions):
    vocab=[words(f['name']+' '+f.get('description','')) for f in functions]
    common=set.intersection(*vocab) if vocab else set()
    specific=words(function['name']+' '+function.get('description',''))-common
    neg=False;clause=0;hits=[]
    for m in re.finditer(r"(?u)[^\W\d_]\w*(?:'\w+)?|[;,.!?]",request):
        token=m.group().lower()
        if token in ';,.!?':neg=False;clause+=1;continue
        if token in ('but','however','rather','instead'):neg=False;clause+=1;continue
        if token in ('not','never','without','avoid',"don't",'dont'):neg=True;continue
        if token in specific:hits.append((not neg,clause,m.start()))
    if not hits:return [0,0,0,0]
    positive=sum(v[0] for v in hits)/len(hits);negative=1-positive;last=hits[-1]
    return [positive,negative,(1 if last[0] else -1)*(1+last[1])/max(1,clause+1),sum(v[0] and v[1]>0 for v in hits)/len(hits)]

def route_features(request,function,literals,functions=None):
    q=words(request);n=words(function['name']);d=words(function.get('description',''));props=function['parameters']['properties'];required=function['parameters'].get('required',[]);types=[p['type'] for p in props.values()]
    available=sum(bool(typed_candidates(request,props[k],literals)) for k in required)/max(1,len(required))
    return [1,overlap(q,n),overlap(q,d),len(q&n)/max(1,len(n)),len(q&d)/max(1,len(q)),len(required)/6,(len(props)-len(required))/6,types.count('string')/6,types.count('integer')/6,sum(t in ('float','number') for t in types)/6,types.count('boolean')/6,available,*clause_signals(request,function,functions or [function])]

def binding_features(request,key,spec,position,count,span,index,compatible_count):
    suffix=re.search(r'(\d+)$',key);role=int(suffix.group())-1 if suffix else position
    denom=max(1,compatible_count-1);value=span['value'];before=words(request[max(0,span['start']-40):span['start']]);after=words(request[span['end']:span['end']+40]);name=words(key);desc=words(spec.get('description',''))
    numeric=float(value) if type(value) in (int,float) else 0;length=max(1,len(request));pos=position/max(1,count-1);idx=index/denom
    polarity=(1 if value else -1) if type(value) is bool else 0
    return [1,float(span['kind']!='boolean_domain'),-abs(role-index)/max(1,count,compatible_count),-abs(pos-idx),overlap(name,before),overlap(name,after),overlap(desc,before|after),span['start']/length,span['end']/length,pos,idx,float(suffix is not None),float(type(value) is str),float(type(value) is int),float(type(value) is float),float(type(value) is bool),polarity if type(value) is bool else numeric/(1+abs(numeric)),(1 if span['kind']=='boolean_literal' else polarity*schema_signal(request,key,spec)) if type(value) is bool else abs(numeric)/(1+abs(numeric)),float(bool(re.search(r'(?i)\b(?:for|in|at|from|to)\s*$',request[:span['start']]))),len(key)/128,compatible_count/512,float(span['kind']=='quoted'),float(type(value) is str and bool(value) and value[0].isupper()),-len(words(str(value)))/max(1,3)]

def solve(public,weights):
    validate(weights)
    if set(public)!={'request','functions','adapter_version'}:raise A.InvalidCall('public input only; no IDs or teachers')
    request=public['request'];literals=spans(request);functions=public['functions']
    if not 2<=len(functions)<=4:raise A.InvalidCall('function count outside contract')
    ranked=sorted(functions,key=lambda f:(-dot(weights['route'],route_features(request,f,literals,functions)),f['name']))
    selected=ranked[0];schema=selected['parameters'];required=set(schema.get('required',[]));keys=sorted(schema['properties']);arguments={};trace=[]
    for position,key in enumerate(keys):
        spec=schema['properties'][key];compatible=typed_candidates(request,spec,literals)
        if key not in required:
            gate=[1,float(bool(compatible)),len(compatible)/64,overlap(words(key),words(request)),float(spec['type']=='boolean'),float(spec['type']=='string'),position/max(1,len(keys)-1),len(required)/6]
            if dot(weights['null_gate'],gate)<=0:continue
        if not compatible:raise A.InvalidCall('required typed candidate absent; no default filling')
        scores=[dot(weights['binding'],binding_features(request,key,spec,position,len(keys),s,i,len(compatible))) for i,s in enumerate(compatible)]
        best=max(range(len(compatible)),key=lambda i:(scores[i],-compatible[i]['start']));chosen=compatible[best];arguments[key]=chosen['value'];trace.append({'parameter':key,'span_start':chosen['start'],'span_end':chosen['end'],'score':scores[best],'origin':chosen['kind'],'evidence_start':chosen['evidence_start'],'evidence_end':chosen['evidence_end']})
    packet=A.validate_packet({'name':selected['name'],'arguments':arguments},public)
    return {'packet':packet,'trace':trace,'template_version':VERSION}

def lexical_baseline(public):
    weights={'route':[0]*16,'binding':[0]*24,'null_gate':[0]*8};weights['route'][1]=4;weights['route'][2]=1;weights['route'][12]=4;weights['route'][13]=-4;weights['route'][14]=1;weights['binding'][2]=4;weights['binding'][18]=4;weights['binding'][22]=1
    return solve(public,weights)