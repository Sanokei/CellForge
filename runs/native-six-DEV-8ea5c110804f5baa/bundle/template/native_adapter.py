"""Restricted one-call BFCL development adapter; never executes functions or infers answers."""
import ast,copy,math,re
VERSION='native-scalar-call-adapter/1'
class InvalidCall(ValueError):pass

def candidate_input(case):
    if not isinstance(case,dict) or set(case)-{'id','question','function'}:raise InvalidCall('only public native input fields permitted')
    q=case['question'];functions=case['function']
    if len(q)!=1 or len(q[0])!=1 or q[0][0]['role']!='user' or type(q[0][0]['content']) is not str:raise InvalidCall('single user turn required')
    if not 2<=len(functions)<=4 or len({f['name'] for f in functions})!=len(functions):raise InvalidCall('two to four unique function options required')
    for f in functions:
        if set(f)-{'name','description','parameters'}:raise InvalidCall('nonpublic schema field')
        schema=f['parameters']
        if schema.get('type') not in ('dict','object') or not isinstance(schema.get('properties'),dict):raise InvalidCall('object argument schema required')
        if len(schema['properties'])>6 or not set(schema.get('required',[]))<=set(schema['properties']):raise InvalidCall('unsupported parameter interface')
        if any(p.get('type') not in ('string','integer','float','number','boolean') for p in schema['properties'].values()):raise InvalidCall('scalar-only preparation scope')
    # ID, exposure metadata and teacher annotations are not model inputs.
    return {'request':q[0][0]['content'],'functions':copy.deepcopy(functions),'adapter_version':VERSION}

def _name(node):
    if isinstance(node,ast.Name):parts=[node.id]
    elif isinstance(node,ast.Attribute):parts=_name(node.value).split('.')+[node.attr]
    else:raise InvalidCall('only static dotted function names permitted')
    if any(not re.fullmatch('[A-Za-z][A-Za-z0-9_]*',p) or p.startswith('__') for p in parts):raise InvalidCall('unsafe function name')
    return '.'.join(parts)

def parse_packet(value):
    if isinstance(value,list):
        if len(value)!=1:raise InvalidCall('exactly one call required')
        value=value[0]
    if isinstance(value,str):
        if len(value)>8192:raise InvalidCall('packet too large')
        try:
            tree=ast.parse(value,mode='eval')
            if sum(1 for _ in ast.walk(tree))>256:raise InvalidCall('packet AST too large')
            call=tree.body
            if not isinstance(call,ast.Call) or call.args:raise InvalidCall('named literal arguments required')
            arguments={}
            for kw in call.keywords:
                if kw.arg is None or kw.arg in arguments:raise InvalidCall('expanded/duplicate keywords forbidden')
                arguments[kw.arg]=ast.literal_eval(kw.value)
            value={'name':_name(call.func),'arguments':arguments}
        except (SyntaxError,ValueError,TypeError,RecursionError) as exc:raise InvalidCall('invalid literal call: '+str(exc)) from exc
    if not isinstance(value,dict) or set(value)!={'name','arguments'} or type(value['name']) is not str or not isinstance(value['arguments'],dict):raise InvalidCall('exact name/arguments packet required')
    if any(type(k) is not str for k in value['arguments']):raise InvalidCall('argument names must be strings')
    return copy.deepcopy(value)

def _typed(value,kind):
    if kind=='string':return type(value) is str
    if kind=='boolean':return type(value) is bool
    if kind=='integer':return type(value) is int
    if kind in ('float','number'):
        try:return type(value) in (int,float) and math.isfinite(value)
        except OverflowError:return False
    return False

def validate_packet(value,public_input):
    packet=parse_packet(value);schemas={f['name']:f['parameters'] for f in public_input['functions']}
    if packet['name'] not in schemas:raise InvalidCall('function not supplied; no host name recovery')
    schema=schemas[packet['name']];arguments=packet['arguments'];properties=schema['properties']
    if not set(schema.get('required',[]))<=set(arguments):raise InvalidCall('missing required argument; no answer/default filling')
    if not set(arguments)<=set(properties):raise InvalidCall('unknown argument')
    for key,val in arguments.items():
        if not _typed(val,properties[key]['type']):raise InvalidCall('wrong scalar type for '+key)
    # Omitted optional flags remain omitted; this adapter never materializes defaults.
    return packet

def check_against_public_annotation(value,case,annotation):
    public=candidate_input(case)
    try:packet=validate_packet(value,public)
    except InvalidCall as exc:return {'schema_valid':False,'joint_exact_match':False,'reason':str(exc),'scope':'adapter contract only; not a model/benchmark score'}
    if annotation.get('id')!=case.get('id'):raise InvalidCall('annotation belongs to another case')
    for expected in annotation['ground_truth']:
        if len(expected)!=1 or packet['name'] not in expected:continue
        alternatives=expected[packet['name']];args=packet['arguments'];schema=next(f['parameters'] for f in public['functions'] if f['name']==packet['name'])
        if set(args)-set(alternatives):continue
        good=True
        for key,values in alternatives.items():
            if key not in args:
                good &= key not in schema.get('required',[]) and '' in values
            else:good &= args[key] in values
        if good:return {'schema_valid':True,'joint_exact_match':True,'scope':'adapter contract only; not a model/benchmark score'}
    return {'schema_valid':True,'joint_exact_match':False,'scope':'adapter contract only; not a model/benchmark score'}