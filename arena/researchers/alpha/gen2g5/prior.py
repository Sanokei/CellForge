import json, sys
d = json.load(open('/srv/cellforge-inbox/two-loop-20261001/shared-cell-registry/LATEST.json'))
def walk(o, depth=0):
    pad = '  '*depth
    if isinstance(o, dict):
        print(pad + 'DICT keys: ' + str(list(o.keys())[:15]))
        for k,v in list(o.items())[:4]:
            print(pad + '[' + str(k) + ']')
            walk(v, depth+1)
    elif isinstance(o, list):
        print(pad + 'LIST len %d' % len(o))
        for v in o[:3]: walk(v, depth+1)
    else:
        print(pad + '%s: %s' % (type(o).__name__, str(o)[:100]))
walk(d)
