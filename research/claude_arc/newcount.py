import json
r=json.load(open('results_training_d2_v4.json'));s={x['task'] for x in r if x['solved']}
o={x.replace(chr(92),'/').split('/')[-1] for x in json.load(open('objsolved.json'))}
print("new",len(o-s),"overlap",len(o&s),"total union",len(o|s))
