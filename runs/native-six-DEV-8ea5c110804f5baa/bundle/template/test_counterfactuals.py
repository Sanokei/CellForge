import copy,unittest,pathlib
import native_slots4 as N
import native_adapter as A

def functions(names,kind='integer'):
    return [{'name':name,'description':name.rsplit('.',1)[-1]+' operation','parameters':{'type':'dict','properties':{'x1':{'type':kind},'x2':{'type':kind}} if kind=='integer' else {'city':{'type':'string'}},'required':['x1','x2'] if kind=='integer' else ['city']}} for name in names]
def public(request,fs):return {'request':request,'functions':fs,'adapter_version':A.VERSION}
W={'route':[0]*16,'binding':[0]*24,'null_gate':[0]*8};W['route'][12]=4;W['route'][13]=-4;W['route'][14]=1;W['binding'][2]=4;W['binding'][18]=4;W['binding'][22]=1
class CounterfactualABI(unittest.TestCase):
    def test_reversed_negated_clause_changes_candidate_features(self):
        fs=functions(['math.add','math.subtract']);a='Do not add 3 and 5; subtract instead';b='Do not subtract 3 and 5; add instead'
        self.assertEqual(N.words(a),N.words(b));self.assertNotEqual(N.route_features(a,fs[0],N.spans(a),fs),N.route_features(b,fs[0],N.spans(b),fs))
        self.assertEqual(N.solve(public(a,fs),W)['packet']['name'],'math.subtract');self.assertEqual(N.solve(public(b,fs),W)['packet']['name'],'math.add')
    def test_reversed_order_and_generic_other_names(self):
        fs=functions(['unit.alpha','unit.beta'])
        for request,name in [('Do not alpha 3 and 5; beta instead','unit.beta'),('beta instead; do not alpha 3 and 5','unit.beta'),('alpha instead; do not beta 3 and 5','unit.alpha')]:
            self.assertEqual(N.solve(public(request,fs),W)['packet']['name'],name)
    def test_unquoted_Paris_is_a_mechanical_value_span(self):
        fs=functions(['weather.current','stocks.current'],'string');request='Weather for Paris';p=public(request,fs);w=copy.deepcopy(W);w['route'][1]=4
        spans=N.spans(request);self.assertTrue(any(s['value']=='Paris' and request[s['start']:s['end']]=='Paris' and s['kind']=='unquoted' for s in spans))
        self.assertEqual(N.solve(p,w)['packet'],{'name':'weather.current','arguments':{'city':'Paris'}})
    def test_unquoted_lowercase_and_multiword_capacity(self):
        self.assertTrue(any(s['value']=='New York' for s in N.spans('weather for New York')))
        fs=functions(['weather.current','stocks.current'],'string');w=copy.deepcopy(W);w['route'][1]=4
        self.assertEqual(N.solve(public('weather for paris',fs),w)['packet']['arguments']['city'],'paris')
    def test_numeric_order_is_not_affected_by_string_candidates(self):
        p=public('add 3 and 5',functions(['math.add','math.subtract']));w=copy.deepcopy(W);w['route'][1]=4
        self.assertEqual(N.solve(p,w)['packet']['arguments'],{'x1':3,'x2':5})
    def test_NO_teacher_or_ID_field_and_NO_execution(self):
        p=public('add 3 and 5',functions(['math.add','math.subtract']));p['ground_truth']={}
        with self.assertRaises(A.InvalidCall):N.solve(p,W)
    def test_span_limit_fails_not_truncates(self):
        with self.assertRaises(A.InvalidCall):N.spans('word '*200)
    def test_version_distinct_and_no_case_specific_names(self):
        self.assertEqual(N.VERSION,'native-tool-slots4/1');source=pathlib.Path(N.__file__).read_text(encoding='utf-8')
        for word in ('Paris','triangle_properties','math.add','math.subtract'):self.assertNotIn(word,source)
if __name__=='__main__':unittest.main(verbosity=2)