"""Broad synthetic-only typed capacity and counterfactual policy checks."""
import copy,json,pathlib,unittest
import native_slots4 as N
import native_adapter as A
W=json.loads((pathlib.Path(__file__).parent.parent/'SYNTHETIC_REFERENCE_GENOME.json').read_text(encoding='utf-8'))
def spec(kind,description=''):return {'type':kind,'description':description}
def public(request,key,kind,description=''):
    return {'request':request,'adapter_version':A.VERSION,'functions':[{'name':'unit.alpha','description':'alpha operation','parameters':{'type':'dict','properties':{key:spec(kind,description)},'required':[key]}},{'name':'unit.beta','description':'beta operation','parameters':{'type':'dict','properties':{key:spec(kind,description)},'required':[key]}}]}
class TypedCandidateContract(unittest.TestCase):
    def test_mixed_raw_identifier_families(self):
        for value in ('A7/B-92','2019/XZ003','R2D2','0012-AB','v3.12.7','aa_bb:77','person@dept.example','alpha+beta#4','\u03b1-12/\u03b2','DEAD-BEEF-001'):
            with self.subTest(value=value):
                request='alpha identifier '+value+'.';found=[c for c in N.typed_candidates(request,spec('string')) if c['value']==value];self.assertTrue(found);self.assertTrue(all(request[c['start']:c['end']]==value for c in found))
    def test_raw_surface_and_trimmed_forms_both_exist(self):
        values={c['value'] for c in N.typed_candidates('alpha (A8/B9),',spec('string'))};self.assertIn('A8/B9',values);self.assertIn('(A8/B9),',values)
    def test_whitespace_strings_still_mechanical(self):
        for value in ('Lower Town','north west zone','New York','paris'):
            self.assertIn(value,[c['value'] for c in N.typed_candidates('alpha for '+value,spec('string'))])
    def test_unit_attached_numbers_have_no_unit_dictionary(self):
        for token,value in [('137quux',137),('-8ms',-8),('+2.75kHz',2.75),('8.2e-3\u03bcs',0.0082),('5%',5),('23\u00b0C',23),('2e3arbUnit',2000)]:
            with self.subTest(token=token):
                request='alpha value '+token;candidates=N.typed_candidates(request,spec('number'));found=[c for c in candidates if c['value']==value];self.assertTrue(found);self.assertTrue(all(c['kind']=='unit_magnitude' for c in found));self.assertTrue(all(request[c['start']:c['evidence_end']]==token for c in found))
    def test_integer_cast_is_exact_without_rounding(self):
        self.assertEqual([c['value'] for c in N.typed_candidates('7.0unit',spec('integer'))],[7]);self.assertEqual(N.typed_candidates('7.25unit',spec('integer')),[])
    def test_signed_decimal_and_scientific_numbers(self):
        for token,value in [('.125',0.125),('-.125',-0.125),('+12',12),('-3E2',-300),('1.25e+2',125)]:
            self.assertIn(value,[c['value'] for c in N.typed_candidates('alpha '+token,spec('number'))])
    def test_identifier_digits_are_not_silently_numeric_fragments(self):
        for token in ('AB12','R2D2','2024/ZX19','v1.2.3'):
            self.assertEqual(N.typed_candidates('alpha '+token,spec('integer')),[])
    def test_nonfinite_number_not_materialized(self):
        self.assertEqual(N.typed_candidates('alpha 1e9999kg',spec('number')),[])
    def test_strict_types_no_boolean_integer_alias(self):
        self.assertEqual(N.typed_candidates('alpha True False',spec('integer')),[]);self.assertEqual({type(c['value']) for c in N.typed_candidates('alpha true false',spec('boolean'))},{bool})
    def test_boolean_finite_domain_for_every_schema_without_defaults(self):
        for key in ('verbose','cache','preview','confirm','overwrite','recursive','compress','dry_run'):
            candidates=N.typed_candidates('alpha request',spec('boolean',key));self.assertEqual([c['value'] for c in candidates],[False,True]);self.assertTrue(all(c['kind']=='boolean_domain' and c['start']==c['end']==0 for c in candidates))
    def test_boolean_defaults_not_read(self):
        for value in (False,True):self.assertEqual(N.typed_candidates('alpha',{'type':'boolean','default':value}),N.typed_candidates('alpha',spec('boolean')))
    def test_implicit_boolean_counterfactuals_change_generic_schema_feature(self):
        for key in ('verbose','cache','preview','compress'):
            yes=public('alpha include '+key,key,'boolean',key+' flag');no=public('alpha do not include '+key,key,'boolean',key+' flag');self.assertGreater(N.schema_signal(yes['request'],key,spec('boolean',key)),0);self.assertLess(N.schema_signal(no['request'],key,spec('boolean',key)),0)
            self.assertTrue(N.solve(yes,W)['packet']['arguments'][key]);self.assertFalse(N.solve(no,W)['packet']['arguments'][key])
    def test_implicit_boolean_trace_declares_domain_not_raw_literal(self):
        result=N.solve(public('alpha include verbose','verbose','boolean','verbose flag'),W);self.assertEqual(result['trace'][0]['origin'],'boolean_domain');self.assertEqual(result['trace'][0]['span_start'],result['trace'][0]['span_end'])
    def test_explicit_false_is_not_overridden_by_positive_schema_words(self):
        for value in (False,True):
            result=N.solve(public('alpha cache '+str(value),'cache','boolean','cache flag'),W);self.assertIs(result['packet']['arguments']['cache'],value);self.assertEqual(result['trace'][0]['origin'],'boolean_literal')
    def test_optional_flags_remain_omitted(self):
        p=public('alpha include cache','cache','boolean');[f['parameters'].update(required=[]) for f in p['functions']];self.assertEqual(N.solve(p,W)['packet']['arguments'],{})
    def test_parameter_order_and_function_order_are_perturbations(self):
        p=public('alpha value 19quux','value','integer');before=N.solve(p,W)['packet'];p['functions'].reverse();self.assertEqual(N.solve(p,W)['packet'],before)
    def test_unquoted_numeric_string_remains_string(self):
        self.assertIn('00027',[c['value'] for c in N.typed_candidates('alpha 00027',spec('string'))]);self.assertIn(27,[c['value'] for c in N.typed_candidates('alpha 00027',spec('integer'))])
    def test_no_ID_teacher_expected_value_or_named_unit_rules(self):
        source=pathlib.Path(N.__file__).read_text(encoding='utf-8')
        for value in ('multiple_','get_case_info','get_discoverer','generate_sound_wave','2022/AL2562','440Hz','Hz','Paris'):self.assertNotIn(value,source)
        p=public('alpha 17unit','value','integer');p['id']='synthetic'
        with self.assertRaises(A.InvalidCall):N.solve(p,W)
    def test_capacity_limit_is_error_not_truncation(self):
        with self.assertRaises(A.InvalidCall):N.spans('token '*200)
    def test_schema_unsupported_types_are_not_filled(self):
        self.assertEqual(N.typed_candidates('alpha anything',spec('array')),[])
if __name__=='__main__':unittest.main(verbosity=2)