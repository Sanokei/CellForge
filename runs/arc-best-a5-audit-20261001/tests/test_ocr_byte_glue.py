import unittest
from pathlib import Path
import numpy as np
from cellforge.construction.ocr_byte_glue import project_ocr_category, SOURCE_VOCAB, DESTINATION_INDICES
from cellforge.ir.graph import Graph
from cellforge.runtime.interp import Interpreter

PARAMS={'projection.gain':[1,1,1,1,1,1,1,1,1,1,1,1,1],
        'projection.bias':[0,0,0,0,0,0,0,0,0,0,0,0,0]}
ROOT=Path(__file__).resolve().parents[1]

class OCRByteGlueTests(unittest.TestCase):
    def test_all_actual_donor_categories_and_recipient_step_parity(self):
        donor=Graph.load(ROOT/'runs/p1/exports/r1-iter-b')
        recipient=Graph.load(ROOT/'runs/action-deduction/component-v2')
        self.assertEqual(donor.meta['vocab'],SOURCE_VOCAB)
        self.assertEqual(recipient.meta['text_codec'],'utf8-bytes/1')
        for i,dest in enumerate(DESTINATION_INDICES):
            projected=project_ocr_category(PARAMS,np.eye(13)[i])
            native=np.eye(259)[dest]
            self.assertTrue(np.array_equal(projected,native))
            args={'ctx':np.zeros((1,64,259)),'state':np.zeros((1,recipient.meta['state_dim'])),'step':np.zeros((1,1))}
            a=Interpreter(recipient).run_bundle(recipient.entries['step'],dict(args,prev=projected[None]),'/bridge')
            b=Interpreter(recipient).run_bundle(recipient.entries['step'],dict(args,prev=native[None]),'/native')
            for k in a:self.assertTrue(np.array_equal(a[k],b[k]))
    def test_negative_observations_and_tensor_contract(self):
        for source in (np.zeros(13),np.ones(13),np.eye(259)[51],np.full(13,np.nan)):
            with self.assertRaises(ValueError):project_ocr_category(PARAMS,source)
        with self.assertRaises(ValueError):project_ocr_category(PARAMS,np.eye(13)[3],failed=True)
        bad=dict(PARAMS,**{'projection.gain':[2]*13})
        with self.assertRaises(ValueError):project_ocr_category(bad,np.eye(13)[3])
    def test_exact_semantic_capacity_gap(self):
        # Port repair cannot invent characters needed by the actual parser.
        required=set('Fact a. Is a entailed?')
        recognized=set(SOURCE_VOCAB[3:])
        self.assertFalse(required.issubset(recognized))
        self.assertNotIn(ord('F')+3,DESTINATION_INDICES)

if __name__=='__main__':unittest.main()
