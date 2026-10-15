import unittest,json,pathlib,copy
from cellforge.construction.circuit_population import CircuitArchive
from cellforge.construction.adhesion import AdhesionCandidate
ROOT=pathlib.Path(__file__).resolve().parents[1]
class TestCircuitPopulation(unittest.TestCase):
 def test_invalid_and_equivalent_proposals_are_charged_without_lineage_growth(self):
  p={"spec":{"topology":"x"},"new_numeric_positions":4}
  a=CircuitArchive({"topology":"baseline"},{"a":True,"b":False},["cmp","mux"],founder_weights={"cmp":{"W":[[1]],"b":[0]},"mux":{"W":[[1]],"b":[0]}})
  d=a.propose(p,valid=False,dev_outcomes={"a":False,"b":True},contribution=True,arm="llm")
  self.assertFalse(d["accepted"]);self.assertEqual(len(a.charges),1);self.assertEqual(len(a.history),1)
  p={"spec":{"topology":"valid"},"new_numeric_positions":4}
  d=a.propose(p,valid=True,dev_outcomes={"a":True,"b":True},contribution=True,arm="llm")
  self.assertTrue(d["accepted"]);self.assertEqual(len(a.history),2)
  equivalent={"spec":{"topology":"different-equivalent"},"new_numeric_positions":4}
  self.assertFalse(a.propose(equivalent,valid=True,dev_outcomes={"a":True,"b":True},contribution=True,arm="llm")["accepted"])
  self.assertEqual(len(a.history),2)
 def test_absent_contribution_cannot_earn_circuit_admission(self):
  a=CircuitArchive({"topology":"baseline"},{"a":False},["cmp","mux"],founder_weights={"cmp":{"W":[[1]],"b":[0]},"mux":{"W":[[1]],"b":[0]}})
  d=a.propose({"spec":{"topology":"changed"},"new_numeric_positions":4},valid=True,dev_outcomes={"a":True},contribution=False,arm="llm")
  self.assertFalse(d["accepted"]);self.assertEqual(len(a.history),1)
 def test_historical_regression_blocks_apparent_gain(self):
  a=CircuitArchive({"topology":"baseline"},{"a":True,"b":False,"c":False},["cmp","mux"],founder_weights={"cmp":{"W":[[1]],"b":[0]},"mux":{"W":[[1]],"b":[0]}})
  d=a.propose({"spec":{"topology":"new"},"new_numeric_positions":4},valid=True,dev_outcomes={"a":False,"b":True,"c":True},contribution=True,arm="llm")
  self.assertFalse(d["accepted"]);self.assertIn("historical_development_regression",d["reasons"])

 def test_genome_identity_binds_inherited_numerical_weights(self):
  first=CircuitArchive({"topology":"baseline"},{"a":False},["cmp","mux"],founder_weights={"cmp":{"W":[[1]],"b":[0]},"mux":{"W":[[1]],"b":[0]}})
  second=CircuitArchive({"topology":"baseline"},{"a":False},["cmp","mux"],founder_weights={"cmp":{"W":[[2]],"b":[0]},"mux":{"W":[[1]],"b":[0]}})
  self.assertNotEqual(first.live,second.live)
  with self.assertRaises(ValueError):
   CircuitArchive({"topology":"baseline"},{"a":False},["cmp","mux"],founder_weights={"cmp":{"W":[[1]],"b":[0]}})
