"""Expected-defect diagnostics; passing these is not scientific qualification."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tests import standard_checks as C
from tests import test_one_standard as O


class QualificationSchemaDiagnostics(unittest.TestCase):
    def declaration(self):
        return dict(arms={'llm': {}, 'random': {}}, seeds=[1],
                    seeds_declared_before_run=True, budget_per_arm=0,
                    niches=[], founders=[], genomes=[], proposals=[],
                    scientific_acceptance=False)

    def test_three_semantic_paths_raise_same_missing_evaluator(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'STANDARD.json').write_text(json.dumps(self.declaration()))
            for check in (C.check_accuracy_recomputed,
                          C.check_llm_vs_random_recomputed,
                          C.check_router_present_and_matters):
                with self.subTest(check=check.__name__):
                    with self.assertRaisesRegex(KeyError, 'evaluator'):
                        check(d)

    def test_empty_population_structural_checks_are_vacuous(self):
        case = O.ContractOfEveryWeightsRun()
        with patch.object(case, 'runs', return_value=[('fixture', self.declaration())]):
            case.test_required_fields()
            case.test_founders_are_llm_written_literal_weights()
            case.test_genomes_are_weights_with_valid_lineage()
            case.test_mutation_origin_matches_arm()

    def test_discovery_ignores_weights_directory_without_standard(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'weights-missing').mkdir()
            p = Path(d, 'weights-present'); p.mkdir()
            (p / 'STANDARD.json').write_text(json.dumps(self.declaration()))
            with patch.object(O, 'RUNS', d):
                self.assertEqual(O.standard_runs(), [str(p / 'STANDARD.json')])


if __name__ == '__main__':
    unittest.main()
