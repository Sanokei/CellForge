"""Run unchanged contract class against the declared local preview root.
This tests schema/discovery; it is not the full semantic/milestone qualification.
"""
import importlib.util
import pathlib
import unittest
ROOT=pathlib.Path(__file__).resolve().parent
def module(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/"contract-reference"/filename)
    result=importlib.util.module_from_spec(spec); spec.loader.exec_module(result); return result
def main():
    contract=module("original_contract","test_one_standard.py")
    contract.RUNS=str(ROOT/"preregistration") # explicit fixture root; source file is unchanged
    discovered=contract.standard_runs()
    if len(discovered)!=1: raise RuntimeError("standard scanner did not find exactly the proposed weights-* manifest")
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(contract.ContractOfEveryWeightsRun))
    semantic=module("original_semantic","standard_checks.py")
    run=str(pathlib.Path(discovered[0]).parent)
    for check in (semantic.check_hash_integrity,semantic.check_ledger_chain,semantic.check_budget_reconciled):
        findings=check(run)
        print(check.__name__+": "+("PASS" if not findings else repr(findings)))
        if findings: raise RuntimeError("contract artifact reconciliation failed")
    print("Only schema, hash, ledger and charge checks tested. Full qualification remains RED/UNQUALIFIED.")
    return 0 if result.wasSuccessful() else 1
if __name__=="__main__": raise SystemExit(main())

