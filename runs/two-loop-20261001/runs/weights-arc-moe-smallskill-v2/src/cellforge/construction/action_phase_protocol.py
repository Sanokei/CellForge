"""Prospective cell-only semantic operators; no full-architecture acceptance."""
import copy, math, random
import numpy as np
from .joint_weight_population import digest
from .anchored_action_protocol import validate_snapshot, resolve_live_router
from .action_population import ActionCellPopulation, ActionOpponentHistory
from .cell_population import validate_literal

FEATURES=("q1","q2","prev_for","is_vocab","has_dot_evenquoted","prev_to","prev_in","prev_readverb",
          "is_function_word","prev_write","prev_replace","prev_with")
FAMILY="action-arguments-lexical"

def argument_caps(params, limit=9):
    validate_literal(params)
    if set(params)!={"argument.W","argument.b"}:raise ValueError("exact argument tensors")
    w=np.asarray(params["argument.W"]);b=np.asarray(params["argument.b"])
    if w.shape!=(3,len(FEATURES)) or b.shape!=(3,):raise ValueError("declared argument interface shape")
    if max(float(np.abs(w).max()),float(np.abs(b).max()))>limit:raise ValueError("parameter resource cap")
    return {"passed":True,"positions":int(w.size+b.size),"maximum_absolute_weight":max(float(np.abs(w).max()),float(np.abs(b).max()))}

def restore(snapshot):
    validate_snapshot(snapshot)
    pop=ActionCellPopulation()
    for name in ("cells","live","events","charges"):setattr(pop,name,copy.deepcopy(snapshot[name]))
    pop.histories={k:ActionOpponentHistory(copy.deepcopy(v)) for k,v in snapshot["histories"].items()}
    return pop

def bind_author(response,prompt,recipient,feedback):
    """Applied literals must match raw response, current parent and computed DEV feedback."""
    if response.get("model")!="gpt-6.1-sol" or response.get("reasoning")!="low":
        raise ValueError("requested maker provenance required")
    if response.get("parent_id")!=recipient["id"] or response.get("parent_params")!=recipient["params"]:
        raise ValueError("raw maker parent mismatch")
    if prompt.get("parent_id")!=recipient["id"] or prompt.get("parent_params")!=recipient["params"]:
        raise ValueError("prompt recipient mismatch")
    if prompt.get("development_evidence")!=feedback or prompt.get("partition")!="development":
        raise ValueError("prompt differs from recomputed development evidence")
    if set(prompt)!={"parent_id","parent_params","development_evidence","partition","feature_order","role_order","edit_contract"}:
        raise ValueError("exact development-only maker prompt")
    if any(response.get(k)!=list(v) or prompt.get(k)!=list(v) for k,v in
           (("feature_order",FEATURES),("role_order",("arg1","arg2","file")))):
        raise ValueError("typed feature/role order mismatch")
    if response.get("prompt_sha256")!=digest(prompt) or response.get("feedback_sha256")!=digest(feedback):
        raise ValueError("prompt or counterexample commitment mismatch")
    params=copy.deepcopy(response["tensors"])
    cap=argument_caps(params)
    contract=prompt["edit_contract"]
    if contract!={"positions":39,"range":[-9,9],"operator":"literal_full_matrix","interface":"tool-action-joint/2"}:
        raise ValueError("declared operator vocabulary mismatch")
    if response.get("charged_positions")!=cap["positions"]:raise ValueError("full authoring budget mismatch")
    # Fixed-width arm cannot smuggle structural mutation under the same family name.
    if response.get("interface")!="tool-action-joint/2":raise ValueError("undeclared IO mutation")
    if params==recipient["params"]:raise ValueError("no actual numeric revision")
    return params,cap

def validate_scores(scores):
    for name in ("fresh","archive","anchor"):
        row=scores[name]
        if row["partition"]!="development":raise ValueError("reporting used for selection")
        if not row["outcomes"] or any(type(v) is not bool for v in row["outcomes"].values()):
            raise ValueError("executed Boolean development outcomes required")
        rate=row["false_positive_given_consistency"]
        if type(rate) not in (int,float) or not math.isfinite(rate) or not 0<=rate<=1:
            raise ValueError("computed false-action rate required")
        traces=row.get("traces")
        if not isinstance(traces,list) or not traces or {t["id"]:t["correct"] for t in traces}!=row["outcomes"]:
            raise ValueError("outcome labels differ from executed traces")
        if any(type(t["files_correct"]) is not bool or type(t["correct"]) is not bool for t in traces):
            raise ValueError("Boolean fixture traces required")
    computed=sum(not t["files_correct"] for t in scores["anchor"]["traces"])/len(scores["anchor"]["traces"])
    if any(row["false_positive_given_consistency"]!=computed for row in scores.values()):
        raise ValueError("fixture false-action rate differs from actual effects")
    return True

def apply_phase(populations,seed,phase,response,prompt,feedback,evaluate,journal):
    """Each arm starts from its OWN survivor. Journal callback runs before each admission."""
    if set(populations)!={"llm","random"}:raise ValueError("both persistent arms required")
    result={}
    for arm in ("llm","random"):
        pop=restore(populations[arm]);recipient_id=pop.live[FAMILY][0];recipient=pop.cells[recipient_id]
        before=copy.deepcopy(recipient)
        raw_params=response.get("tensors") if arm=="llm" else None
        candidate=None
        if raw_params is not None:
            try:candidate=digest({"structure":FAMILY,"params":raw_params})
            except (ValueError,TypeError):pass
        if arm=="llm":
            # Charge attempted authoring BEFORE binding, shape rejection or evaluator.
            journal({"seed":seed,"phase":phase,"arm":arm,"parent":recipient_id,"positions":39,
                     "candidate":candidate,"charged":True})
            params,cap=bind_author(response,prompt,recipient,feedback)
            provenance={"raw_response":digest(response),"model":response["model"],"prompt":digest(prompt),
              "feedback":digest(feedback),"newly_authored":"39 literal positions","typed_io":"ordered 12 features, 3 copied spans"}
        else:
            rng=random.Random(seed+phase*1000);v=[rng.uniform(-9,9) for _ in range(39)]
            params={"argument.W":[v[i*12:(i+1)*12] for i in range(3)],"argument.b":v[36:]}
            cap=argument_caps(params);provenance={"source":"independent_arm_literal_random","seed":seed,"phase":phase}
        # Durable caller journal BEFORE evaluator/admission, including invalid candidates.
        charge={"seed":seed,"phase":phase,"arm":arm,"parent":recipient_id,"positions":39,
                "candidate":digest({"structure":FAMILY,"params":params}),"charged":True}
        if arm=="random":journal(charge)
        scores=evaluate(params)
        fixtures=validate_scores(scores)
        decision=pop.birth(FAMILY,params,[recipient_id],provenance,scores,
                           caps_passed=cap["passed"],fixtures_passed=fixtures,arm=arm)
        validate_snapshot(pop.snapshot())
        result[arm]={"parent":before["id"],"parent_tensor_hash":digest(before["params"]),
                    "decision":decision,"caps":cap,"fixtures":fixtures,"population":pop.snapshot()}
    return result

def validate_phase_receipt(expected,stored,charges):
    if expected!=stored:raise ValueError("executed lineage/selection replay mismatch")
    expectedcharges=[]
    for context in expected["contexts"]:
        for arm in ("llm","random"):
            row=context["arms"][arm]
            expectedcharges.append({"seed":context["seed"],"phase":expected["phase"],"arm":arm,
              "parent":row["parent"],"positions":39,
              "candidate":row["population"]["charges"][-1]["child"],"charged":True})
    if expectedcharges!=charges:raise ValueError("dropped or fabricated charged proposal")
    return True
