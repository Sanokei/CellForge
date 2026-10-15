"""Bounded discrete parameter evolution and matched founder random search.

No derivatives, finite differences or gradient estimators. Count-fitted seeds
carry explicit development provenance; primary proposals are discrete and typed.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.governance.ledger import Ledger
from cellforge.reference.language_curriculum import evaluate_language_curriculum, validate_partition_manifests


def reject_adaptive_mixer(graph):
    """Primary owner policy excludes multiplicative posterior/Hedge updates."""
    if 'adaptive_mixer_recipe' in graph.meta:
        raise ValueError('adaptive multiplicative Bayesian mixer excluded from primary no-gradient arms')
    for name, definition in graph.defs.items():
        node_ids={str(node.get('id','')).split('.')[-1] for node in definition.get('nodes',[])}
        if {'posterior','posterior_kept','prior_shrunk','weights_updated'}.issubset(node_ids):
            raise ValueError('known multiplicative posterior module excluded from primary no-gradient arms')
        labels=[name]+[str(node.get('id',''))+' '+str(node.get('use','')) for node in definition.get('nodes',[])]
        if any('mixer.posterior' in label or 'multiplicative' in label.lower() for label in labels):
            raise ValueError('adaptive mixer posterior/multiplicative module excluded from primary no-gradient arms')


def domain_parameters(graph, island):
    roots = ["router"] if island == 0 else [n for n in graph.defs if n.startswith("ex_")]
    seen, names = set(), set()
    def visit(name):
        if name in seen:
            return
        seen.add(name)
        names.update(graph.params_of(name))
        for child in graph.children(name):
            visit(child)
    for name in roots:
        if name in graph.defs:
            visit(name)
    return sorted(names or graph.params)


def mutate(parent, island, seed):
    rng = np.random.default_rng(seed)
    names = domain_parameters(parent,island)
    nonzero = rng.random() < .75
    eligible = [(n,np.flatnonzero((parent.params[n].ravel()!=0) if nonzero else (parent.params[n].ravel()==0))) for n in names]
    eligible = [(n,indices) for n,indices in eligible if len(indices)]
    if not eligible:
        eligible = [(n,np.arange(parent.params[n].size)) for n in names if parent.params[n].size]
    name,indices = eligible[int(rng.integers(len(eligible)))]
    index = int(indices[int(rng.integers(len(indices)))])
    old = float(parent.params[name].ravel()[index])
    delta = float(rng.choice([-.5,-.125,-.03125,.03125,.125,.5]))
    child = parent.copy()
    child.params[name].ravel()[index] = old+delta
    proposal = {"protocol":"discrete-coordinate-mutation/1","parent_hash":parent.hash(),
                "seed":seed,"island":island,"domain":"router ancestry" if island==0 else "expert ancestries",
                "parameter":name,"index":index,"old":old,"new":old+delta,
                "operator":"reachable nonzero75% / zero activation25%, independent fixed delta palette",
                "gradient_estimator":False}
    child.meta["evolution_parent_hash"] = proposal["parent_hash"]
    child.meta["evolution_seed"] = seed
    child.validate()
    proposal["child_hash"] = child.hash()
    return child,proposal


def legacy_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact",required=True)
    parser.add_argument("--private",required=True)
    parser.add_argument("--public",required=True)
    parser.add_argument("--calibration",required=True)
    args = parser.parse_args()
    private,public = Path(args.private).resolve(),Path(args.public)
    root = private/"parameter-evolution-v1"
    if private.parent != Path("/srv/cellforge-private") or root.exists() or public.exists():
        raise SystemExit("fresh private pilot required")
    root.mkdir(mode=0o700)
    founder = Graph.load(args.artifact)
    selection_raw = (private/"selection.json").read_bytes()
    selection = json.loads(selection_raw)
    calibration = json.loads(Path(args.calibration).read_bytes())
    if calibration["policy"]["founder_hash"] != founder.hash() or calibration["policy"]["selection_sha256"] != hashlib.sha256(selection_raw).hexdigest():
        raise SystemExit("calibration/model/selection mismatch")
    policy = {"protocol":"discrete-language-islands/1","founder_hash":founder.hash(),
        "islands":2,"population_slots_per_island":4,"initial_population":"four founder copies per island; initial diversity zero",
        "generations":3,"offspring_per_island_generation":2,"proposals_per_arm":12,
        "arms":["evolution_selected_parents","random_search_founder_parents"],
        "mutation":"same75%reachable nonzero/25%zero activation and fixed signed delta palette; common seed per arm/island/generation/proposal",
        "selection":"eligible minimum content bpb, tie preserve older candidate then hash; elitist top4",
        "stage":"parameter-coordinate evolution only, no parameter-block/module crossover or migration in first pilot",
        "selection_sha256":hashlib.sha256(selection_raw).hexdigest(),"report_policy":"report-only pilot data loaded only after all3generations andwinnerhashes frozen; not final/unexposed data",
        "metric":"UTF8 content-only bits per byte, EOS excluded; quality and MC accuracy not fitness",
        "gradient_steps":0,"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "evaluator_sha256":hashlib.sha256(Path(__import__('inspect').getfile(evaluate_language_curriculum)).read_bytes()).hexdigest(),
        "exposure_limitations":"whole exact-textSHA families disjoint; public dev_train source; near-family/topic overlap not audited"}
    (root/"policy.json").write_text(json.dumps(policy,sort_keys=True))
    ledger = Ledger(root/"ledger.jsonl")
    ledger.append("policy_frozen",policy,"evolution-operator")
    cache = {founder.hash():calibration["fitness"]}
    graphs = {founder.hash():founder}
    births = {founder.hash():0}
    models = root/"models"
    founder.save(models/founder.hash())
    started = time.monotonic()
    evaluations = 0
    def score(graph):
        nonlocal evaluations
        key = graph.hash()
        if key not in cache:
            cache[key] = evaluate_language_curriculum(graph,selection,include_raw=False)
            evaluations += 1
            ledger.append("selection_evaluation",{"hash":key,"fitness":cache[key]},"evolution-operator")
        return float(cache[key]["fitness_bits_per_byte"]) if cache[key]["eligible_for_selection"] else float("inf")
    def rank(graph):
        return (score(graph),births[graph.hash()],graph.hash())
    populations = {arm:[[founder]*4 for _ in range(2)] for arm in policy["arms"]}
    curves = {arm:[{"generation":0,"winner_hash":founder.hash(),"selection_bpb":score(founder)}] for arm in populations}
    winners = {arm:[founder.hash()] for arm in populations}
    def publish(status):
        output = {"protocol":policy["protocol"],"status":status,"selection_curves":curves,
                  "selection_evaluations":evaluations,"elapsed_seconds":time.monotonic()-started,
                  "report_evaluations":0,"final_evaluations":0,"gradient_steps":0}
        temporary = public.with_suffix(".partial")
        temporary.write_text(json.dumps(output,indent=2)+"\n")
        temporary.chmod(0o644)
        temporary.replace(public)
    publish("running")
    for generation in range(1,4):
        for arm in populations:
            for island in range(2):
                pool = populations[arm][island]
                children = []
                for proposal_index in range(2):
                    seed = 628000+generation*100+island*10+proposal_index
                    parent = min(pool,key=rank) if arm=="evolution_selected_parents" else founder
                    child,proposal = mutate(parent,island,seed)
                    graphs[child.hash()] = child
                    births.setdefault(child.hash(),generation)
                    if not (models/child.hash()).exists():
                        child.save(models/child.hash())
                    proposal.update({"arm":arm,"generation":generation,"proposal_index":proposal_index})
                    ledger.append("mutation_proposal",proposal,"evolution-operator")
                    score(child)
                    children.append(child)
                populations[arm][island] = sorted(pool+children,key=rank)[:4]
            best = min([g for pool in populations[arm] for g in pool],key=rank)
            winners[arm].append(best.hash())
            curves[arm].append({"generation":generation,"winner_hash":best.hash(),"selection_bpb":score(best)})
            ledger.append("generation_winner_frozen",{"arm":arm,"generation":generation,"hash":best.hash(),"selection_bpb":score(best)},"evolution-operator")
        publish("running")
    (root/"all-winners-frozen.json").write_text(json.dumps(winners,sort_keys=True))
    ledger.append("all_winners_frozen",winners,"evolution-operator")
    # No report item has been read by this process before this point.
    report_raw = (private/"report-only.json").read_bytes()
    report = json.loads(report_raw)
    validate_partition_manifests({"partition":"selection","items":selection},
                                 {"partition":"report","items":report})
    report_cache = {}
    report_curves = {}
    for arm,hashes in winners.items():
        report_curves[arm] = []
        for generation,key in enumerate(hashes):
            if key not in report_cache:
                report_cache[key] = evaluate_language_curriculum(graphs[key],report,include_raw=False)
            report_curves[arm].append({"generation":generation,"hash":key,"report":report_cache[key]})
    result = {"protocol":policy["protocol"],"status":"complete_bounded_pilot","policy":policy,
        "selection_curves":curves,"report_only_curves":report_curves,"selection_evaluations":evaluations,
        "unique_report_evaluations":len(report_cache),"report_sha256":hashlib.sha256(report_raw).hexdigest(),
        "elapsed_seconds":time.monotonic()-started,"ledger_head":ledger.head(),"gradient_steps":0,"final_evaluations":0,
        "limitations":"tiny parameter-evolution pilot, no module breeding/gradient estimator, no unexposed final or frontier comparison; report curves never used for selection"}
    (root/"receipt.json").write_text(json.dumps(result,indent=2)+"\n")
    temporary = public.with_suffix(".partial")
    temporary.write_text(json.dumps(result,indent=2)+"\n")
    temporary.chmod(0o644)
    temporary.replace(public)


OPERATORS = ("row_scale", "block_scale", "expert_slice", "router_gate", "typed_module", "trunk_tile", "mixing_gate", "recipe_update", "recipe_transfer")


def selection_items(manifest):
    """Exporter dictionaries or explicitly recorded legacy selection lists."""
    if isinstance(manifest,dict):
        if manifest.get('partition') not in ('selection','development','dev_train_selection'):
            raise ValueError('report/audit/final or unlabelled manifests cannot select candidates')
        items=manifest.get('items')
    else:items=manifest
    if not isinstance(items,list) or not items or any(not isinstance(i,dict) or i.get('kind')!='next_byte' for i in items):
        raise ValueError('nonempty next-byte development selection items required')
    return items


def mapped_operator(graph,scheduled):
    """Predeclared contract fallbacks; never select a mapping by measured loss."""
    fitted=bool(graph.meta.get('fitted_provenance'))
    if scheduled in ('recipe_update','recipe_transfer') and not fitted:
        return {'recipe_update':'block_scale','recipe_transfer':'expert_slice'}[scheduled],'pure founder has no fitted recipe'
    if scheduled=='trunk_tile' and not any(n.startswith('fitted_ngram.tile_') for n in graph.params):
        if fitted and any(n.startswith('cfwb.') for n in graph.params):
            return 'mixing_gate','WB CSR trunk stays immutable; activate authored residual gate instead of unavailable legacy tiles'
        return 'block_scale','no legacy fitted tiles; explicit authored expert block fallback'
    if scheduled=='mixing_gate':
        gates=[node for definition in graph.defs.values() for node in definition.get('nodes',[])
               if node['use']=='op:scale' and any(tag in node['id'] for tag in ('mix','residual','cache'))]
        if not gates:return 'block_scale','no declared mixing gate; explicit authored expert block fallback'
    return scheduled,None


def archive_probabilities(archive):
    """Discrete performance/novelty ranks plus uniform mass, including failures."""
    if not archive: raise ValueError('nonempty archive required')
    n=len(archive)
    scores=np.array([a.get('performance',float('inf')) for a in archive])
    novelty=np.array([a.get('novelty',0.) for a in archive])
    def ranked(values):
        return np.array([1+np.count_nonzero(values<value) for value in values],dtype=float)
    perf=ranked(-scores); novel=ranked(novelty)
    probabilities=.1/n+.6*perf/perf.sum()+.3*novel/novel.sum()
    return probabilities/probabilities.sum()


def parameter_novelty(candidate, previous):
    """Nearest normalized parameter distance, independent of corpus outputs."""
    distances=[]
    for graph in previous:
        common=[n for n in candidate.params if n in graph.params and candidate.params[n].shape==graph.params[n].shape]
        values=[float(np.mean((candidate.params[n]-graph.params[n])**2))/(1+float(np.mean(graph.params[n]**2))) for n in common]
        definitions=set(candidate.defs)|set(graph.defs)
        structure=sum(candidate.defs.get(n)!=graph.defs.get(n) for n in definitions)/max(1,len(definitions))
        distances.append((float(np.sqrt(np.mean(values))) if values else 1.)+structure)
    return min(distances) if distances else 1.


def recipe_specs(graph):
    specs=graph.meta.get('evolution_recipe_parameters',[])
    if not specs and graph.meta.get('adaptive_mixer_recipe'):
        specs=[{'parameter':name,'minimum':1e-9 if name.endswith('.prior') else 0.,
                'maximum':4. if name.endswith('.eta') else 1.,
                'normalization':'simplex' if name.endswith('.prior') else None}
               for name in sorted(graph.params) if name.endswith(('.eta','.decay','.prior'))]
    if not specs: raise ValueError('exporter-declared low-dimensional recipe parameters required')
    for spec in specs:
        name=spec['parameter']
        if name not in graph.params or graph.params[name].size>64:
            raise ValueError('recipe parameters must be explicit arrays of at most64 slots')
        if any(part in name.lower() for part in ('keys','offsets','columns')):
            raise ValueError('structural CSR arrays forbidden')
    return specs


def adaptive_probabilities(stats, floor=.05):
    """Smoothed discrete acceptance rates; no loss derivative or estimator."""
    if not 0 < floor < 1/len(OPERATORS):
        raise ValueError("nonzero floor below uniform probability required")
    rates = np.array([(stats.get(op, {}).get("accepted", 0)+1)/
                      (stats.get(op, {}).get("tried", 0)+2) for op in OPERATORS])
    return floor+(1-len(OPERATORS)*floor)*rates/rates.sum()


def structured_mutate(parent, donor, operator, seed):
    """Structured proposal with exact lineage. Runtime scoring admits shapes."""
    from cellforge.construction.language_crossover import crossover_module
    rng = np.random.default_rng(seed)
    child = parent.copy()
    details = {}
    experts = sorted(n for n in parent.defs if n.startswith("ex_") and parent.defs[n]["kind"]=="bundle")
    if operator in ('recipe_update','recipe_transfer'):
        specs=recipe_specs(parent); spec=specs[int(rng.integers(len(specs)))]; name=spec['parameter']
        array=child.params[name]; old=array.copy()
        lower=float(spec.get('minimum',0)); upper=float(spec.get('maximum',1))
        if operator=='recipe_transfer':
            if name not in donor.params or donor.params[name].shape!=array.shape: raise ValueError('recipe donor mismatch')
            array[...] = donor.params[name]
        else:
            index=int(rng.integers(array.size)); delta=float(rng.choice([-.1,-.025,.025,.1]))*(upper-lower)
            array.reshape(-1)[index]+=delta
        array[...] = np.clip(array,lower,upper)
        if spec.get('normalization')=='simplex':
            if array.sum()<=0: raise ValueError('positive recipe simplex required')
            array[...] /= array.sum()
        if not np.isfinite(array).all(): raise ValueError('nonfinite recipe')
        details={'parameter':name,'old':old.tolist(),'new':array.tolist(),'constraint':spec,'donor_hash':donor.hash() if operator=='recipe_transfer' else None}
    elif operator=='mixing_gate':
        gates=[(name,node['id']) for name,definition in child.defs.items() for node in definition.get('nodes',[])
               if node['use']=='op:scale' and any(tag in node['id'] for tag in ('mix','residual','cache'))
               and isinstance(node.get('args',{}).get('c'),(int,float))]
        if not gates: raise ValueError('no declared mixing scale gate')
        name,nodeid=gates[int(rng.integers(len(gates)))]; node=next(n for n in child.defs[name]['nodes'] if n['id']==nodeid)
        old=float(node['args']['c']); delta=float(rng.choice([-.05,-.01,.01,.05])); new=float(np.clip(old+delta,0,1))
        node['args']['c']=new
        details={'definition':name,'node':nodeid,'old':old,'new':new,'operator':'additive bounded gate including zero activation'}
    elif operator=="trunk_tile":
        names=sorted(n for n in parent.params if n.startswith('fitted_ngram.tile_'))
        if not names: raise ValueError('no fitted trunk; unavailable operator remains recorded rejection')
        name=names[int(rng.integers(len(names)))]; factor=float(rng.choice([.875,.9375,1.0625,1.125]))
        child.params[name] *= factor
        details={'parameter':name,'tile_size':int(child.params[name].size),'factor':factor,'domain':'fitted count table; no recounting'}
    elif operator in ("row_scale", "block_scale"):
        names = [n for n in domain_parameters(parent,1) if parent.params[n].size and np.any(parent.params[n])]
        if not names: raise ValueError("no active expert arrays")
        name = names[int(rng.integers(len(names)))]; array = child.params[name]
        factor = float(rng.choice([.75,.875,1.125,1.25]))
        if operator=="row_scale":
            row = int(rng.integers(array.shape[0])) if array.ndim else 0
            if array.ndim: array[row] *= factor
            else: array[...] *= factor
            details = {"parameter":name,"row":row,"factor":factor}
        else:
            flat=array.reshape(-1); size=min(len(flat),max(2,int(np.sqrt(len(flat)))))
            start=int(rng.integers(len(flat)-size+1)); flat[start:start+size] *= factor
            details={"parameter":name,"start":start,"size":size,"factor":factor}
    elif operator=="expert_slice":
        names=domain_parameters(parent,1)
        compatible=[n for n in names if n in donor.params and donor.params[n].shape==parent.params[n].shape and parent.params[n].size]
        if not compatible: raise ValueError("no compatible expert parameter slices")
        name=compatible[int(rng.integers(len(compatible)))]; flat=child.params[name].reshape(-1)
        size=min(len(flat),max(2,int(np.sqrt(len(flat))))); start=int(rng.integers(len(flat)-size+1))
        flat[start:start+size]=donor.params[name].reshape(-1)[start:start+size]
        details={"parameter":name,"start":start,"size":size,"donor_hash":donor.hash(),"matching":"same parameter identity and exact shape"}
    elif operator=="router_gate":
        reachable=set(); pending=["router"]
        while pending:
            name=pending.pop()
            if name in reachable or name not in parent.defs: continue
            reachable.add(name); pending.extend(parent.children(name))
        gates=[(name,node["id"]) for name in sorted(reachable) for node in child.defs[name].get("nodes",[])
               if node["use"]=="op:scale" and isinstance(node.get("args",{}).get("c"),(int,float))]
        if not gates: raise ValueError("no numeric router scale gates")
        name,nodeid=gates[int(rng.integers(len(gates)))]; node=next(n for n in child.defs[name]["nodes"] if n["id"]==nodeid)
        old=float(node["args"]["c"]); factor=float(rng.choice([.75,.875,1.125,1.25])); node["args"]["c"]=old*factor
        details={"definition":name,"node":nodeid,"argument":"c","old":old,"new":old*factor}
    elif operator=="typed_module":
        candidates=[n for n in experts if n in donor.defs]
        if not candidates: raise ValueError("no compatible expert modules")
        target=candidates[int(rng.integers(len(candidates)))]
        child,details=crossover_module(parent,donor,target,target,f"evo_{seed}_")
    else: raise ValueError("unknown operator")
    child.validate()
    lineage={"operator":operator,"seed":seed,"parent_hash":parent.hash(),"donor_hash":donor.hash(),"details":details,"gradient_estimator":False}
    child.meta["structured_evolution"]=lineage
    return child,dict(lineage,child_hash=child.hash())


def accept_two_batches(parent_scores, child_scores):
    return all(c["eligible_for_selection"] and p["eligible_for_selection"] and
               np.isfinite(c.get('composite_bpb',c["fitness_bits_per_byte"])) and
               c.get('composite_bpb',c["fitness_bits_per_byte"]) < p.get('composite_bpb',p["fitness_bits_per_byte"])-1e-10 and
               c['fitness_bits_per_byte'] <= p['fitness_bits_per_byte']*1.01
               for p,c in zip(parent_scores,child_scores)) and len(parent_scores)==len(child_scores)==2


def main():
    process_started=time.monotonic()
    parser=argparse.ArgumentParser(description="Stage3 structured DEV-only evolution; no server launch")
    parser.add_argument("--artifact",required=True); parser.add_argument("--batch-a",required=True)
    parser.add_argument("--fitted-artifact",required=True)
    parser.add_argument("--batch-b",required=True); parser.add_argument("--out",required=True)
    parser.add_argument('--context-a'); parser.add_argument('--context-b')
    parser.add_argument('--qualified-fast-fitness',action='store_true',help='operator confirms independent equivalence and throughput qualification')
    parser.add_argument('--operator-plan',choices=('recipe','structured'),default='structured')
    parser.add_argument("--steps",type=int,default=10); parser.add_argument("--seed",type=int,default=93001)
    args=parser.parse_args()
    if not args.context_a or not args.context_b:
        raise ValueError('both fixed context manifests required for composite selection')
    if not args.qualified_fast_fitness:
        raise ValueError('fast masked fitness must be independently qualified before running')
    if not 1<=args.steps<=100: raise ValueError("bounded steps required")
    output=Path(args.out)
    if output.exists(): raise ValueError("fresh output required")
    raw=[Path(p).read_bytes() for p in (args.batch_a,args.batch_b)]
    batch_manifests=[json.loads(r) for r in raw]
    batches=[selection_items(m) for m in batch_manifests]
    context_raw=[Path(p).read_bytes() for p in (args.context_a,args.context_b)]
    context_manifests=[json.loads(r) for r in context_raw]
    contexts=[selection_items(m) for m in context_manifests]
    if any(not c or any(not i.get('score_byte_ranges') for i in c) for c in contexts):
        raise ValueError('nonempty masked context targets required')
    if any(any(i.get('dependency_distance_bytes',0)<=8 for i in c) for c in contexts):
        raise ValueError('each context target requires declared dependency beyond eight bytes')
    if any(any(i.get('kind')!='next_byte' for i in b) for b in batches): raise ValueError("next-byte DEV batches required")
    validate_partition_manifests({"partition":"selection","items":batches[0]}, {"partition":"report","items":batches[1]})
    validate_partition_manifests({'partition':'selection','items':batches[0]+contexts[0]},
                                 {'partition':'report','items':batches[1]+contexts[1]})
    # The second label above invokes a disjointness validator only; BOTH batches are DEV selection.
    if args.operator_plan=='recipe':
        raise ValueError('recipe plan excluded: adaptive mixer has no primary owner authorization')
    founder=Graph.load(args.artifact); fitted=Graph.load(args.fitted_artifact)
    reject_adaptive_mixer(founder); reject_adaptive_mixer(fitted)
    for key in ('vocab','text_codec','ctx_len','state_dim'):
        if founder.meta.get(key)!=fitted.meta.get(key): raise ValueError('seed contract mismatch')
    if not fitted.meta.get('fitted_provenance'): raise ValueError('fitted seed provenance required')
    parent=fitted; pure_parent=founder; random_best=fitted
    stats={op:{"tried":0,"accepted":0} for op in OPERATORS}; random_stats=json.loads(json.dumps(stats))
    cache={}; pure_stats={op:{"tried":0,"accepted":0} for op in OPERATORS}
    evaluated_bytes=0; evaluations=0; started=time.monotonic(); records=[]; evaluation_curve=[]
    from cellforge.reference.evolution_fitness import EvolutionFitness
    scorers={}
    def failed_score(graph,items,exc):
        selected=sum(sum(stop-start for start,stop in item.get('score_byte_ranges',[[0,len(item['text'].encode('utf8'))]])) for item in items)
        full=sum(len(item['text'].encode('utf8')) for item in items)
        penalty=max(8.,float(np.log2(len(graph.meta['vocab']))+1))
        return {'fitness_bits_per_byte':penalty,'eligible_for_selection':False,
                'next_byte':{'content_bytes':selected,'content_bits':penalty*selected,'failed_documents':len(items)},
                'timing':{'evaluated_bytes':0,'reserved_full_prefix_bytes':full+len(items)},
                'failure_type':type(exc).__name__,'failure_message':str(exc)[:2000]}
    def score(graph,arm='internal',step=None):
        nonlocal evaluated_bytes,evaluations
        key=graph.hash()
        cache_hit=key in cache;tick=time.monotonic()
        if key not in cache:
            seed_key=founder.hash() if not graph.meta.get('fitted_provenance') else fitted.hash()
            cache[key]=[]
            for batch_index,(corpus_scorer,context_scorer) in enumerate(scorers[seed_key]):
                try:corpus=corpus_scorer.evaluate(graph)
                except Exception as exc:corpus=failed_score(graph,batches[batch_index],exc)
                try:context=context_scorer.evaluate(graph)
                except Exception as exc:context=failed_score(graph,contexts[batch_index],exc)
                composite=dict(corpus)
                composite['composite_bpb']=.75*corpus['fitness_bits_per_byte']+.25*context['fitness_bits_per_byte']
                composite['eligible_for_selection']=corpus['eligible_for_selection'] and context['eligible_for_selection']
                composite['context_component']=context
                cache[key].append(composite)
                evaluated_bytes+=corpus.get('timing',{}).get('evaluated_bytes',corpus['next_byte']['content_bytes'])+context.get('timing',{}).get('evaluated_bytes',context['next_byte']['content_bytes'])
            evaluations+=4
        if arm!='internal':
            event={'evaluation_index':len(evaluation_curve),'arm':arm,'step':step,'graph_hash':key,'cache_hit':cache_hit,
                   'seconds':time.monotonic()-tick,'batches':cache[key]}
            evaluation_curve.append(event);ledger.append('fitness_request',event,'evolution-operator');publish('running')
        return cache[key]
    if args.operator_plan=='recipe': recipe_specs(fitted)
    def archive_entry(graph,failed=False):
        measured=score(graph)
        valid=all(x['eligible_for_selection'] and np.isfinite(x['composite_bpb']) for x in measured)
        failed=failed or not valid
        value=sum(x['composite_bpb'] for x in measured)/2 if valid and not failed else float('inf')
        return {'hash':graph.hash(),'performance':value,'novelty':float(len({graph.hash()})), 'failed':failed}
    graph_archive={founder.hash():founder,fitted.hash():fitted}
    policy={"protocol":"structured-evolution/3","batch_sha256":[hashlib.sha256(r).hexdigest() for r in raw],
            "both_batches":"fixed disjoint development selection, never report-only",
            "context_sha256":[hashlib.sha256(r).hexdigest() for r in context_raw],
            "operators":list(OPERATORS),"probability_floor":.05,"acceptance":"strict composite improvement BOTH batches epsilon1e-10; each corpus <=parent*1.01; all eligible",
            "operator_plan":args.operator_plan,"archive_selection":"0.6 discrete performance rank +0.3 nearest normalized parameter-distance novelty rank +0.1 uniform; failed entries retained with nonzero probability; no gradient estimators",
            "pure_operator_mapping":{"recipe_update":"block_scale","recipe_transfer":"expert_slice","limitation":"proposal budgets matched; pure/fitted mutation families differ because founder has no fitted mixer"},
            "structured_operator_mapping":{"WB6_trunk_tile":"mixing_gate/authored_residual; CSR keys/offsets/columns/values immutable in this pilot",
                "pure_unavailable_trunk_or_mixing":"block_scale; explicit per-proposal reason recorded"},
            "recipe_domains":"exporter-declared arrays<=64slots; adaptive mixer defaults eta[0,4],decay[0,1],positive normalized prior; CSR keys/offsets/columns never mutated",
            "islands":{"A":"router/expert/residual: row_scale block_scale expert_slice router_gate typed_module", "B":"fitted trunk/cache/mixing: trunk_tile mixing_gate"},
            "composite":"0.75 corpus content bpb +0.25 full-prefix conditional target-span bpb; fixed weights",
            "control":"four arms: fitted+evolution, fitted+random, fitted alone, founder+evolution; probability trajectory adapted from fitted evolution only and shared; fitted random parents always fitted seed; donors remain within seed lineage",
            "founder_hash":founder.hash(),"fitted_hash":fitted.hash(),"fitted_provenance":fitted.meta['fitted_provenance'],
            "report_evaluations":0,"gradient_steps":0,"seed":args.seed,"steps":args.steps,
            "equal_budget":{"proposals_per_search_arm":args.steps,'reserved_batch_slots_per_search_arm':4*args.steps,
                'cache_policy':'identical hashes share deterministic scores; actual cache hits and executed bytes recorded separately',
                'fitted_alone':'unchanged fitted seed, shared initial baseline; zero search proposals'},
            "manifest_formats":[m.get('partition') if isinstance(m,dict) else 'legacy unlabelled list, used only as explicit CLI selection' for m in batch_manifests+context_manifests],
            "fitness_source_sha256":hashlib.sha256(Path(__import__('inspect').getfile(EvolutionFitness)).read_bytes()).hexdigest(),
            "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    policy_path=output.with_name(output.stem+'-policy.json');ledger_path=output.with_name(output.stem+'-ledger.jsonl')
    progress_path=output.with_name(output.stem+'-progress.json')
    if any(p.exists() for p in (policy_path,ledger_path,progress_path)):raise ValueError('preserve prior policy/ledger/progress')
    output.parent.mkdir(parents=True,exist_ok=True)
    policy_path.write_text(json.dumps(policy,indent=2)+'\n')
    ledger=Ledger(ledger_path);ledger.append('policy_frozen',policy,'evolution-operator')
    def publish(status):
        progress={'protocol':policy['protocol'],'status':status,'policy_sha256':hashlib.sha256(policy_path.read_bytes()).hexdigest(),
                  'elapsed_seconds':time.monotonic()-started,'completed_steps':len(records),'records':records,
                  'evaluation_curve':evaluation_curve,'batch_evaluations':evaluations,'evaluated_content_bytes':evaluated_bytes,
                  'report_evaluations':0,'ledger_head':ledger.head()}
        temporary=progress_path.with_suffix('.partial');temporary.write_text(json.dumps(progress,indent=2)+'\n');temporary.replace(progress_path)
    publish('initializing_fixed_prefix_scorers')
    try:
        for seed_graph in (founder,fitted):
            scorers[seed_graph.hash()]=[(EvolutionFitness(seed_graph,b),EvolutionFitness(seed_graph,c)) for b,c in zip(batches,contexts)]
    except Exception as exc:
        ledger.append('initialization_failed',{'failure_type':type(exc).__name__,'failure_message':str(exc)[:2000]},'evolution-operator')
        publish('failed_initialization');raise
    score(founder,'founder_baseline',0);score(fitted,'fitted_alone',0)
    archives={'fitted':[archive_entry(fitted)],'pure':[archive_entry(founder)]}
    for step in range(args.steps):
        probs=adaptive_probabilities(stats); rng=np.random.default_rng(args.seed+step)
        island='A' if step%2==0 else 'B'
        allowed=(('recipe_update','recipe_transfer','mixing_gate') if args.operator_plan=='recipe' else
                 (('row_scale','block_scale','expert_slice','router_gate','typed_module') if island=='A' else ('trunk_tile','mixing_gate')))
        mask=np.array([op in allowed for op in OPERATORS])
        probs=np.where(mask,probs,0); probs/=probs.sum()
        op=str(rng.choice(OPERATORS,p=probs)); seed=args.seed+1000+step; donor=fitted
        record={"step":step,"island":island,"operator":op,"seed":seed,"chosen_probabilities":dict(zip(OPERATORS,map(float,probs))),"arms":{}}
        next_parent=parent; next_pure=pure_parent
        archive_choices={}; archive_donors={}
        for lineage in ('fitted','pure'):
            probabilities=archive_probabilities(archives[lineage]); choice=int(rng.choice(len(probabilities),p=probabilities))
            archive_choices[lineage]=graph_archive[archives[lineage][choice]['hash']]
            donor_index=int(rng.choice(len(probabilities),p=probabilities))
            archive_donors[lineage]=graph_archive[archives[lineage][donor_index]['hash']]
            record.setdefault('archive_parent_selection',{})[lineage]={'index':choice,'donor_index':donor_index,'probabilities':probabilities.tolist(),'entry':archives[lineage][choice]}
        pure_stats.setdefault(op,{"tried":0,"accepted":0})
        for arm,base,counters,arm_donor in (("fitted_evolution",archive_choices['fitted'],stats,archive_donors['fitted']),("fitted_random",fitted,random_stats,archive_donors['fitted']),("founder_evolution",archive_choices['pure'],pure_stats,archive_donors['pure'])):
            counters[op]["tried"]+=1
            applied_op,fallback=mapped_operator(base,op)
            try:
                child,lineage=structured_mutate(base,arm_donor,applied_op,seed)
                reject_adaptive_mixer(child)
                lineage['scheduled_operator']=op
                lineage['operator_fallback_reason']=fallback
                measured=score(child,arm,step); accepted=accept_two_batches(score(base),measured)
                counters[op]["accepted"]+=int(accepted)
                if arm=="fitted_evolution" and accepted and accept_two_batches(score(parent),measured): next_parent=child
                if arm=="founder_evolution" and accepted and accept_two_batches(score(pure_parent),measured): next_pure=child
                if arm=="fitted_random" and accept_two_batches(score(random_best),measured): random_best=child
                record["arms"][arm]={"lineage":lineage,"accepted_vs_parent":accepted,"scores":measured,'reserved_batch_slots':4}
                if arm!='fitted_random':
                    family='pure' if arm=='founder_evolution' else 'fitted'; entry=archive_entry(child,not all(x['eligible_for_selection'] for x in measured))
                    entry['novelty']=parameter_novelty(child,[graph_archive[x['hash']] for x in archives[family]])
                    archives[family].append(entry)
                graph_archive[child.hash()]=child
            except Exception as exc:
                record["arms"][arm]={"accepted_vs_parent":False,"failure_type":type(exc).__name__,
                    'failure_message':str(exc)[:2000],'scheduled_operator':op,'applied_operator':applied_op,'operator_fallback_reason':fallback,
                    'parent_hash':base.hash(),'reserved_batch_slots':4,
                    'charged_failure_scores':[dict(failed_score(base,batches[i],exc),context_component=failed_score(base,contexts[i],exc)) for i in range(2)]}
                ledger.append('proposal_failed',dict(record['arms'][arm],arm=arm,step=step),'evolution-operator')
                if arm!='fitted_random': archives['pure' if arm=='founder_evolution' else 'fitted'].append(dict(archive_entry(base,True),failure_type=type(exc).__name__,novelty=0.))
        parent=next_parent; pure_parent=next_pure; records.append(record)
        ledger.append('proposal_step_complete',record,'evolution-operator');publish('running')
    elapsed=time.monotonic()-started
    result={"policy":policy,"records":records,"archive":archives,"operator_metrics":stats,"random_operator_metrics":random_stats,"pure_operator_metrics":pure_stats,
            "evaluated_content_bytes":evaluated_bytes,"batch_evaluations":evaluations,"seconds":elapsed,"evaluations_per_second":evaluations/elapsed,
            'evaluation_curve':evaluation_curve,'proposal_counts':{arm:len(records) for arm in ('fitted_evolution','fitted_random','founder_evolution')},
            'reserved_batch_slots_per_search_arm':4*len(records),'report_evaluations':0,
            "selected_hash":parent.hash(),"random_best_hash":random_best.hash(),"selected_scores":score(parent),"random_best_scores":score(random_best),
            "fitted_alone_scores":score(fitted),"pure_scores":score(pure_parent),"pure_hash":pure_parent.hash(),
            "limitations":"development-only pilot; no frontier claim or report selection; fitted provenance explicit, pure donors isolated; donor sharing is explicit adaptive control policy"}
    for counters in (stats,random_stats,pure_stats):
        for values in counters.values(): values['acceptance_rate']=values['accepted']/values['tried'] if values['tried'] else None
    serialization_started=time.monotonic()
    parent.save(output.parent/(output.stem+'-winner'))
    for key,graph in graph_archive.items(): graph.save(output.parent/(output.stem+'-archive')/key)
    frozen={'fitted_evolution':parent.hash(),'fitted_random':random_best.hash(),'fitted_alone':fitted.hash(),'founder_evolution':pure_parent.hash()}
    ledger.append('all_winners_frozen',frozen,'evolution-operator')
    result['frozen_winners']=frozen;result['policy_sha256']=hashlib.sha256(policy_path.read_bytes()).hexdigest()
    result['ledger_entries_verified']=ledger.verify();result['ledger_head']=ledger.head()
    result['artifact_serialization_seconds']=time.monotonic()-serialization_started
    result['total_wall_seconds']=time.monotonic()-process_started
    result['throughput_scope']='search seconds include fixed-prefix construction, candidate compilation/hash/evaluation and live receipts; artifact serialization and total wall separate'
    output.write_text(json.dumps(result,indent=2)+'\n');publish('complete_bounded_development')


if __name__ == "__main__": main()

