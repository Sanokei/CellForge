"""Development-only earned typed circuit archive; no ARC/generalization admission."""
import copy
from .joint_weight_population import digest
class CircuitArchive:
    def __init__(self, baseline, baseline_outcomes, founder_ids, *, founder_weights):
        self.founder_ids=list(founder_ids)
        from .cell_population import validate_literal
        if not self.founder_ids or len(set(self.founder_ids))!=len(self.founder_ids) or set(founder_weights)!=set(self.founder_ids):
            raise ValueError("exact independently selected founder identities and literal weights required")
        for params in founder_weights.values():validate_literal(params)
        self.founder_weights=copy.deepcopy(founder_weights)
        self.live=digest({"spec":baseline,"founders":self.founder_weights});self.records={self.live:{"spec":copy.deepcopy(baseline),"outcomes":dict(baseline_outcomes),"state":"bootstrap-control","parents":[]}}
        self.history=[self.live];self.events=[];self.charges=[]
    def propose(self,proposal,*,valid,dev_outcomes,contribution,arm):
        # Charge even malformed/duplicate proposals before admission.
        self.charges.append({"arm":arm,"positions":proposal["new_numeric_positions"],"charged":True,"proposal":digest(proposal)})
        current=self.live;cid=digest({"spec":proposal["spec"],"founders":self.founder_weights});reasons=[]
        if not valid:reasons.append("invalid_typed_transfer")
        if not contribution:reasons.append("unproved_numeric_group_contribution")
        if any(v and not dev_outcomes.get(k,False) for hid in self.history for k,v in self.records[hid]["outcomes"].items()):reasons.append("historical_development_regression")
        if sum(dev_outcomes.values())<=sum(self.records[current]["outcomes"].values()):reasons.append("strict_development_gain")
        if cid in self.records:reasons.append("duplicate")
        accepted=not reasons
        if cid not in self.records:
            self.records[cid]={"spec":copy.deepcopy(proposal["spec"]),"outcomes":dict(dev_outcomes),"state":"admitted" if accepted else "rejected",
              "parents":{"circuit":current,"cells":list(self.founder_ids)},"new_weight_positions":proposal["new_numeric_positions"]}
        self.events.append({"event":"proposal","id":cid,"parent":current,"accepted":accepted,"reasons":reasons})
        if accepted:
            self.records[current]["state"]="retired";self.live=cid;self.history.append(cid)
            self.events.append({"event":"admitted","id":cid,"retired":current})
        return {"accepted":accepted,"reasons":reasons,"child":cid,"parent":current}
    def snapshot(self):
        return {"live":self.live,"founder_weights":copy.deepcopy(self.founder_weights),"records":self.records,"historical_opponents":self.history,"events":self.events,"charges":self.charges}
