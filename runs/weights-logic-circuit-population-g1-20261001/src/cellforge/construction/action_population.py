"""Explicit action-fixture niche policy; never replaces ARC real-anchor admission."""
import copy
from .cell_population import CellPopulation
from .red_queen_history import OpponentHistory,digest

class ActionOpponentHistory(OpponentHistory):
    def admission(self,parent,candidate):
        reasons=[]
        for label in ("fresh","archive","anchor"):
            a,b=parent[label],candidate[label]
            if any(s["partition"]!="development" for s in (a,b)):
                raise ValueError("reporting cannot drive action niche selection")
            if a["source"]!=b["source"] or a["manifest"]!=b["manifest"] or a["outcomes"].keys()!=b["outcomes"].keys():
                raise ValueError("matched declared fixture evidence required")
            if label=="anchor" and a["source"]!="isolated-action-fixture-safety":
                raise ValueError("explicit action safety fixture anchor required; not ARC")
            for s in (a,b):
                if not s["outcomes"] or any(type(v) is not bool for v in s["outcomes"].values()):
                    raise ValueError("executed boolean outcomes required")
                if s["false_positive_given_consistency"] is None:
                    reasons.append({"gate":label+"_undefined_false_action_rate"})
            if any(v and not b["outcomes"][k] for k,v in a["outcomes"].items()):
                reasons.append({"gate":label+"_regression"})
            if (a["false_positive_given_consistency"] is not None and b["false_positive_given_consistency"] is not None and
                    b["false_positive_given_consistency"]>a["false_positive_given_consistency"]):
                reasons.append({"gate":label+"_false_action"})
        if sum(candidate["fresh"]["outcomes"].values())<=sum(parent["fresh"]["outcomes"].values()):
            reasons.append({"gate":"strict_fresh_gain"})
        if candidate.get("caps_passed") is not True or candidate.get("fixtures_passed") is not True:
            reasons.append({"gate":"caps_or_fixtures"})
        for record in self.records:
            old=record["scores"]["archive"];new=candidate["archive"]
            if old["manifest"]!=new["manifest"]:raise ValueError("history manifest mismatch")
            if any(v and not new["outcomes"][k] for k,v in old["outcomes"].items()):
                reasons.append({"gate":"historical_opponent_regression","opponent":record["graph_hash"]})
        return {"accepted":not reasons,"reasons":reasons,"scope":"controlled lexical action niche only; no real ARC admission",
                "opponent_count":len(self.records),"history_hash":digest(self.records)}

class ActionCellPopulation(CellPopulation):
    def seed(self,*args,**kwargs):
        cid=super().seed(*args,**kwargs)
        family=self.cells[cid]["structure"]
        self.histories[family]=ActionOpponentHistory(self.histories[family].records)
        return cid
