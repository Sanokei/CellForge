"""True RHAE scorer — exact spec from ARC-AGI-3 tech report (arXiv 2603.24621, sec 4.1).

For completed level l of environment e:
    S_l = min(1.0, h_l / a_l)^2        (h_l = second-best human actions, a_l = agent actions)
Incomplete levels score 0.
Environment: E = sum(l * S_l for l in 1..n) / (n(n+1)/2)   (later levels weighted more)
Total: mean of E across environments. 0..100 (reported as %).
"""
import json


def level_score(human_actions, agent_actions):
    """RHAE for one completed level. Returns 0.0 if level not completed (agent_actions=None)."""
    if agent_actions is None or agent_actions <= 0:
        return 0.0
    return min(1.0, human_actions / agent_actions) ** 2


def environment_score(baselines, per_level_agent_actions):
    """baselines: list of human action counts per level (0-indexed).
    per_level_agent_actions: dict {level_1indexed: actions} for COMPLETED levels.
    Returns (env_score_0_1, {level: score})."""
    n = len(baselines)
    denom = n * (n + 1) / 2
    scores = {}
    total = 0.0
    for l in range(1, n + 1):
        s = level_score(baselines[l - 1], per_level_agent_actions.get(l))
        scores[l] = s
        total += l * s
    return total / denom, scores


def verify():
    # Hand-checkable example: 5 levels, human [10,20,30,40,50].
    # Agent: L1 in 10 (->1.0), L2 in 40 (->(20/40)^2=0.25), L3 in 30 (->1.0),
    #        L4 not completed (->0), L5 in 100 (->(50/100)^2=0.25).
    # E = (1*1.0 + 2*0.25 + 3*1.0 + 4*0 + 5*0.25)/15 = (1+0.5+3+0+1.25)/15 = 5.75/15
    e, s = environment_score([10, 20, 30, 40, 50], {1: 10, 2: 40, 3: 30, 5: 100})
    assert abs(e - 5.75 / 15) < 1e-9, e
    assert s[4] == 0.0 and abs(s[1] - 1.0) < 1e-9 and abs(s[2] - 0.25) < 1e-9
    # cap check: agent beats human 2x -> capped at 1.0, not 4.0
    assert abs(level_score(20, 10) - 1.0) < 1e-9
    print("RHAE scorer verified against hand computation.")


if __name__ == "__main__":
    verify()
