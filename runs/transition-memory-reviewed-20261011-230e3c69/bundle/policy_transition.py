"""Observation-conditioned transition-memory policy. New template, not deployed."""
import inspect
import numpy as np

SHAPES = {"effect": 8, "action": 8, "rates": 2, "mix": 2,
          "lookahead": 4, "pixel": 6, "memory_gate": 1, "context_gate": 1}

def observation_features(frame):
    colours, counts = np.unique(frame, return_counts=True)
    density = np.zeros((64, 64))
    for colour, count in zip(colours, counts):
        density += (frame == colour) / float(count)
    ys, xs = np.indices((64, 64))
    mass = float(np.sum(density))
    x = float(np.sum(xs * density) / mass) / 63.0
    y = float(np.sum(ys * density) / mass) / 63.0
    sx = float(np.sqrt(np.sum(((xs / 63.0 - x) ** 2) * density) / mass))
    sy = float(np.sqrt(np.sum(((ys / 63.0 - y) ** 2) * density) / mass))
    p = counts / float(np.sum(counts))
    entropy = float(-np.sum(p * np.log(p)) / np.log(16.0))
    edges = float(np.mean(frame[1:, :] != frame[:-1, :])
                  + np.mean(frame[:, 1:] != frame[:, :-1])) / 2.0
    return np.array([x, y, sx, sy, len(colours) / 16.0, entropy, edges,
                     float(np.min(p))])

def update_row(row, delta, noop, progress, alpha):
    row[:8] += alpha * (delta - row[:8])
    row[8] += alpha * (noop - row[8])
    row[9] += alpha * (progress - row[9])
    row[10] += 1.0
    row[11] = noop * (row[11] + 1.0)

def action_value(params, row, repeated, successor_visits):
    effect = float(np.dot(params["effect"], row[:8]))
    features = np.array([1.0, effect, row[9], -row[8],
                         -row[11] / (1.0 + row[11]),
                         1.0 / np.sqrt(1.0 + row[10]),
                         -float(repeated),
                         -successor_visits / (1.0 + successor_visits)])
    return float(np.dot(params["action"], features))

def act(params, mem, frame, avail, info):
    if "resets" not in mem or mem["resets"] != info["resets"]:
        mem.clear()
        mem["resets"] = info["resets"]
        mem["global"] = np.zeros((max(avail) + 1, 12))
        mem["nodes"] = {}
        mem["pixel_visits"] = np.zeros((64, 64))
        mem["last"] = 0
        mem["eligibility"] = []
    if max(avail) >= len(mem["global"]):
        grown = np.zeros((max(avail) + 1, 12))
        grown[:len(mem["global"])] = mem["global"]
        mem["global"] = grown
    features = observation_features(frame)
    # Quantized observation features; no IDs, labels or external solution lookup.
    key = tuple(np.floor(features[:7] * 16.0).astype(int).tolist())
    nodes = mem["nodes"]
    if key not in nodes:
        if len(nodes) >= 256:
            old = next(iter(nodes))
            nodes.pop(old)
        nodes[key] = {"rows": np.zeros((max(avail) + 1, 12)),
                      "next": {}, "legal": list(avail), "visits": 0}
    node = nodes[key]
    if max(avail) >= len(node["rows"]):
        grown = np.zeros((max(avail) + 1, 12))
        grown[:len(node["rows"])] = node["rows"]
        node["rows"] = grown
    node["legal"] = list(avail)
    node["visits"] += 1
    alpha = 1.0 / (1.0 + np.exp(-np.clip(params["rates"], -20, 20)))
    mix = 1.0 / (1.0 + np.exp(-np.clip(params["mix"], -20, 20)))
    if "prev_features" in mem:
        delta = np.abs(features - mem["prev_features"])
        delta[7] = float(np.mean(frame != mem["prev_frame"]))
        noop = float(delta[7] == 0.0)
        progress = float(max(0, info["levels"] - mem["prev_levels"]))
        last = mem["last"]
        # Credit observed progress to up to seven earlier decisions, without replay/query.
        history = mem["eligibility"]
        if progress > 0:
            for lag in range(1, len(history)):
                old_key, old_action = history[-1 - lag]
                target_progress = progress * mix[1] ** lag
                old_global = mem["global"][old_action]
                old_global[9] += alpha[0] * (target_progress - old_global[9])
                old_node = nodes.get(old_key)
                if old_node is not None:
                    old_row = old_node["rows"][old_action]
                    old_row[9] += alpha[1] * (target_progress - old_row[9])
        update_row(mem["global"][last], delta, noop, progress, alpha[0])
        previous = nodes.get(mem["prev_key"])
        if previous is not None:
            update_row(previous["rows"][last], delta, noop, progress, alpha[1])
            previous["next"][last] = key
    enabled = float(params["memory_gate"][0])
    scores = []
    for action in avail:
        count = node["rows"][action, 10]
        context = float(params["context_gate"][0])
        blend = context * mix[0] * count / (1.0 + count)
        row = enabled * (blend * node["rows"][action] + (1.0 - blend) * mem["global"][action])
        successor = node["next"].get(action) if enabled and context else None
        target = nodes.get(successor) if successor is not None else None
        visits = target["visits"] if target is not None else 0
        score = action_value(params, row, enabled and action == mem["last"], visits)
        continuation = 0.0
        uncertainty = 1.0
        if target is not None:
            values = [action_value(params, target["rows"][a], False, 0)
                      for a in target["legal"]]
            continuation = max(values)
            uncertainty = 1.0 / np.sqrt(1.0 + target["visits"])
        look = np.array([mix[1] * continuation,
                         -visits / (1.0 + visits), uncertainty,
                         -float(successor == key)])
        score += enabled * context * float(np.dot(params["lookahead"], look))
        scores.append(score)
    action = avail[int(np.argmax(np.asarray(scores)))]
    # Same spatial component for all founders; separated from action-memory credit.
    rarity = np.zeros((64, 64))
    for colour in np.unique(frame):
        mask = frame == colour
        rarity += mask / np.sqrt(1.0 + np.sum(mask))
    edges = np.zeros((64, 64))
    edges[1:, :] += frame[1:, :] != frame[:-1, :]
    edges[:-1, :] += frame[:-1, :] != frame[1:, :]
    edges[:, 1:] += frame[:, 1:] != frame[:, :-1]
    edges[:, :-1] += frame[:, :-1] != frame[:, 1:]
    visits = mem["pixel_visits"]
    ys, xs = np.indices((64, 64))
    p = params["pixel"]
    pixel_score = (p[0] * rarity + p[1] * edges / 4.0
                   + p[2] / np.sqrt(1.0 + visits)
                   - p[3] * visits / (1.0 + visits)
                   + p[4] * xs / 63.0 + p[5] * ys / 63.0)
    index = int(np.argmax(pixel_score))
    y, x = index // 64, index % 64
    if action == 6:
        visits[y, x] += 1
    mem["prev_features"] = features.copy()
    mem["prev_frame"] = frame.copy()
    mem["prev_levels"] = info["levels"]
    mem["prev_key"] = key
    mem["last"] = action
    mem["eligibility"].append((key, action))
    if len(mem["eligibility"]) > 8:
        mem["eligibility"].pop(0)
    return action, x, y

SOURCE = "\n\n".join(inspect.getsource(fn) for fn in
                        (observation_features, update_row, action_value, act))

