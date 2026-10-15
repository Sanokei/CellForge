"""Generic scripted frontier explorer comparator; no game-specific rules."""
import inspect
import numpy as np

def act(params, mem, frame, avail, info):
    if "resets" not in mem or mem["resets"] != info["resets"]:
        mem.clear()
        mem["resets"] = info["resets"]
        mem["nodes"] = {}
        mem["visits"] = np.zeros((64, 64))
    nodes = mem["nodes"]
    key = info["key"]
    if key not in nodes:
        if len(nodes) >= 256:
            nodes.pop(next(iter(nodes)))
        nodes[key] = {"tries": {}, "next": {}}
    node = nodes[key]
    if "last_key" in mem:
        previous = nodes.get(mem["last_key"])
        if previous is not None:
            previous["next"][mem["last_action"]] = key
    # Try each legal class at the current observed state; otherwise route to an observed frontier.
    scores = [node["tries"].get(a, 0) for a in avail]
    action = avail[int(np.argmin(np.asarray(scores)))]
    if min(scores) > 0:
        queue = [(key, [])]
        seen = {key}
        route = []
        while queue and not route:
            current, path = queue.pop(0)
            entry = nodes.get(current)
            if entry is not None:
                if path and any(entry["tries"].get(a, 0) == 0 for a in avail):
                    route = path
                for a, target in entry["next"].items():
                    if target not in seen and target in nodes:
                        seen.add(target)
                        queue.append((target, path + [a]))
        if route and route[0] in avail:
            action = route[0]
    node["tries"][action] = node["tries"].get(action, 0) + 1
    rarity = np.zeros((64, 64))
    for colour in np.unique(frame):
        mask = frame == colour
        rarity += mask / np.sqrt(1.0 + np.sum(mask))
    edges = np.zeros((64, 64))
    edges[1:, :] += frame[1:, :] != frame[:-1, :]
    edges[:-1, :] += frame[:-1, :] != frame[1:, :]
    edges[:, 1:] += frame[:, 1:] != frame[:, :-1]
    edges[:, :-1] += frame[:, :-1] != frame[:, 1:]
    visits = mem["visits"]
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
    mem["last_key"] = key
    mem["last_action"] = action
    return action, x, y

SOURCE = inspect.getsource(act)

