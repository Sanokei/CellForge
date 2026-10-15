"""New spatial exploration template; generic mechanism, assistant-authored.
Never changes old templates. No game ID, target labels, or fixed colour constants.
"""
def act(params, mem, frame, avail, info):
    if "prev" not in mem:
        mem["stats"] = np.zeros((max(avail) + 1, 4))
        mem["visits"] = np.zeros((64, 64))
        mem["seen"] = {}
        mem["last"] = 0
        mem["resets"] = info["resets"]
    if info["resets"] != mem["resets"]:
        mem["stats"] = np.zeros((max(avail) + 1, 4))
        mem["visits"] = np.zeros((64, 64))
        mem["seen"] = {}
        mem["last"] = 0
        mem["resets"] = info["resets"]
        mem.pop("prev", None)
    if max(avail) >= len(mem["stats"]):
        grown = np.zeros((max(avail) + 1, 4))
        grown[:len(mem["stats"])] = mem["stats"]
        mem["stats"] = grown
    stats = mem["stats"]
    if "prev" in mem:
        change = float(np.mean(frame != mem["prev"]))
        rate = 1.0 / (1.0 + np.exp(-np.clip(params["rates"], -20, 20)))
        last = mem["last"]
        stats[last, 0] += rate[0] * (change - stats[last, 0])
        stats[last, 1] += rate[1] * (float(change == 0) - stats[last, 1])
        stats[last, 2] += 1
        stats[last, 3] = float(change == 0) * (stats[last, 3] + 1)
    key = info["key"]
    repeats = mem["seen"].get(key, 0)
    if len(mem["seen"]) >= 4096:
        mem["seen"].clear()
    mem["seen"][key] = repeats + 1
    scores = []
    for a in avail:
        row = stats[a]
        features = np.array([1.0, row[0], -row[1],
                             1.0 / np.sqrt(1 + row[2]),
                             -float(a == mem["last"]),
                             -row[3] / (1 + row[3]),
                             -float(a == mem["last"]) * repeats / (1 + repeats),
                             float(a == 6)])
        scores.append(float(np.dot(params["action"], features)))
    a = avail[int(np.argmax(np.asarray(scores)))]
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
    score = (p[0] * rarity + p[1] * edges / 4.0
             + p[2] / np.sqrt(1 + visits) - p[3] * visits / (1 + visits)
             + p[4] * xs / 63.0 + p[5] * ys / 63.0)
    index = int(np.argmax(score))
    y, x = index // 64, index % 64
    if a == 6:
        visits[y, x] += 1
    mem["prev"] = frame.copy()
    mem["last"] = a
    return a, x, y

