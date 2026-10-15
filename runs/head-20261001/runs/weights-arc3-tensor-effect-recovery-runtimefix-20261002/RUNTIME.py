def features(stats, action, last, global_change):
    row = stats[action]
    return np.array([1.0, row[0], row[1], row[2] / (1.0 + row[2]), float(action == last), global_change, 1.0 / np.sqrt(1.0 + row[2]), row[3]])


def predict(params, feat):
    effect = 1.0 / (1.0 + np.exp(-np.clip(np.dot(params['effect'], feat), -30, 30)))
    noop = 1.0 / (1.0 + np.exp(-np.clip(np.dot(params['recovery'], feat), -30, 30)))
    return effect, noop


def act(params, mem, frame, avail, info):
    if 'stats' not in mem:
        mem['stats'] = np.zeros((max(avail) + 1, 4))
        mem['last'] = 0
        mem['global'] = 0.0
    stats = mem['stats']
    if max(avail) >= len(stats):
        grown = np.zeros((max(avail) + 1, 4))
        grown[:len(stats)] = stats
        stats = grown
        mem['stats'] = stats
    if 'prev' in mem:
        last = mem['last']
        changed = float(np.mean(frame != mem['prev']))
        noop = float(changed == 0.0)
        rates = 1.0 / (1.0 + np.exp(-np.clip(params['rates'], -20, 20)))
        stats[last, 0] = (1.0 - rates[0]) * stats[last, 0] + rates[0] * changed
        stats[last, 1] = (1.0 - rates[1]) * stats[last, 1] + rates[1] * noop
        stats[last, 2] += 1.0
        stats[last, 3] = noop * (stats[last, 3] + 1.0) / (2.0 + stats[last, 3])
        mem['global'] = (mem['global'] + changed) / 2.0
    scores = []
    for action in avail:
        feat = features(stats, action, mem['last'], mem['global'])
        effect, noop = predict(params, feat)
        route = np.array([effect, -noop, feat[6], -feat[2]])
        scores.append(float(np.dot(params['router'], route)))
    best = int(np.argmax(np.asarray(scores)))
    action = avail[best]
    step = info['step']
    click = params['click']
    x = int(np.abs(click[0] * step + click[1]) % 64)
    y = int(np.abs(click[2] * step + click[3]) % 64)
    mem['prev'] = frame.copy()
    mem['last'] = action
    return action, x, y
