"""Can the fly connectome play Snake? Real wiring vs scrambled wiring vs baselines.

1. For every combination of the 5 senses, stimulate the brain for STIM_MS and record
   descending neuron (brain -> body) spike counts, REPEATS times (Poisson input = noisy).
2. Fit a linear decoder from descending-neuron counts to the reference move, on half the
   repeats. Games use the other half, so the decoder never saw those exact brain responses.
3. Play GAMES games per player and save the scoreboard to results/.
"""

import itertools
import json
import os
import time

import numpy as np

from brain import Brain, load, scramble, stabilise
from snake import SENSES, Snake

STIM_MS = 80
REPEATS = 12
GAMES = 100


def sense_key(s):
    return tuple(int(s[k]) for k in SENSES)


def record_bank(brain):
    bank = {}
    for key in itertools.product([0, 1], repeat=len(SENSES)):
        drive = dict(zip(SENSES, key))
        rows = []
        for _ in range(REPEATS):
            brain.reset()
            c = brain.run(drive, STIM_MS)
            rows.append(c[brain.descending])
        bank[key] = np.array(rows)
    return bank


def teacher_label(key):
    """Reference move for a sense pattern (the same rule Snake.teacher uses)."""
    g = Snake()
    s = dict(zip(SENSES, key))
    g.senses = lambda: s
    return g.teacher()


class Decoder:
    """Multinomial logistic regression, plain numpy."""

    def fit(self, X, y, l2=1e-2, iters=3000, lr=0.5):
        X = np.log1p(X)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        X = (X - self.mu) / self.sd
        n, d = X.shape
        self.Wt, self.b = np.zeros((d, 3)), np.zeros(3)
        Y = np.eye(3)[y]
        for _ in range(iters):
            p = self._softmax(X @ self.Wt + self.b)
            self.Wt -= lr * (X.T @ (p - Y) / n + l2 * self.Wt)
            self.b -= lr * (p - Y).mean(0)
        return self

    @staticmethod
    def _softmax(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def predict(self, X):
        X = (np.log1p(np.atleast_2d(X)) - self.mu) / self.sd
        return (X @ self.Wt + self.b).argmax(1)


def split(bank, cols=None):
    tr, te = {}, {}
    for k, v in bank.items():
        v = v if cols is None else v[:, cols]
        tr[k], te[k] = v[:REPEATS // 2], v[REPEATS // 2:]
    return tr, te


def train(bank_train):
    X = np.concatenate(list(bank_train.values()))
    y = np.concatenate([[teacher_label(k)] * len(v) for k, v in bank_train.items()])
    return Decoder().fit(X, y)


def accuracy(dec, bank_test):
    X = np.concatenate(list(bank_test.values()))
    y = np.concatenate([[teacher_label(k)] * len(v) for k, v in bank_test.items()])
    return float((dec.predict(X) == y).mean())


def play(policy, seed):
    g = Snake(seed=seed)
    while g.alive:
        g.step(policy(g))
    return g.score, g.steps, g.death


def main():
    os.makedirs('results', exist_ok=True)
    W, neurons = load()
    W_fly = stabilise(W, neurons)
    brains = {'fly': Brain(W_fly, neurons, seed=1),
              'scrambled': Brain(scramble(W_fly, seed=0), neurons, seed=1)}
    rng = np.random.default_rng(0)

    banks, steer_cols = {}, {}
    for name, b in brains.items():
        t = time.time()
        banks[name] = record_bank(b)
        pos = {n: i for i, n in enumerate(b.descending)}
        steer_cols = {s: [pos[i] for i in b.steer[s]] for s in 'LR'}
        print(f'{name}: recorded {len(banks[name]) * REPEATS} brain runs in {time.time() - t:.0f}s', flush=True)

    players, info = {}, {}

    def bank_player(bank_test, dec):
        return lambda g: int(dec.predict(bank_test[sense_key(g.senses())][rng.integers(REPEATS // 2)])[0])

    for name in brains:
        tr, te = split(banks[name])
        dec = train(tr)
        acc = accuracy(dec, te)
        players[f'{name} brain, all 1314 descending neurons + decoder'] = bank_player(te, dec)
        info[f'{name} brain, all 1314 descending neurons + decoder'] = {'decoder_accuracy': acc}
        cols = steer_cols['L'] + steer_cols['R']
        tr4, te4 = split(banks[name], cols)
        dec4 = train(tr4)
        players[f'{name} brain, 4 steering neurons + decoder'] = bank_player(te4, dec4)
        info[f'{name} brain, 4 steering neurons + decoder'] = {'decoder_accuracy': accuracy(dec4, te4)}
        np.savez(f'results/decoder_{name}.npz', W=dec.Wt, b=dec.b, mu=dec.mu, sd=dec.sd)

        def wired(g, te=te):
            r = te[sense_key(g.senses())][rng.integers(REPEATS // 2)]
            diff = r[steer_cols['L']].sum() - r[steer_cols['R']].sum()
            return 0 if diff > 0 else 2 if diff < 0 else 1
        players[f'{name} brain, steering neurons only (no training)'] = wired

    players['random moves'] = lambda g: int(rng.integers(3))
    players['reference rule (no brain)'] = lambda g: g.teacher()

    results = {}
    for name, p in players.items():
        games = [play(p, seed) for seed in range(GAMES)]
        scores = np.array([s for s, _, _ in games])
        deaths = {}
        for _, _, d in games:
            deaths[d] = deaths.get(d, 0) + 1
        results[name] = {'mean_score': float(scores.mean()), 'best': int(scores.max()),
                         'mean_steps': float(np.mean([s for _, s, _ in games])), 'deaths': deaths,
                         **info.get(name, {})}
        print(f"{name:55s} mean {scores.mean():5.2f}  best {scores.max():3d}  "
              f"{info.get(name, {})}  deaths {deaths}", flush=True)

    # How differently the brain responds to each sense (does it keep left vs right apart?).
    for name in brains:
        means = {k: v.mean(0) for k, v in banks[name].items()}
        base = dict(zip(SENSES, [0] * 5))
        sep = {}
        for s in SENSES:
            k1 = tuple(int(x) for x in {**base, s: 1}.values())
            sep[s] = int(np.count_nonzero(means[k1]))
        results[f'_{name}_descending_neurons_active_per_sense'] = sep
    json.dump(results, open('results/scoreboard.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
