"""Snake game with fly-style senses: what's to my left/right/ahead, relative to my heading."""

import numpy as np

ACTIONS = ['left', 'straight', 'right']
SENSES = ['food_L', 'food_R', 'danger_L', 'danger_R', 'danger_ahead']


class Snake:
    def __init__(self, size=14, seed=0, max_idle=300):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.max_idle = max_idle
        c = size // 2
        self.body = [(c, c), (c - 1, c), (c - 2, c)]  # head first
        self.dir = (1, 0)
        self.score = 0
        self.steps = 0
        self.idle = 0
        self.alive = True
        self.death = None
        self._place_food()

    def _place_food(self):
        free = [(x, y) for x in range(self.size) for y in range(self.size) if (x, y) not in self.body]
        self.food = free[self.rng.integers(len(free))]

    @staticmethod
    def _left(d):
        return (-d[1], d[0])

    @staticmethod
    def _right(d):
        return (d[1], -d[0])

    def _blocked(self, p):
        return not (0 <= p[0] < self.size and 0 <= p[1] < self.size) or p in self.body[:-1]

    def senses(self):
        hx, hy = self.body[0]
        d, l, r = self.dir, self._left(self.dir), self._right(self.dir)
        rel = (self.food[0] - hx, self.food[1] - hy)
        lat = rel[0] * l[0] + rel[1] * l[1]
        fwd = rel[0] * d[0] + rel[1] * d[1]
        return {
            'food_L': float(lat > 0 or (lat == 0 and fwd > 0)),
            'food_R': float(lat < 0 or (lat == 0 and fwd > 0)),
            'danger_L': float(self._blocked((hx + l[0], hy + l[1]))),
            'danger_R': float(self._blocked((hx + r[0], hy + r[1]))),
            'danger_ahead': float(self._blocked((hx + d[0], hy + d[1]))),
        }

    def teacher(self):
        """Greedy reference player: never step into danger, head toward food."""
        s = self.senses()
        safe = [a for a, k in zip(range(3), ['danger_L', 'danger_ahead', 'danger_R']) if not s[k]]
        if not safe:
            return 1
        want = 0 if s['food_L'] and not s['food_R'] else 2 if s['food_R'] and not s['food_L'] else 1
        if s['food_L'] == 0 and s['food_R'] == 0:
            want = 0  # food behind: start turning
        return want if want in safe else (1 if 1 in safe else safe[0])

    def step(self, action):
        if action == 0:
            self.dir = self._left(self.dir)
        elif action == 2:
            self.dir = self._right(self.dir)
        head = (self.body[0][0] + self.dir[0], self.body[0][1] + self.dir[1])
        self.steps += 1
        self.idle += 1
        if self._blocked(head):
            self.alive = False
            self.death = 'wall' if not (0 <= head[0] < self.size and 0 <= head[1] < self.size) else 'itself'
            return
        self.body.insert(0, head)
        if head == self.food:
            self.score += 1
            self.idle = 0
            if len(self.body) == self.size ** 2:
                self.alive = False
                self.death = 'won'
                return
            self._place_food()
        else:
            self.body.pop()
        if self.idle > self.max_idle:
            self.alive = False
            self.death = 'looping'
