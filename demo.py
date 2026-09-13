"""Render the demo video: the simulated fly connectome playing Snake, live.

Every move: the snake's senses stimulate real visual neurons, the whole 165k-neuron
connectome is simulated for 80 ms, and the move is read out from descending neurons.
Spikes are drawn at each neuron's real soma position (brain on top, nerve cord below).

  .venv/bin/python demo.py            -> videos/fly_plays_snake.mp4 (run from repo root)
"""

import json
import os
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

from brain import Brain, load, stabilise
from experiment import STIM_MS
from snake import ACTIONS, SENSES, Snake

W_VID, H_VID, FPS = 1920, 1080, 30
FRAMES_PER_MOVE = 3
BG = (9, 12, 18)
PANEL = (17, 22, 32)
GRID = (26, 33, 46)
TEXT = (228, 232, 240)
MUTED = (132, 142, 160)
AMBER = (255, 176, 64)
GREEN = (92, 220, 120)
RED = (255, 84, 84)
CYAN = (70, 210, 255)

FONT = '/System/Library/Fonts/HelveticaNeue.ttc'


def font(size, bold=False):
    return ImageFont.truetype(FONT, size, index=1 if bold else 0)


class BrainMap:
    """Dorsal view of the CNS from soma coordinates, with decaying spike glow."""

    def __init__(self, neurons, brain, box):
        self.x0, self.y0, self.w, self.h = box
        ok = neurons[['x', 'z']].notna().all(axis=1).values
        x, z = neurons.x.values, neurons.z.values
        # Fly's left on screen left (left-side somas have larger x).
        sx = (np.nanmax(x) - x) / (np.nanmax(x) - np.nanmin(x))
        sz = (z - np.nanmin(z)) / (np.nanmax(z) - np.nanmin(z))
        scale = min(self.w / 1.0, self.h / ((np.nanmax(z) - np.nanmin(z)) / (np.nanmax(x) - np.nanmin(x))))
        wpx = scale
        hpx = scale * (np.nanmax(z) - np.nanmin(z)) / (np.nanmax(x) - np.nanmin(x))
        ox, oy = (self.w - wpx) / 2, (self.h - hpx) / 2
        self.px = np.where(ok, ox + sx * (wpx - 1), -1).astype(int)
        self.py = np.where(ok, oy + sz * (hpx - 1), -1).astype(int)
        self.ok = ok

        # Colour per neuron: inputs green/red, descending cyan, everything else amber.
        col = np.tile(np.array(AMBER, np.float32) / 255, (len(neurons), 1))
        for k, idx in brain.inputs.items():
            col[idx] = np.array(GREEN if k.startswith('food') else RED) / 255
        col[brain.descending] = np.array(CYAN) / 255
        self.col = col
        self.weight = np.ones(len(neurons), np.float32)
        self.weight[brain.descending] = 4.0

        base = np.zeros((self.h, self.w), np.float32)
        np.add.at(base, (self.py[ok], self.px[ok]), 1)
        base = gaussian_filter(base, 0.7)
        base = np.clip(base / np.percentile(base[base > 0], 99), 0, 1) ** 0.6
        self.base = (base[..., None] * np.array([40, 52, 72], np.float32)[None, None]).astype(np.float32)
        self.glow = np.zeros((self.h, self.w, 3), np.float32)

    def add_spikes(self, idx):
        idx = idx[self.ok[idx]]
        self.glow *= 0.55
        if idx.size:
            for c in range(3):
                np.add.at(self.glow[..., c], (self.py[idx], self.px[idx]), self.col[idx, c] * self.weight[idx])

    def image(self):
        g = gaussian_filter(self.glow, (1.3, 1.3, 0))
        img = self.base + 255 * (1 - np.exp(-g * 1.6))
        return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


class Renderer:
    def __init__(self, neurons, brain):
        self.brain = brain
        self.map = BrainMap(neurons, brain, (1110, 150, 520, 800))
        sprite = Image.open('assets/fly_top.png')  # facing +x
        self.board = (60, 70, 940)
        self.cell = self.board[2] // 14
        s = int(self.cell * 1.9)
        sprite = sprite.resize((int(sprite.width * s / sprite.height), s), Image.LANCZOS)
        self.sprites = {(1, 0): sprite, (0, 1): sprite.rotate(90, expand=True),
                        (-1, 0): sprite.rotate(180, expand=True), (0, -1): sprite.rotate(-90, expand=True)}
        self.f_big, self.f_mid, self.f_small = font(40, True), font(26, True), font(21)
        self.f_tiny = font(17)

    def cell_xy(self, p):
        bx, by, _ = self.board
        # Game y grows upward; screen y grows downward.
        return bx + p[0] * self.cell, by + (13 - p[1]) * self.cell

    def draw_board(self, d, frame, g, senses):
        bx, by, bs = self.board
        d.rounded_rectangle([bx - 12, by - 12, bx + bs + 12, by + bs + 12], 18, fill=PANEL)
        for i in range(15):
            d.line([bx + i * self.cell, by, bx + i * self.cell, by + 14 * self.cell], fill=GRID)
            d.line([bx, by + i * self.cell, bx + 14 * self.cell, by + i * self.cell], fill=GRID)
        # Danger cells the fly currently sees looming.
        hx, hy = g.body[0]
        dd = g.dir
        dirs = {'danger_L': (-dd[1], dd[0]), 'danger_R': (dd[1], -dd[0]), 'danger_ahead': dd}
        for k, v in dirs.items():
            if senses[k]:
                x, y = self.cell_xy((hx + v[0], hy + v[1]))
                d.rectangle([x + 2, y + 2, x + self.cell - 2, y + self.cell - 2], outline=RED, width=4)
        # Fruit.
        fx, fy = self.cell_xy(g.food)
        c = self.cell
        d.ellipse([fx + c * .18, fy + c * .22, fx + c * .82, fy + c * .86], fill=(214, 40, 60))
        d.ellipse([fx + c * .32, fy + c * .34, fx + c * .44, fy + c * .46], fill=(255, 140, 150))
        d.polygon([(fx + c * .5, fy + c * .24), (fx + c * .72, fy + c * .06), (fx + c * .62, fy + c * .28)],
                  fill=(90, 190, 90))
        # Body: amber segments fading toward the tail.
        n = len(g.body)
        for i, p in enumerate(reversed(g.body[1:])):
            x, y = self.cell_xy(p)
            t = (i + 1) / n
            col = tuple(int(a * (0.35 + 0.65 * t)) for a in AMBER)
            d.rounded_rectangle([x + 7, y + 7, x + c - 7, y + c - 7], 12, fill=col)
        spr = self.sprites[g.dir]
        x, y = self.cell_xy(g.body[0])
        frame.alpha_composite(spr, (int(x + c / 2 - spr.width / 2), int(y + c / 2 - spr.height / 2)))

    def bar(self, d, x, y, w, label, value, vmax, color, h=26):
        d.text((x, y), label, font=self.f_small, fill=TEXT)
        yb = y + 30
        d.rounded_rectangle([x, yb, x + w, yb + h], 6, fill=GRID)
        fill = int(w * min(1, value / vmax)) if vmax else 0
        if fill > 4:
            d.rounded_rectangle([x, yb, x + fill, yb + h], 6, fill=color)

    def draw_panel(self, d, st):
        x = 1680
        d.text((x, 150), 'EYES  (input)', font=self.f_mid, fill=MUTED)
        rows = [('fruit, left eye', 'food_L', GREEN), ('fruit, right eye', 'food_R', GREEN),
                ('looming, left', 'danger_L', RED), ('looming, ahead', 'danger_ahead', RED),
                ('looming, right', 'danger_R', RED)]
        for i, (lab, k, col) in enumerate(rows):
            y = 195 + i * 40
            on = st['senses'][k] > 0
            d.ellipse([x, y + 4, x + 20, y + 24], fill=col if on else GRID)
            d.text((x + 32, y), lab, font=self.f_small, fill=TEXT if on else MUTED)

        d.text((x, 420), 'TURN NEURONS', font=self.f_mid, fill=MUTED)
        d.text((x, 452), 'DNa01 + DNa02 spikes', font=self.f_tiny, fill=MUTED)
        self.bar(d, x, 478, 200, f"left   {st['steer_L']}", st['steer_L'], 25, CYAN)
        self.bar(d, x, 540, 200, f"right  {st['steer_R']}", st['steer_R'], 25, CYAN)

        d.text((x, 625), 'DECISION', font=self.f_mid, fill=MUTED)
        for i, a in enumerate(ACTIONS):
            y = 665 + i * 44
            chosen = st['action'] == i
            d.rounded_rectangle([x, y, x + 200, y + 36], 8, fill=AMBER if chosen else GRID)
            d.text((x + 14, y + 6), a, font=self.f_small, fill=BG if chosen else MUTED)

        d.text((x, 820), 'NEURONS FIRING', font=self.f_mid, fill=MUTED)
        d.text((x, 855), f"{st['active']:,}", font=self.f_big, fill=AMBER)
        d.text((x, 900), f"in {STIM_MS} ms of brain time", font=self.f_tiny, fill=MUTED)

    def frame(self, g, st, caption, sub):
        img = Image.new('RGBA', (W_VID, H_VID), BG + (255,))
        d = ImageDraw.Draw(img)
        self.draw_board(d, img, g, st['senses'])
        d.text((1110, 40), caption, font=self.f_big, fill=TEXT)
        d.text((1110, 92), sub, font=self.f_small, fill=MUTED)
        mx, my = self.map.x0, self.map.y0
        img.paste(self.map.image(), (mx, my))
        d.text((mx, my + self.map.h + 8), 'brain', font=self.f_tiny, fill=MUTED)
        lx = mx + 80
        for lab, col in [('eye neurons (in)', GREEN), ('descending neurons (out)', CYAN), ('everything else', AMBER)]:
            d.ellipse([lx, my + self.map.h + 13, lx + 12, my + self.map.h + 25], fill=col)
            d.text((lx + 18, my + self.map.h + 8), lab, font=self.f_tiny, fill=MUTED)
            lx += 30 + d.textlength(lab, font=self.f_tiny)
        d.text((mx, my + self.map.h + 34), 'nerve cord below', font=self.f_tiny, fill=MUTED)
        d.text((60, 1030), f"SCORE  {g.score}", font=self.f_mid, fill=AMBER)
        d.text((280, 1034), f"move {g.steps}", font=self.f_small, fill=MUTED)
        self.draw_panel(d, st)
        return np.array(img.convert('RGB'))


def card(lines, sizes, colors, extra=None):
    img = Image.new('RGB', (W_VID, H_VID), BG)
    d = ImageDraw.Draw(img)
    y = H_VID // 2 - sum(s + 22 for s in sizes) // 2
    for text, s, col in zip(lines, sizes, colors):
        f = font(s, s >= 40)
        w = d.textlength(text, font=f)
        d.text(((W_VID - w) / 2, y), text, font=f, fill=col)
        y += s + 22
    if extra:
        extra(d)
    return np.array(img)


def decode_policy(dec, brain):
    def act(counts):
        X = (np.log1p(counts[brain.descending])[None] - dec['mu']) / dec['sd']
        return int((X @ dec['W'] + dec['b']).argmax())
    return act


def wired_policy(brain):
    def act(counts):
        diff = counts[brain.steer['L']].sum() - counts[brain.steer['R']].sum()
        return 0 if diff > 0 else 2 if diff < 0 else 1
    return act


def play_live(brain, policy, seed, max_moves):
    """Play a game with the live brain, recording per-move spikes for rendering."""
    g = Snake(seed=seed)
    moves = []
    while g.alive and g.steps < max_moves:
        s = g.senses()
        brain.reset()
        rec = []
        counts = brain.run(s, STIM_MS, record=rec)
        a = policy(counts)
        moves.append({'senses': s, 'spikes': rec, 'action': a,
                      'steer_L': int(counts[brain.steer['L']].sum()),
                      'steer_R': int(counts[brain.steer['R']].sum()),
                      'active': int(np.count_nonzero(counts))})
        g.step(a)
    return moves


def render_game(writer, renderer, brain, moves, seed, caption, sub, tail_frames=45):
    g = Snake(seed=seed)
    for m in moves:
        chunks = np.array_split(np.arange(len(m['spikes'])), FRAMES_PER_MOVE)
        for ch in chunks:
            spk = np.concatenate([m['spikes'][i] for i in ch]) if len(ch) else np.zeros(0, int)
            renderer.map.add_spikes(spk)
            writer.append_data(renderer.frame(g, m, caption, sub))
        g.step(m['action'])
    end = f"{'crashed into ' + g.death if g.death in ('wall', 'itself') else g.death or 'end'}  ·  score {g.score}"
    last = dict(moves[-1])
    for i in range(tail_frames):
        renderer.map.add_spikes(np.zeros(0, int))
        f = renderer.frame(g, last, caption, sub)
        img = Image.fromarray(f)
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([230, 470, 830, 580], 18, fill=PANEL)
        fnt = font(38, True)
        d.text((530 - d.textlength(end, font=fnt) / 2, 502), end, font=fnt, fill=AMBER)
        writer.append_data(np.array(img))
    return g


def pick_seed(brain, policy, seeds, lo, hi, max_moves):
    best = None
    for seed in seeds:
        t = time.time()
        moves = play_live(brain, policy, seed, max_moves)
        g = Snake(seed=seed)
        for m in moves:
            g.step(m['action'])
        print(f'  seed {seed}: score {g.score}, {len(moves)} moves ({time.time() - t:.0f}s)', flush=True)
        if lo <= g.score <= hi:
            return seed, moves
        if best is None or g.score > best[2]:
            best = (seed, moves, g.score)
    return best[0], best[1]


def main():
    os.makedirs('videos', exist_ok=True)
    W, neurons = load()
    brain = Brain(stabilise(W, neurons), neurons, seed=7)
    dec = dict(np.load('results/decoder_fly.npz'))
    board = json.load(open('results/scoreboard.json'))
    renderer = Renderer(neurons, brain)

    print('choosing games...')
    seed_w, moves_w = pick_seed(brain, wired_policy(brain), range(0, 12), 5, 12, 160)
    seed_d, moves_d = pick_seed(brain, decode_policy(dec, brain), range(0, 12), 30, 45, 700)

    out = 'videos/fly_plays_snake.mp4'
    writer = imageio.get_writer(out, fps=FPS, quality=8, macro_block_size=1)
    n_neurons = f"{len(neurons):,}"
    title = card(['A fruit fly brain plays Snake',
                  f'MaleCNS connectome  ·  {n_neurons} neurons  ·  25.6 million connections  ·  simulated live',
                  'Each move: the snake\'s eyes stimulate real visual neurons, the whole nervous system runs for 80 ms,',
                  'and the move is read from the neurons that carry commands from brain to body.'],
                 [64, 30, 26, 26], [TEXT, AMBER, MUTED, MUTED])
    for _ in range(FPS * 5):
        writer.append_data(title)

    part1 = card(['Part 1  ·  wiring only, zero training',
                  'Fruit on the left eye  →  left turning neurons (DNa01/DNa02) fire  →  turn left.',
                  'This comes straight from the connectome: the circuit male flies use to chase a female.',
                  'Nothing in that circuit knows about walls.'],
                 [52, 28, 28, 28], [TEXT, GREEN, MUTED, MUTED])
    for _ in range(FPS * 4):
        writer.append_data(part1)
    render_game(writer, renderer, brain, moves_w, seed_w,
                'Wiring only · no training', 'move = which turning neurons fired more (left vs right)')

    part2 = card(['Part 2  ·  same brain + a learned readout',
                  'A linear readout learned to interpret all 1,314 descending neurons.',
                  'The brain itself is not trained. Only the translator from neurons to moves is.'],
                 [52, 28, 28], [TEXT, CYAN, MUTED])
    for _ in range(FPS * 4):
        writer.append_data(part2)
    render_game(writer, renderer, brain, moves_d, seed_d,
                'Fly brain + readout', 'move = linear readout of 1,314 descending neurons', tail_frames=60)

    rows = [('real fly wiring + readout', 'fly brain, all 1314 descending neurons + decoder', AMBER),
            ('scrambled wiring + readout', 'scrambled brain, all 1314 descending neurons + decoder', MUTED),
            ('real fly wiring, no training', 'fly brain, steering neurons only (no training)', GREEN),
            ('scrambled wiring, no training', 'scrambled brain, steering neurons only (no training)', MUTED),
            ('random moves', 'random moves', MUTED)]

    def chart(d):
        x0, y0, wmax = 760, 330, 900
        vmax = max(board[k]['mean_score'] for _, k, _ in rows)
        for i, (lab, k, col) in enumerate(rows):
            y = y0 + i * 70
            v = board[k]['mean_score']
            f = font(28)
            d.text((x0 - 30 - d.textlength(lab, font=f), y + 6), lab, font=f, fill=TEXT)
            d.rounded_rectangle([x0, y, x0 + max(8, int(wmax * v / vmax)), y + 44], 8, fill=col)
            d.text((x0 + max(8, int(wmax * v / vmax)) + 16, y + 6), f'{v:.1f}', font=font(28, True), fill=TEXT)
    final = card(['Average score over 100 games', '', '', '', '', '', '', '', '', '',
                  'Scrambled = same neurons and synapse counts, connections rewired at random.',
                  'The real wiring carries what the eyes see to the body. Random wiring mostly loses it.'],
                 [52] + [28] * 11, [TEXT] + [MUTED] * 11, extra=chart)
    for _ in range(FPS * 8):
        writer.append_data(final)
    writer.close()
    print('wrote', out)


if __name__ == '__main__':
    main()
