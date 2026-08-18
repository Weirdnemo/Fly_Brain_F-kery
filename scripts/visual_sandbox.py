"""
Visual sandbox: a bee sprite moves around a 2D world containing a flower
(sugar), water, and bad food (bitter). Locomotion is ordinary game-AI (move
toward the nearest attractive target) -- NOT brain-driven, per our design
decision. The brain's only job is the feed decision: as the bee gets closer
to the flower, taste-neuron stimulation frequency rises (mapped from real
distance), looked up against the precomputed MN9 response curve, and once
MN9's rate crosses a threshold, the bee actually feeds. This is the real
FlyWire proboscis-extension-reflex circuit driving one concrete decision.

Run locally with a real window:  python visual_sandbox.py
(Runs headless here to produce a preview GIF instead.)
"""
import json, math, os, sys
import pygame

W, H = 800, 500
FLOWER = (620, 130)
WATER = (150, 400)
BAD_FOOD = (620, 400)
BEE_SPEED = 2.2
SENSE_RANGE = 260
FEED_MN9_THRESHOLD = 20

curve = json.load(open('mn9_response_curve.json', 'r'))
curve = {int(k): v for k, v in curve.items()}
freqs_sorted = sorted(curve.keys())

def mn9_response(freq):
    """Linear-interpolate the precomputed real MN9-vs-frequency curve."""
    if freq <= freqs_sorted[0]:
        return curve[freqs_sorted[0]]
    if freq >= freqs_sorted[-1]:
        return curve[freqs_sorted[-1]]
    for i in range(len(freqs_sorted) - 1):
        f0, f1 = freqs_sorted[i], freqs_sorted[i + 1]
        if f0 <= freq <= f1:
            t = (freq - f0) / (f1 - f0)
            return curve[f0] + t * (curve[f1] - curve[f0])

def dist_to_freq(d, max_range=SENSE_RANGE, max_freq=200):
    if d > max_range:
        return 0
    return max_freq * (1 - d / max_range)

pygame.init()
screen = pygame.display.set_mode((W, H))
font = pygame.font.SysFont('monospace', 14)

bee = [400.0, 300.0]
state = 'idle'
frames_fed = 0
frame_log = []
snapshots = []

for frame in range(240):
    dx_f, dy_f = FLOWER[0] - bee[0], FLOWER[1] - bee[1]
    dist_flower = math.hypot(dx_f, dy_f)
    freq = dist_to_freq(dist_flower)
    mn9 = mn9_response(freq)

    if state != 'feeding':
        if dist_flower < 20:
            state = 'feeding' if mn9 >= FEED_MN9_THRESHOLD else 'idle'
        else:
            state = 'approaching_flower'
            bee[0] += BEE_SPEED * dx_f / dist_flower
            bee[1] += BEE_SPEED * dy_f / dist_flower
    else:
        frames_fed += 1
        if frames_fed > 30:
            state = 'idle'
            frames_fed = 0
            bee[0], bee[1] = 400.0, 300.0
    frame_log.append((frame, round(dist_flower, 1), round(freq, 1), round(mn9, 1), state))

    screen.fill((235, 245, 230))
    pygame.draw.circle(screen, (220, 130, 40), FLOWER, 14)
    pygame.draw.circle(screen, (60, 140, 220), WATER, 14)
    pygame.draw.circle(screen, (140, 60, 60), BAD_FOOD, 14)
    bee_color = (240, 190, 30) if state != 'feeding' else (255, 60, 60)
    pygame.draw.circle(screen, bee_color, (int(bee[0]), int(bee[1])), 9)
    txt = f'dist={dist_flower:5.0f}  stim={freq:5.1f}Hz  MN9~{mn9:4.1f}  {state}'
    screen.blit(font.render(txt, True, (20, 20, 20)), (10, 10))
    if frame % 4 == 0:
        snapshots.append(pygame.surfarray.array3d(screen).transpose(1, 0, 2).copy())

print('sample of frame log:')
for row in frame_log[::30]:
    print(f'  frame {row[0]:3d}  dist={row[1]:6.1f}  stim={row[2]:6.1f}Hz  MN9~{row[3]:5.1f}  state={row[4]}')

import numpy as np
from PIL import Image
imgs = [Image.fromarray(s) for s in snapshots]
imgs[0].save('visual_sandbox_preview.gif', save_all=True, append_images=imgs[1:], duration=60, loop=0)
print(f'saved visual_sandbox_preview.gif ({len(imgs)} frames)')
