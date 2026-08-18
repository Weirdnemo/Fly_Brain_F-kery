"""
Interactive sandbox: LEFT-CLICK moves the flower (sugar), RIGHT-CLICK moves
the water, MIDDLE-CLICK moves the bad food (bitter). The bee chases whichever
real target currently gives the higher curve-predicted MN9 response, and is
pushed away from bad food proportional to real bitter-GRN stimulation
strength -- that repulsion is scripted game logic using a real biological
quantity as input, NOT a decoded brain output (we don't have a real
aversion/avoidance descending neuron identified yet).

Press L to pause and run one REAL, live two-population Brian2 simulation
(dominant attractor + bitter, mirroring the earlier flower_then_bad test)
and compare its true MN9 count against the curve's prediction. This blocks
for several seconds -- it's an on-demand sanity check, not the main loop.

Run normally for a real, unbounded interactive window:
    python interactive_sandbox.py
Run with --preview to generate a short headless GIF (dev/testing only):
    python interactive_sandbox.py --preview
"""
import json, math, os, sys, time
PREVIEW = '--preview' in sys.argv
if PREVIEW:
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
import pygame

W, H = 800, 500
BEE_SPEED = 3.0
SENSE_RANGE = 300
REPEL_RANGE = 180
REPEL_STRENGTH = 2.5

PATH_COMP = 'completeness_subset.csv'
PATH_CON = 'connectivity_subset.parquet'
ID_MN9 = [720575940660219265, 720575940645521262]
POPS = {
    'sugar': [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
        720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
        720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
        720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
        720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663, 720575940611875570],
    'water': [720575940612950568, 720575940631898285, 720575940606002609, 720575940612579053,
        720575940622902535, 720575940616177458, 720575940660292225, 720575940622486922,
        720575940613786774, 720575940629852866, 720575940625861168, 720575940613996959,
        720575940617857694, 720575940644965399, 720575940625203504, 720575940630553415,
        720575940635172191, 720575940634796536],
    'bitter': [720575940621778381, 720575940602353632, 720575940617094208, 720575940619197093,
        720575940626287336, 720575940618600651, 720575940627692048, 720575940630195909,
        720575940646212996, 720575940610483162, 720575940645743412, 720575940627578156,
        720575940622298631, 720575940621008895, 720575940629146711, 720575940610259370,
        720575940610481370, 720575940619028208, 720575940614281266, 720575940613061118, 720575940604027168],
}

curves = json.load(open('mn9_response_curves.json'))
curves = {name: {int(k): v for k, v in c.items()} for name, c in curves.items()}

def mn9_response(pop, freq):
    curve = curves[pop]
    freqs = sorted(curve.keys())
    if freq <= freqs[0]:
        return curve[freqs[0]]
    if freq >= freqs[-1]:
        return curve[freqs[-1]]
    for i in range(len(freqs) - 1):
        f0, f1 = freqs[i], freqs[i + 1]
        if f0 <= freq <= f1:
            t = (freq - f0) / (f1 - f0)
            return curve[f0] + t * (curve[f1] - curve[f0])

def dist_to_freq(d, max_range=SENSE_RANGE, max_freq=200):
    return 0 if d > max_range else max_freq * (1 - d / max_range)

def run_live_check(dominant_pop, freq_dominant, freq_bitter):
    """On-demand REAL Brian2 run -- rebuilds the subset network fresh."""
    from brian2 import Network, ms, Hz
    from model import create_model, poi, default_params
    import pandas as pd
    df_comp = pd.read_csv(PATH_COMP, index_col=0)
    flyid2i = {j: i for i, j in enumerate(df_comp.index)}
    params = dict(default_params)
    params['t_run'] = 300 * ms
    params['r_poi'] = freq_dominant * Hz
    params['r_poi2'] = freq_bitter * Hz
    neu, syn, spk_mon = create_model(PATH_COMP, PATH_CON, params)
    exc = [flyid2i[f] for f in POPS[dominant_pop] if f in flyid2i]
    exc2 = [flyid2i[f] for f in POPS['bitter'] if f in flyid2i] if freq_bitter > 0 else []
    pois, neu = poi(neu, exc, exc2, params)
    net = Network(neu, syn, spk_mon, *pois)
    net.run(params['t_run'])
    return sum(len(spk_mon.spike_trains().get(flyid2i[m], [])) for m in ID_MN9 if m in flyid2i)

pygame.init()
screen = pygame.display.set_mode((W, H))
font = pygame.font.SysFont('monospace', 14)
clock = pygame.time.Clock()

flower = [600.0, 150.0]
water = [200.0, 400.0]
bad_food = [600.0, 420.0]
bee = [400.0, 300.0]
running = True
snapshots = []
frame = 0
live_result = None       # (predicted, real, dominant_pop) after an L-press
live_message = ''
scripted_events = {30: ('flower', 150, 400), 90: ('water', 650, 150), 130: ('bad', 250, 300), 160: ('live', None, None)}

while running and (not PREVIEW or frame < 220):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                flower = list(event.pos)
            elif event.button == 3:
                water = list(event.pos)
            elif event.button == 2:
                bad_food = list(event.pos)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                bee = [400.0, 300.0]
            elif event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_l:
                scripted_events[frame] = ('live', None, None)  # trigger on real keypress too

    if PREVIEW and frame in scripted_events:
        target, x, y = scripted_events[frame]
        if target == 'live':
            pass  # handled below
        else:
            (flower if target == 'flower' else water if target == 'water' else bad_food)[:] = [x, y]

    d_flower = math.hypot(flower[0] - bee[0], flower[1] - bee[1])
    d_water = math.hypot(water[0] - bee[0], water[1] - bee[1])
    d_bad = math.hypot(bad_food[0] - bee[0], bad_food[1] - bee[1])
    f_flower = dist_to_freq(d_flower)
    f_water = dist_to_freq(d_water)
    f_bad = dist_to_freq(d_bad, max_range=REPEL_RANGE)
    mn9_flower = mn9_response('sugar', f_flower)
    mn9_water = mn9_response('water', f_water)

    if mn9_flower >= mn9_water and mn9_flower > 0.5:
        target, dx, dy, d, dom_pop, dom_freq = flower, flower[0] - bee[0], flower[1] - bee[1], d_flower, 'sugar', f_flower
    elif mn9_water > 0:
        target, dx, dy, d, dom_pop, dom_freq = water, water[0] - bee[0], water[1] - bee[1], d_water, 'water', f_water
    else:
        target, dx, dy, d, dom_pop, dom_freq = flower, flower[0] - bee[0], flower[1] - bee[1], d_flower, 'sugar', f_flower

    move_x, move_y = 0.0, 0.0
    if d > 12:
        move_x += BEE_SPEED * dx / d
        move_y += BEE_SPEED * dy / d
        state = 'seeking'
    else:
        state = 'feeding' if target is flower else 'drinking'

    if f_bad > 0 and d_bad > 1:
        rx, ry = (bee[0] - bad_food[0]) / d_bad, (bee[1] - bad_food[1]) / d_bad
        move_x += REPEL_STRENGTH * (f_bad / 200) * rx
        move_y += REPEL_STRENGTH * (f_bad / 200) * ry
        if f_bad > 100:
            state = 'avoiding bad food'

    bee[0] += move_x
    bee[1] += move_y

    if frame in scripted_events and scripted_events[frame][0] == 'live':
        live_message = 'computing real brain response...'
        predicted = mn9_response(dom_pop, dom_freq)
        real = run_live_check(dom_pop, dom_freq, f_bad)
        live_result = (round(predicted, 1), real, dom_pop)
        live_message = ''

    screen.fill((235, 245, 230))
    pygame.draw.circle(screen, (220, 130, 40), (int(flower[0]), int(flower[1])), 14)
    pygame.draw.circle(screen, (60, 140, 220), (int(water[0]), int(water[1])), 14)
    pygame.draw.circle(screen, (140, 60, 60), (int(bad_food[0]), int(bad_food[1])), 14)
    bee_color = (255, 60, 60) if state in ('feeding', 'drinking') else \
                (120, 60, 200) if state == 'avoiding bad food' else (240, 190, 30)
    pygame.draw.circle(screen, bee_color, (int(bee[0]), int(bee[1])), 9)
    lines = [
        f'flower: dist={d_flower:5.0f} stim={f_flower:5.1f}Hz MN9~{mn9_flower:4.1f}',
        f'water:  dist={d_water:5.0f} stim={f_water:5.1f}Hz MN9~{mn9_water:4.1f}',
        f'bad food: dist={d_bad:5.0f} stim={f_bad:5.1f}Hz (repulsion only, not a decoded brain output)',
        f'state: {state}    L = run real brain check (blocks a few seconds)',
    ]
    if live_result:
        lines.append(f'last live check ({live_result[2]}): curve predicted {live_result[0]} spikes, real brain gave {live_result[1]} spikes')
    if live_message:
        lines.append(live_message)
    for i, line in enumerate(lines):
        screen.blit(font.render(line, True, (20, 20, 20)), (10, 10 + i * 18))

    if PREVIEW and frame % 3 == 0:
        snapshots.append(pygame.surfarray.array3d(screen).transpose(1, 0, 2).copy())
    frame += 1
    if not PREVIEW:
        pygame.display.flip()
    clock.tick(60)

if PREVIEW:
    from PIL import Image
    imgs = [Image.fromarray(s) for s in snapshots]
    imgs[0].save('interactive_sandbox_preview.gif', save_all=True, append_images=imgs[1:], duration=50, loop=0)
    print(f'saved interactive_sandbox_preview.gif ({len(imgs)} frames)')
    if live_result:
        print(f'live check result: {live_result}')
pygame.quit()
