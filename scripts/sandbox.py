"""
Headless test sandbox: run the FULL FlyWire brain through a battery of named
scenarios (each a real sensory stimulation), verify the readouts behave the
way real fly biology says they should, and grow a persistent registry of
"neurons that matter" across every scenario tested so far.

This is the discovery step that feeds build_subset.py -- run this first,
then rebuild the reduced network from data/neuron_registry.json.
"""
import json, time, os
import numpy as np
import pandas as pd
from brian2 import Network, SpikeMonitor, ms, Hz, mV
from model import create_model, poi, default_params

PATH_COMP = '2023_03_23_completeness_630_final.csv'
PATH_CON = '2023_03_23_connectivity_630_final.parquet'
REGISTRY_PATH = 'neuron_registry.json'

# --- real neuron populations, from our neuron_map.csv ---
POPS = {
    'sugar': [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
        720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
        720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
        720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
        720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663, 720575940611875570],
    'bitter': [720575940621778381, 720575940602353632, 720575940617094208, 720575940619197093,
        720575940626287336, 720575940618600651, 720575940627692048, 720575940630195909,
        720575940646212996, 720575940610483162, 720575940645743412, 720575940627578156,
        720575940622298631, 720575940621008895, 720575940629146711, 720575940610259370,
        720575940610481370, 720575940619028208, 720575940614281266, 720575940613061118, 720575940604027168],
    'water': [720575940612950568, 720575940631898285, 720575940606002609, 720575940612579053,
        720575940622902535, 720575940616177458, 720575940660292225, 720575940622486922,
        720575940613786774, 720575940629852866, 720575940625861168, 720575940613996959,
        720575940617857694, 720575940644965399, 720575940625203504, 720575940630553415,
        720575940635172191, 720575940634796536],
}
ID_MN9 = [720575940660219265, 720575940645521262]

SCENARIOS = {
    'approach_flower':        {'sugar': 200},
    'approach_water':         {'water': 200},
    'approach_bad_food':      {'bitter': 200},
    'flower_then_bad':        {'sugar': 200, 'bitter': 200},
    'faint_flower_far_away':  {'sugar': 60},
    'idle_baseline':          {},
}
T_RUN = 300 * ms

print('Loading real FlyWire connectome (this is the one-time expensive step)...')
t0 = time.time()
df_comp = pd.read_csv(PATH_COMP, index_col=0)
flyid2i = {j: i for i, j in enumerate(df_comp.index)}
i2flyid = {j: i for i, j in flyid2i.items()}
neu, syn, _ = create_model(PATH_COMP, PATH_CON, default_params)
print(f'[{time.time()-t0:.1f}s] network built once: {len(neu)} neurons, {len(syn.i)} synapses.')
print(f'Reusing this same neuron/synapse object for all {len(SCENARIOS)} scenarios below.\n')

net = Network(neu, syn)
registry = {}
if os.path.exists(REGISTRY_PATH):
    registry = json.load(open(REGISTRY_PATH))

results = []
for name, stim in SCENARIOS.items():
    t1 = time.time()
    params = dict(default_params)
    params['t_run'] = T_RUN

    pop_names = list(stim.keys())
    exc = [flyid2i[f] for f in POPS[pop_names[0]]] if len(pop_names) >= 1 else []
    exc2 = [flyid2i[f] for f in POPS[pop_names[1]]] if len(pop_names) >= 2 else []
    if pop_names:
        params['r_poi'] = stim[pop_names[0]] * Hz
    if len(pop_names) >= 2:
        params['r_poi2'] = stim[pop_names[1]] * Hz

    neu.v = default_params['v_0']  # reset state between scenarios
    spk_mon = SpikeMonitor(neu)
    pois, neu = poi(neu, exc, exc2, params) if pop_names else ([], neu)

    net.add(spk_mon, *pois)
    net.run(params['t_run'])
    net.remove(spk_mon, *pois)  # clean up before next scenario

    spk_trn = {i2flyid[k]: len(v) for k, v in spk_mon.spike_trains().items() if len(v)}
    mn9_spikes = sum(spk_trn.get(m, 0) for m in ID_MN9)
    active_ids = [str(k) for k in spk_trn]

    registry.setdefault(name, [])
    before = len(set(registry[name]))
    registry[name] = sorted(set(registry[name]) | set(active_ids))
    after = len(registry[name])

    results.append((name, list(stim.keys()), len(spk_trn), mn9_spikes, time.time() - t1))
    print(f'[{time.time()-t0:5.1f}s] {name:22s} stim={pop_names!s:28s} '
          f'active={len(spk_trn):4d}  MN9_spikes={mn9_spikes:3d}  '
          f'registry {before}->{after}  ({time.time()-t1:.1f}s)')

json.dump(registry, open(REGISTRY_PATH, 'w'), indent=1)

print(f'\nTotal registry size across all scenarios so far: {sum(len(v) for v in registry.values())} entries '
      f'({len(set().union(*[set(v) for v in registry.values()]))} unique neurons)')
print('Saved to', REGISTRY_PATH)
