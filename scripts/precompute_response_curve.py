"""
Run the (subset) brain at several stimulation frequencies to get a real
distance-to-response curve, so the visual sandbox can look up actual brain
behavior instantly instead of re-simulating every frame.
"""
import json, time
import pandas as pd
from brian2 import Network, SpikeMonitor, ms, Hz
from model import create_model, poi, default_params

PATH_COMP = 'completeness_subset.csv'   # the 534-neuron reduced network
PATH_CON = 'connectivity_subset.parquet'
ID_MN9 = [720575940660219265, 720575940645521262]
POP_SUGAR = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663, 720575940611875570]

df_comp = pd.read_csv(PATH_COMP, index_col=0)
flyid2i = {j: i for i, j in enumerate(df_comp.index)}
neu, syn, _ = create_model(PATH_COMP, PATH_CON, default_params)
net = Network(neu, syn)

curve = {}
for freq in [0, 40, 80, 120, 160, 200]:
    t1 = time.time()
    params = dict(default_params)
    params['t_run'] = 300 * ms
    params['r_poi'] = freq * Hz
    neu.v = default_params['v_0']
    spk_mon = SpikeMonitor(neu)
    exc = [flyid2i[f] for f in POP_SUGAR if f in flyid2i]
    pois, neu = poi(neu, exc, [], params) if freq > 0 else ([], neu)
    net.add(spk_mon, *pois)
    net.run(params['t_run'])
    net.remove(spk_mon, *pois)
    mn9 = sum(len(spk_mon.spike_trains().get(flyid2i[m], [])) for m in ID_MN9 if m in flyid2i)
    curve[freq] = mn9
    print(f'sugar freq={freq:3d}Hz -> MN9 spikes={mn9:2d}  ({time.time()-t1:.1f}s)')

json.dump(curve, open('mn9_response_curve.json', 'w'), indent=1)
print('saved mn9_response_curve.json')
