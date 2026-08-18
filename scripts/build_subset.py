import json, pandas as pd, time
t0 = time.time()

id_mn9_pair = [720575940660219265, 720575940645521262]
registry = json.load(open('neuron_registry.json'))
keep = set(id_mn9_pair)
for scenario, ids in registry.items():
    keep.update(int(i) for i in ids)

print(f'keep-set size (union across {len(registry)} scenarios + MN9): {len(keep)} neurons '
      f'({len(keep)/127400*100:.2f}% of full network)')

df_comp = pd.read_csv('2023_03_23_completeness_630_final.csv', index_col=0)
df_comp_sub = df_comp[df_comp.index.isin(keep)]
df_comp_sub.to_csv('completeness_subset.csv')
print(f'[{time.time()-t0:.1f}s] completeness subset: {len(df_comp_sub)} rows')

con = pd.read_parquet('2023_03_23_connectivity_630_final.parquet')
con_sub = con[con['Presynaptic_ID'].isin(keep) & con['Postsynaptic_ID'].isin(keep)].copy()
new_flyid2i = {fid: i for i, fid in enumerate(df_comp_sub.index)}
con_sub['Presynaptic_Index'] = con_sub['Presynaptic_ID'].map(new_flyid2i)
con_sub['Postsynaptic_Index'] = con_sub['Postsynaptic_ID'].map(new_flyid2i)
con_sub.to_parquet('connectivity_subset.parquet')
print(f'[{time.time()-t0:.1f}s] connectivity subset: {len(con_sub)} synapses (full: {len(con)})')
