"""
Interactive visualizer for the FlyWire whole-brain LIF model.

Run from inside the Drosophila_brain_model repo directory:
    python visualizer.py
Then open http://127.0.0.1:8050 in a browser.

Pick one or two neuron populations to stimulate (built-in sugar/bitter/water/Ir94e
sets, any of the 106 named SEZ cell types, or paste your own FlyWire IDs), set
their stimulation frequency, hit Run, and get an interactive raster + activity
bar chart back. Each click rebuilds and reruns the model fresh (a few seconds on
a modern CPU once Brian2's compiled-code cache is warm) -- this is a
click-and-wait tool, not a live-dragging-slider animation, because a full
127k-neuron network run isn't real-time.
"""
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback_context

from model import create_model, poi, default_params
from brian2 import Network, ms, Hz

PATH_COMP = '2023_03_23_completeness_630_final.csv'
PATH_CON = '2023_03_23_connectivity_630_final.parquet'

# ---- built-in real neuron populations, no Codex lookup needed ----
BUILTIN = {
    'sugar (GRN, sweet taste)': [720575940624963786, 720575940630233916, 720575940637568838,
        720575940638202345, 720575940617000768, 720575940630797113, 720575940632889389,
        720575940621754367, 720575940621502051, 720575940640649691, 720575940639332736,
        720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543,
        720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239,
        720575940629176663, 720575940611875570],
    'bitter (GRN, bitter taste)': [720575940621778381, 720575940602353632, 720575940617094208,
        720575940619197093, 720575940626287336, 720575940618600651, 720575940627692048,
        720575940630195909, 720575940646212996, 720575940610483162, 720575940645743412,
        720575940627578156, 720575940622298631, 720575940621008895, 720575940629146711,
        720575940610259370, 720575940610481370, 720575940619028208, 720575940614281266,
        720575940613061118, 720575940604027168],
    'water (GRN, hygrosensation)': [720575940612950568, 720575940631898285, 720575940606002609,
        720575940612579053, 720575940622902535, 720575940616177458, 720575940660292225,
        720575940622486922, 720575940613786774, 720575940629852866, 720575940625861168,
        720575940613996959, 720575940617857694, 720575940644965399, 720575940625203504,
        720575940630553415, 720575940635172191, 720575940634796536],
    'Ir94e (chemosensory)': [720575940614211295, 720575940638218173, 720575940628832256,
        720575940626016017, 720575940621375231, 720575940612920386, 720575940614273292,
        720575940628198503, 720575940626241636, 720575940619387814, 720575940624604560,
        720575940615274425, 720575940610683315, 720575940627265265, 720575940624079544,
        720575940629211607, 720575940615089369, 720575940631082124],
}
with open('sez_neurons.pickle', 'rb') as f:
    SEZ_TYPES = pickle.load(f)  # 106 named SEZ cell types -> FlyWire IDs
for name, ids in SEZ_TYPES.items():
    BUILTIN[f'SEZ: {name}'] = ids

READOUT_NAME = 'MN9 (proboscis motor neuron)'
ID_MN9 = 720575940660219265

POP_OPTIONS = [{'label': k, 'value': k} for k in sorted(BUILTIN.keys())]

print('Loading completeness table (neuron ID <-> model index)...')
df_comp = pd.read_csv(PATH_COMP, index_col=0)
FLYID2I = {j: i for i, j in enumerate(df_comp.index)}
I2FLYID = {j: i for i, j in FLYID2I.items()}
print(f'Ready. {len(FLYID2I)} neurons available.')

app = Dash(__name__)
app.title = 'FlyWire brain visualizer'

app.layout = html.Div(style={'fontFamily': 'sans-serif', 'maxWidth': '1000px', 'margin': 'auto'}, children=[
    html.H2('FlyWire connectome \u2014 interactive stimulation'),
    html.Div([
        html.Div([
            html.Label('Population A (stimulated at Freq A)'),
            dcc.Dropdown(id='pop-a', options=POP_OPTIONS, value='sugar (GRN, sweet taste)'),
            html.Label('Freq A (Hz)'),
            dcc.Slider(id='freq-a', min=0, max=250, step=10, value=200,
                       marks={i: str(i) for i in range(0, 251, 50)}),
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.Label('Population B (optional, stimulated at Freq B)'),
            dcc.Dropdown(id='pop-b', options=[{'label': '(none)', 'value': ''}] + POP_OPTIONS, value=''),
            html.Label('Freq B (Hz)'),
            dcc.Slider(id='freq-b', min=0, max=250, step=10, value=0,
                       marks={i: str(i) for i in range(0, 251, 50)}),
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
    ]),
    html.Br(),
    html.Label('Trial duration (ms)'),
    dcc.Slider(id='t-run', min=100, max=1000, step=100, value=300,
               marks={i: str(i) for i in range(100, 1001, 200)}),
    html.Br(),
    html.Button('Run simulation', id='run-btn', n_clicks=0,
                style={'fontSize': '16px', 'padding': '8px 20px'}),
    html.Span(id='status', style={'marginLeft': '15px', 'color': '#555'}),
    dcc.Graph(id='raster-plot'),
    dcc.Graph(id='rate-plot'),
])


@app.callback(
    Output('raster-plot', 'figure'),
    Output('rate-plot', 'figure'),
    Output('status', 'children'),
    Input('run-btn', 'n_clicks'),
    State('pop-a', 'value'), State('freq-a', 'value'),
    State('pop-b', 'value'), State('freq-b', 'value'),
    State('t-run', 'value'),
    prevent_initial_call=True,
)
def run_and_plot(n_clicks, pop_a, freq_a, pop_b, freq_b, t_run):
    params = dict(default_params)
    params['t_run'] = t_run * ms
    params['r_poi'] = freq_a * Hz
    params['r_poi2'] = (freq_b or 0) * Hz

    ids_a = BUILTIN[pop_a]
    ids_b = BUILTIN[pop_b] if pop_b else []

    neu, syn, spk_mon = create_model(PATH_COMP, PATH_CON, params)
    exc = [FLYID2I[f] for f in ids_a if f in FLYID2I]
    exc2 = [FLYID2I[f] for f in ids_b if f in FLYID2I]
    pois, neu = poi(neu, exc, exc2, params)

    net = Network(neu, syn, spk_mon, *pois)
    net.run(params['t_run'])

    spk_trn = {I2FLYID[k]: np.asarray(v / ms) for k, v in spk_mon.spike_trains().items() if len(v)}

    set_a, set_b = set(ids_a), set(ids_b)
    other_ids = [k for k in spk_trn if k not in set_a and k not in set_b and k != ID_MN9]
    other_ids.sort(key=lambda k: len(spk_trn[k]), reverse=True)
    ordered = ([k for k in ids_a if k in spk_trn] + [k for k in ids_b if k in spk_trn]
               + ([ID_MN9] if ID_MN9 in spk_trn else []) + other_ids)

    fig_raster = go.Figure()
    groups = [(ids_a, 'Population A', '#D55E00'), (ids_b, 'Population B', '#0072B2'),
              ([ID_MN9], READOUT_NAME, '#009E73'), (other_ids, 'downstream', '#000000')]
    for ids, label, color in groups:
        xs, ys = [], []
        for nid in ids:
            if nid not in spk_trn:
                continue
            row = ordered.index(nid)
            for t in spk_trn[nid]:
                xs.append(t)
                ys.append(row)
        if xs:
            fig_raster.add_trace(go.Scattergl(x=xs, y=ys, mode='markers',
                marker=dict(size=4, color=color), name=label,
                hovertemplate='t=%{x:.1f} ms<extra>' + label + '</extra>'))
    fig_raster.update_layout(title='Spike raster', xaxis_title='Time (ms)',
        yaxis_title='Neuron (grouped by role)', yaxis_autorange='reversed',
        template='plotly_white', height=500)

    counts = sorted(((k, len(v)) for k, v in spk_trn.items()), key=lambda x: x[1], reverse=True)[:25]
    labels = [('A' if k in set_a else 'B' if k in set_b else 'MN9' if k == ID_MN9 else str(k)[-4:])
              for k, _ in counts]
    rates = [c / (t_run / 1000.0) for _, c in counts]
    fig_rate = go.Figure(go.Bar(x=labels, y=rates, marker_color='#555'))
    fig_rate.update_layout(title='Top 25 most active neurons', xaxis_title='Neuron',
        yaxis_title='Firing rate (Hz)', template='plotly_white', height=350)

    status = f'{len(spk_trn)} neurons fired. MN9: {len(spk_trn.get(ID_MN9, []))} spikes.'
    return fig_raster, fig_rate, status


if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=8050)
